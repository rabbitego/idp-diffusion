"""
Torus-native wrapped diffusion.

This implements a variance-preserving-style diffusion directly on the torus,
following the wrapped-diffusion / torsional-diffusion idea: instead of adding
Gaussian noise in an ambient Euclidean space and hoping samples stay near the
manifold, we add *wrapped* noise that is intrinsic to the circle. The forward
kernel for each angle is a wrapped normal whose width grows with the timestep;
in the high-noise limit it becomes uniform on the circle, which is exactly the
prior we sample from at generation time.

Design choices
--------------
* **x0 / angle-space parameterisation.** The network predicts the clean angles
  directly (an "x0-prediction" model), expressed as a 4-d (cos, sin) feature
  vector that we convert to angles. We found this far more stable than
  epsilon-prediction on a circle, because "the noise" is not well defined once
  wrapping has occurred (the winding number is unidentifiable). Predicting the
  destination side-steps that entirely and makes the Ramachandran regulariser
  trivial to apply, since it acts on a concrete predicted angle.

* **Cosine-style schedule.** ``alpha_bar`` follows the Nichol-Dhariwal cosine
  schedule. We translate it into an angular noise standard deviation via
  ``sigma(t) = sigma_max * sqrt(1 - alpha_bar(t))`` clamped so the top of the
  schedule is wide enough (several radians) to be effectively uniform on the
  circle.

* **Reverse step.** Given the network's predicted clean angle ``x0_hat`` at
  time ``t``, we form the posterior mean on the circle by taking a wrapped
  interpolation between ``x0_hat`` and the current ``x_t`` and adding wrapped
  posterior noise. All interpolation is done with :func:`angular_difference`
  so it follows the short arc.

The class is deliberately framework-light: it holds the schedule buffers and
exposes ``q_sample`` (forward), ``training_targets`` (what the loss needs), and
``p_sample`` / ``sample`` (reverse). The network is passed in, so this file has
no dependency on the specific Transformer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..utils.angles import (
    angles_to_features,
    angular_difference,
    features_to_angles,
    random_uniform_angles,
    wrap_to_pi,
    wrapped_normal_sample,
)


def cosine_alpha_bar(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol-Dhariwal cosine schedule for alpha_bar over ``num_steps``."""
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    f = torch.cos(((steps / num_steps) + s) / (1.0 + s) * (torch.pi / 2)) ** 2
    alpha_bar = f / f[0]
    return alpha_bar[1:].to(torch.float32)


@dataclass
class DiffusionConfig:
    num_steps: int = 1000
    sigma_max: float = 3.5  # radians; wide enough to be ~uniform on the circle
    sigma_min: float = 0.01
    schedule_s: float = 0.008


class WrappedDiffusion:
    """Wrapped (circular) diffusion with an x0-prediction parameterisation."""

    def __init__(self, config: DiffusionConfig, device: torch.device | str = "cpu"):
        self.config = config
        self.device = torch.device(device)

        alpha_bar = cosine_alpha_bar(config.num_steps, config.schedule_s).to(self.device)
        # Map the variance-preserving alpha_bar onto an angular sigma. At t=0
        # sigma ~ sigma_min (almost clean); at t=T sigma ~ sigma_max (~uniform).
        sigma = config.sigma_max * torch.sqrt(1.0 - alpha_bar)
        sigma = torch.clamp(sigma, min=config.sigma_min)

        self.alpha_bar = alpha_bar
        self.sigma = sigma

    # -- forward process -------------------------------------------------
    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x0) by adding wrapped noise of width sigma[t].

        Parameters
        ----------
        x0 : (B, L, 2) clean angles in radians.
        t  : (B,) long tensor of timestep indices.
        """
        sigma_t = self.sigma[t].view(-1, 1, 1)  # (B, 1, 1)
        return wrapped_normal_sample(x0, sigma_t.expand_as(x0), generator=generator)

    def training_targets(self, x0: torch.Tensor, t: torch.Tensor):
        """Return (x_t, x0) -- inputs and target for an x0-prediction model.

        We hand the network the *features* of x_t and ask it to predict the
        features of x0; the caller computes the angular loss. Returning x0 (not
        epsilon) keeps the winding-number ambiguity out of the objective.
        """
        x_t = self.q_sample(x0, t)
        return x_t, x0

    # -- reverse process -------------------------------------------------
    @torch.no_grad()
    def p_sample_step(
        self,
        x_t: torch.Tensor,
        t: int,
        x0_hat: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """One reverse step from x_t to x_{t-1} given predicted clean x0_hat.

        We interpolate along the short arc from ``x_t`` toward ``x0_hat`` by the
        fraction implied by the sigma schedule, then inject wrapped posterior
        noise sized by the remaining-step sigma. This is the circular analogue
        of the usual DDPM posterior mean/variance, written entirely in terms of
        :func:`angular_difference` so nothing crosses the +/-pi seam.
        """
        if t == 0:
            return wrap_to_pi(x0_hat)

        sigma_t = self.sigma[t]
        sigma_prev = self.sigma[t - 1]
        # Fraction of the way to move toward the clean estimate this step.
        # Uses the drop in angular variance between consecutive steps.
        frac = 1.0 - (sigma_prev ** 2) / (sigma_t ** 2)
        frac = float(torch.clamp(frac, 0.0, 1.0))

        delta = angular_difference(x0_hat, x_t)  # short arc x_t -> x0_hat
        mean = wrap_to_pi(x_t + frac * delta)

        posterior_sigma = sigma_prev * (1.0 - frac) ** 0.5
        return wrapped_normal_sample(
            mean, torch.full_like(mean, float(posterior_sigma)), generator=generator
        )

    @torch.no_grad()
    def sample(
        self,
        model,
        seq_embedding: torch.Tensor,
        mask: torch.Tensor,
        n_residues: int,
        generator: torch.Generator | None = None,
        return_trajectory: bool = False,
    ):
        """Generate angles for a batch by running the reverse chain.

        Parameters
        ----------
        model : callable mapping (features, t, seq_embedding, mask) -> feature
                prediction of shape (B, L, 4).
        seq_embedding : (B, L, D) per-residue conditioning (e.g. ESM).
        mask : (B, L) boolean, True at real residues.
        n_residues : L.
        """
        batch = seq_embedding.shape[0]
        x_t = random_uniform_angles(
            (batch, n_residues, 2), device=self.device, generator=generator
        )
        trajectory = []
        for step in reversed(range(self.config.num_steps)):
            t_batch = torch.full((batch,), step, device=self.device, dtype=torch.long)
            feats = angles_to_features(x_t)
            pred_feats = model(feats, t_batch, seq_embedding, mask)
            x0_hat = features_to_angles(pred_feats)
            x_t = self.p_sample_step(x_t, step, x0_hat, generator=generator)
            if return_trajectory and (step % max(1, self.config.num_steps // 50) == 0):
                trajectory.append(x_t.clone())
        x_t = wrap_to_pi(x_t)
        if return_trajectory:
            return x_t, trajectory
        return x_t
