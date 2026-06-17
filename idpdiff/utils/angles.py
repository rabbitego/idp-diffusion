"""
Angle and torus utilities.

All backbone torsion angles (phi, psi) live on the circle S^1, so the natural
domain for a residue's (phi, psi) pair is the 2-torus T^2 = S^1 x S^1, and a
full chain of L residues lives on T^(2L).

This module centralises every operation that must respect that geometry:
wrapping to (-pi, pi], the wrapped-normal distribution used by the diffusion
process, sin/cos featurisation for the network input, and conversion back to
angles. Keeping these in one place avoids the single most common bug in
torsion-space models: mixing a Euclidean operation (plain subtraction, plain
Gaussian noise, plain MSE) into a manifold that is not Euclidean.

Conventions
-----------
* Angles are stored in radians in the half-open interval (-pi, pi].
* A "feature" tensor is the 4-d (cos phi, sin phi, cos psi, sin psi) encoding
  used as network input; angle tensors carry the raw (phi, psi) radians.
* Tensors are PyTorch tensors of shape (..., 2) for angle pairs or (..., 4) for
  features, where the last axis is the only one with fixed meaning. Everything
  else (batch, residue) is broadcast over.
"""

from __future__ import annotations

import math

import torch

TWO_PI = 2.0 * math.pi


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles (radians) into the half-open interval (-pi, pi].

    This is the projection back onto the canonical fundamental domain of the
    circle. It is idempotent and safe to call liberally.
    """
    return torch.remainder(angles + math.pi, TWO_PI) - math.pi


def angular_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Signed smallest-arc difference a - b on the circle, in (-pi, pi].

    Plain ``a - b`` is wrong on a circle: the difference between 170 deg and
    -170 deg is +20 deg, not -340 deg. Every loss or metric that compares two
    angles must route through this function.
    """
    return wrap_to_pi(a - b)


def angles_to_features(angles: torch.Tensor) -> torch.Tensor:
    """Map (..., 2) angle pairs to (..., 4) (cos, sin) features.

    The ordering of the output last axis is
    ``[cos(phi), sin(phi), cos(psi), sin(psi)]``.
    """
    if angles.shape[-1] != 2:
        raise ValueError(f"expected last dim 2 (phi, psi), got {angles.shape[-1]}")
    phi = angles[..., 0]
    psi = angles[..., 1]
    return torch.stack(
        [torch.cos(phi), torch.sin(phi), torch.cos(psi), torch.sin(psi)],
        dim=-1,
    )


def features_to_angles(features: torch.Tensor) -> torch.Tensor:
    """Map (..., 4) (cos, sin) features back to (..., 2) angle pairs.

    ``atan2`` is used so the result is already wrapped into (-pi, pi] and is
    numerically stable even when the (cos, sin) pair is not exactly unit norm
    (as will be the case for raw network output before renormalisation).
    """
    if features.shape[-1] != 4:
        raise ValueError(f"expected last dim 4, got {features.shape[-1]}")
    phi = torch.atan2(features[..., 1], features[..., 0])
    psi = torch.atan2(features[..., 3], features[..., 2])
    return torch.stack([phi, psi], dim=-1)


def wrapped_normal_sample(
    mean: torch.Tensor, std: torch.Tensor | float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Sample from a wrapped normal: draw a Gaussian, then wrap to (-pi, pi].

    The wrapped normal WN(mean, std) is the pushforward of N(mean, std^2) under
    the wrapping map. For the small-to-moderate std used in the forward
    diffusion this is an excellent and cheap stand-in for the diffusion kernel
    on the circle (the wrapped/heat kernel), and sampling is exact regardless of
    std because wrapping a Gaussian sample is exact.
    """
    if isinstance(std, (int, float)):
        std = torch.full_like(mean, float(std))
    noise = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
    return wrap_to_pi(mean + std * noise)


def wrapped_normal_logprob(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor | float, n_wraps: int = 5
) -> torch.Tensor:
    """Log density of a wrapped normal at ``x``, summed over winding numbers.

    The wrapped-normal density is an infinite sum over integer shifts of 2*pi;
    in practice ``n_wraps`` terms on each side are far more than enough once std
    is below roughly pi. Used by the (optional) likelihood-style score checks
    and by the wrapped-diffusion reverse step, not by the main training loss.
    """
    if isinstance(std, (int, float)):
        std = torch.full_like(mean, float(std))
    var = std ** 2
    ks = torch.arange(-n_wraps, n_wraps + 1, device=x.device, dtype=x.dtype)
    # shape (..., 2*n_wraps + 1)
    diff = (x - mean).unsqueeze(-1) + TWO_PI * ks
    log_terms = -0.5 * (diff ** 2) / var.unsqueeze(-1) - 0.5 * torch.log(
        TWO_PI * var.unsqueeze(-1)
    )
    return torch.logsumexp(log_terms, dim=-1)


def random_uniform_angles(
    shape: tuple[int, ...], device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Uniform angles on (-pi, pi], the high-noise limit of wrapped diffusion.

    As the forward process std grows, the wrapped normal converges to the
    uniform distribution on the circle. The reverse process is therefore seeded
    from this distribution at sampling time.
    """
    u = torch.rand(shape, device=device, generator=generator)
    return (u * TWO_PI) - math.pi
