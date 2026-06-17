"""
Losses: angular reconstruction loss + Ramachandran-KDE regulariser.

Two terms.

1. **Angular reconstruction loss.** The x0-prediction model outputs predicted
   (cos, sin) features; the target is the clean angle. We score agreement with
   ``1 - cos(delta)`` where ``delta`` is the short-arc angular error. This is
   the natural loss on a circle: it is smooth, periodic, minimised at delta=0,
   maximal at delta=pi, and has none of the seam artefacts of an MSE on raw
   radians. (Equivalently, up to scale, it is the squared chord length between
   the unit vectors, so it also equals an MSE on perfectly unit-norm features --
   but writing it as ``1 - cos`` means we never have to renormalise the target.)

2. **Ramachandran-KDE regulariser -- the project's central contribution.**
   We hold a precomputed kernel-density estimate of the empirical (phi, psi)
   distribution and add an auxiliary penalty equal to the negative
   log-density of the model's *predicted clean angles* under that KDE. This
   pulls generated angles toward populated regions of Ramachandran space.

   Two correctness requirements are enforced by construction here:

   * The KDE is evaluated per residue type. The :class:`RamachandranKDE` holds
     one density per residue class (general / GLY / PRO / PRE-PRO at minimum),
     and the regulariser indexes into them using a per-position class id. A
     single global density would blur the Gly/Pro/pre-Pro basins that are
     physically distinct, so we never collapse them.
   * The penalty is applied to the predicted *clean* angle x0_hat, not to the
     noised state, and is annealed by a caller-supplied weight so that data
     dominates late in training and the regulariser cannot by itself drive the
     ensemble into mode collapse.

   The KDE itself is built from disorder-appropriate statistics (PED-derived or
   disordered-region-derived), *not* from folded-protein Ramachandran maps --
   see ``scripts/build_ramachandran_kde.py``. Using a folded-protein prior here
   would bias the model away from the PPII-enriched IDP landscape it is meant to
   learn, so the KDE source is a deliberate, documented choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..utils.angles import angular_difference, features_to_angles
from ..constants import RAMA_CLASSES, RAMA_CLASS_TO_ID  # noqa: F401 (re-exported)

TWO_PI = 2.0 * math.pi


def angular_reconstruction_loss(
    pred_features: torch.Tensor, target_angles: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean ``1 - cos(delta)`` over phi and psi at masked positions.

    Parameters
    ----------
    pred_features : (B, L, 4) raw network output (need not be unit norm).
    target_angles : (B, L, 2) clean angles in radians.
    mask : (B, L) boolean, True at real residues.
    """
    pred_angles = features_to_angles(pred_features)  # (B, L, 2)
    delta = angular_difference(pred_angles, target_angles)  # (B, L, 2)
    per_angle = 1.0 - torch.cos(delta)  # (B, L, 2)
    per_residue = per_angle.mean(dim=-1)  # (B, L)
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    return (per_residue * mask_f).sum() / denom


@dataclass
class RamachandranKDEConfig:
    bandwidth_deg: float = 15.0  # von Mises-ish bandwidth, expressed in degrees
    grid_size: int = 72  # 5-degree bins over [-pi, pi] in each axis


class RamachandranKDE:
    """Per-residue-class density over (phi, psi), evaluated on a torus grid.

    The density for each class is stored as a normalised probability table on a
    ``grid_size x grid_size`` grid over (-pi, pi]^2. Lookups for arbitrary
    angles use bilinear interpolation with wrap-around in both axes, so the
    density is genuinely periodic. This grid representation is differentiable
    w.r.t. the query angle, which is what lets us back-propagate the regulariser
    into the network.

    Build one with :meth:`from_angle_samples` (used by the build script) or load
    precomputed tables with :meth:`load`.
    """

    def __init__(self, log_density_tables: torch.Tensor, config: RamachandranKDEConfig):
        # log_density_tables: (C, G, G), one log-prob grid per residue class.
        if log_density_tables.dim() != 3:
            raise ValueError("expected (C, G, G) log-density tables")
        self.log_tables = log_density_tables
        self.config = config
        self.n_classes = log_density_tables.shape[0]
        self.grid = log_density_tables.shape[-1]

    def to(self, device):
        self.log_tables = self.log_tables.to(device)
        return self

    @classmethod
    def from_angle_samples(
        cls,
        angles_by_class: dict[str, torch.Tensor],
        config: RamachandranKDEConfig | None = None,
    ) -> "RamachandranKDE":
        """Build KDE tables from empirical angle samples per class.

        ``angles_by_class`` maps a class name in :data:`RAMA_CLASSES` to a
        tensor of shape (N_c, 2) of (phi, psi) radians. Classes with no samples
        fall back to a uniform density so they never produce -inf penalties.
        """
        config = config or RamachandranKDEConfig()
        G = config.grid_size
        # Bin centres in radians.
        centres = torch.linspace(-math.pi, math.pi, G + 1)[:-1] + (math.pi / G)
        phi_c, psi_c = torch.meshgrid(centres, centres, indexing="ij")  # (G, G)
        # Concentration of a von Mises whose circular sd ~ bandwidth.
        bw = math.radians(config.bandwidth_deg)
        kappa = 1.0 / (bw ** 2)

        tables = []
        for name in RAMA_CLASSES:
            samples = angles_by_class.get(name)
            if samples is None or samples.numel() == 0:
                tables.append(torch.zeros(G, G))  # uniform in log space
                continue
            phi_s = samples[:, 0].view(-1, 1, 1)
            psi_s = samples[:, 1].view(-1, 1, 1)
            # Product of two von Mises kernels summed over samples.
            log_k = kappa * (
                torch.cos(phi_c.unsqueeze(0) - phi_s)
                + torch.cos(psi_c.unsqueeze(0) - psi_s)
            )
            dens = torch.logsumexp(log_k, dim=0)  # (G, G), unnormalised log dens
            dens = dens - math.log(samples.shape[0])
            tables.append(dens)
        log_tables = torch.stack(tables, dim=0)  # (C, G, G)
        # Normalise each class to sum to 1 over the grid (proper prob table).
        cell_area = (TWO_PI / G) ** 2
        log_norm = torch.logsumexp(
            log_tables.view(len(RAMA_CLASSES), -1), dim=-1
        ) + math.log(cell_area)
        log_tables = log_tables - log_norm.view(-1, 1, 1)
        return cls(log_tables, config)

    def save(self, path: str) -> None:
        torch.save(
            {"log_tables": self.log_tables.cpu(), "config": self.config.__dict__}, path
        )

    @classmethod
    def load(cls, path: str) -> "RamachandranKDE":
        blob = torch.load(path, map_location="cpu")
        return cls(blob["log_tables"], RamachandranKDEConfig(**blob["config"]))

    def log_prob(self, angles: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        """Differentiable log-density lookup with wrap-around bilinear interp.

        Parameters
        ----------
        angles : (..., 2) query (phi, psi) radians.
        class_ids : (...,) long tensor of residue-class ids selecting the table.
        """
        G = self.grid
        # Map [-pi, pi) to continuous grid coordinates [0, G).
        coords = (angles + math.pi) / TWO_PI * G  # (..., 2)
        c0 = torch.floor(coords).long()
        frac = coords - c0.float()
        c0 = c0 % G
        c1 = (c0 + 1) % G

        phi0, psi0 = c0[..., 0], c0[..., 1]
        phi1, psi1 = c1[..., 0], c1[..., 1]
        fphi, fpsi = frac[..., 0], frac[..., 1]

        tbl = self.log_tables[class_ids]  # (..., G, G) via advanced indexing
        # gather the four corners
        def corner(a, b):
            return torch.gather(
                torch.gather(tbl, -2, a.unsqueeze(-1).unsqueeze(-1).expand(*a.shape, 1, G)),
                -1,
                b.unsqueeze(-1).unsqueeze(-1).expand(*b.shape, 1, 1),
            ).squeeze(-1).squeeze(-1)

        v00 = corner(phi0, psi0)
        v01 = corner(phi0, psi1)
        v10 = corner(phi1, psi0)
        v11 = corner(phi1, psi1)
        # bilinear in log space (interpolating log-density is a smooth,
        # well-behaved proxy and keeps gradients finite)
        v0 = v00 * (1 - fpsi) + v01 * fpsi
        v1 = v10 * (1 - fpsi) + v11 * fpsi
        return v0 * (1 - fphi) + v1 * fphi

    def regularizer(
        self,
        pred_features: torch.Tensor,
        class_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean negative log-density of predicted clean angles at real residues.

        Returns a non-negative-ish scalar suitable for adding to the loss with a
        schedule weight. Lower means the predictions sit in populated
        Ramachandran regions for their residue class.
        """
        pred_angles = features_to_angles(pred_features)
        logp = self.log_prob(pred_angles, class_ids)  # (B, L)
        nll = -logp
        mask_f = mask.float()
        denom = mask_f.sum().clamp(min=1.0)
        return (nll * mask_f).sum() / denom
