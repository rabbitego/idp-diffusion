"""
Baselines for comparison.

A learned generative model that is only compared against its own ablation will
not pass review. Two external baselines are provided here.

1. **Residue-specific statistical coil (the field's null model).**
   The classic flexible-meccano / statistical-coil idea: sample each residue's
   (phi, psi) independently from an empirical, residue-type-specific
   distribution, with no coupling between residues and no sequence information
   beyond residue identity. This is the bar any learned model *must* clear -- if
   the Transformer cannot beat an independent residue-wise sampler on ensemble
   observables, that is a finding to confront early, which is why the run-order
   recommendation is to evaluate this first, not last.

   We implement it as sampling from the same per-class Ramachandran densities
   used by the regulariser (so the comparison is apples-to-apples on marginals),
   optionally refined to per-residue-identity tables when enough data exist.

2. **idpGAN interface (external learned baseline).**
   A thin adapter describing how to run idpGAN (Janson et al., Nat. Commun.
   2023) on the same held-out sequences and return angle ensembles in this
   repo's (n_conf, L, 2) convention, so the comparison flows through the same
   metrics. The actual idpGAN weights/code are fetched and run where there is
   network access; this class documents the contract and raises a clear message
   if invoked offline.
"""

from __future__ import annotations

import math

import numpy as np

from ..diffusion.losses import RAMA_CLASSES
from ..data.torsions import _residue_class_ids


class StatisticalCoilModel:
    """Independent residue-wise sampler from per-class Ramachandran densities.

    Build from the same KDE log-tables used in training (pass the
    ``RamachandranKDE`` instance). Sampling draws each residue's (phi, psi)
    independently from its class table by inverse-CDF on the flattened grid.
    """

    def __init__(self, rama_kde, seed: int = 0):
        self.kde = rama_kde
        self.rng = np.random.default_rng(seed)
        # Precompute per-class flattened probability tables and grid centres.
        log_tables = rama_kde.log_tables.cpu().numpy()  # (C, G, G)
        self.probs = []
        for c in range(log_tables.shape[0]):
            p = np.exp(log_tables[c] - log_tables[c].max())
            p = p.flatten()
            p = p / p.sum()
            self.probs.append(p)
        self.grid = rama_kde.grid
        centres = np.linspace(-math.pi, math.pi, self.grid + 1)[:-1] + (math.pi / self.grid)
        self.centres = centres
        self.bin_width = 2 * math.pi / self.grid

    def sample_ensemble(self, sequence: str, n_conf: int) -> np.ndarray:
        """Return (n_conf, L, 2) angles sampled independently per residue."""
        class_ids = _residue_class_ids(sequence)
        L = len(sequence)
        out = np.empty((n_conf, L, 2), dtype=np.float64)
        for i in range(L):
            p = self.probs[class_ids[i]]
            idx = self.rng.choice(len(p), size=n_conf, p=p)
            gi, gj = np.unravel_index(idx, (self.grid, self.grid))
            # centre + uniform jitter within the bin to avoid grid artefacts
            jitter = (self.rng.random((n_conf, 2)) - 0.5) * self.bin_width
            out[:, i, 0] = self.centres[gi] + jitter[:, 0]
            out[:, i, 1] = self.centres[gj] + jitter[:, 1]
        # wrap into (-pi, pi]
        out = (out + math.pi) % (2 * math.pi) - math.pi
        return out


class IDPGANBaseline:
    """Adapter contract for running idpGAN on held-out sequences.

    Expected use (on a networked machine):
        1. Clone idpGAN and install its deps.
        2. Point ``repo_dir`` / ``weights`` at the checkout.
        3. Call ``sample_ensemble(seq, n_conf)`` -> (n_conf, L, 2) radians.

    idpGAN natively emits Cartesian CA traces / its own internal representation;
    the adapter is responsible for converting to backbone (phi, psi) so results
    enter this repo's metrics unchanged. The conversion notes live in
    ``docs/baselines.md``.
    """

    def __init__(self, repo_dir: str | None = None, weights: str | None = None):
        self.repo_dir = repo_dir
        self.weights = weights

    def sample_ensemble(self, sequence: str, n_conf: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError(
            "Run idpGAN where you have its code, weights, and network access. "
            "Implement the conversion from idpGAN output to (n_conf, L, 2) phi/psi "
            "here following docs/baselines.md, then results flow through the same "
            "metrics as the main model."
        )
