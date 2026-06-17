"""
Ensemble-fidelity metrics.

Generating angles is easy; demonstrating that the *ensemble* is right is the
hard part and is where an IDP paper is accepted or rejected. This module
provides the quantitative comparisons between a generated ensemble and a
held-out reference ensemble.

Metrics
-------
* **Per-residue phi-psi Jensen-Shannon divergence.** For each residue position
  we histogram (phi, psi) over the ensemble on a shared 2-D grid and compute the
  JSD between generated and reference. JSD (not KL) because it is symmetric,
  bounded in [0, 1] (log base 2), and finite even when one distribution has
  empty bins -- essential for small ensembles.

* **Global-observable distributional distance.** IDPs are defined by the
  *spread* of their ensemble, so we compare the full distributions of radius of
  gyration, end-to-end distance, and asphericity -- not just their means -- using
  the 1-Wasserstein distance. A model that matches mean Rg but collapses its
  variance is wrong in precisely the way that matters most for IDPs, and only a
  distributional metric catches that.

* **Diversity.** Mean pairwise angular distance within an ensemble, so the
  with/without-regulariser comparison can report whether the Ramachandran
  penalty improved marginals at the cost of collapsing diversity. Reported
  alongside fidelity, never instead of it.

All functions take plain numpy arrays so they can be used both inside training
(periodic validation) and in standalone analysis scripts.
"""

from __future__ import annotations

import math

import numpy as np

from ..validation.reconstruct import (
    end_to_end_distance,
    radius_of_gyration,
    reconstruct_backbone,
)


def _phi_psi_histogram(angles: np.ndarray, bins: int = 36) -> np.ndarray:
    """2-D histogram of (phi, psi) over [-pi, pi]^2, normalised to a pmf."""
    edges = np.linspace(-math.pi, math.pi, bins + 1)
    h, _, _ = np.histogram2d(angles[:, 0], angles[:, 1], bins=[edges, edges])
    total = h.sum()
    if total > 0:
        h = h / total
    return h


def _jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (log base 2) between two pmfs."""
    p = p.flatten() + eps
    q = q.flatten() + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        return float(np.sum(a * (np.log2(a) - np.log2(b))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def per_residue_jsd(
    gen_angles: np.ndarray, ref_angles: np.ndarray, bins: int = 36
) -> np.ndarray:
    """Per-position phi-psi JSD between generated and reference ensembles.

    Parameters
    ----------
    gen_angles : (n_gen, L, 2) generated ensemble angles (radians).
    ref_angles : (n_ref, L, 2) reference ensemble angles (radians).

    Returns
    -------
    (L,) array of JSD values, one per residue.
    """
    L = gen_angles.shape[1]
    out = np.empty(L)
    for i in range(L):
        pg = _phi_psi_histogram(gen_angles[:, i, :], bins)
        pr = _phi_psi_histogram(ref_angles[:, i, :], bins)
        out[i] = _jsd(pg, pr)
    return out


def _wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """1-D 1-Wasserstein distance via sorted-quantile differences."""
    a = np.sort(a)
    b = np.sort(b)
    n = max(len(a), len(b))
    qs = (np.arange(n) + 0.5) / n
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)
    return float(np.mean(np.abs(qa - qb)))


def _asphericity(ca: np.ndarray) -> float:
    """Asphericity from the gyration tensor of one conformer's CA coordinates."""
    c = ca - ca.mean(axis=0)
    gyr = (c[:, :, None] * c[:, None, :]).mean(axis=0)  # (3,3)
    w = np.linalg.eigvalsh(gyr)
    w = np.sort(w)
    return float(w[2] - 0.5 * (w[0] + w[1]))


def ensemble_observables(angles: np.ndarray) -> dict[str, np.ndarray]:
    """Reconstruct each conformer and return per-conformer global observables.

    ``angles`` is (n_conf, L, 2). Returns dict of arrays length n_conf for
    'rg', 'ree', 'asphericity'.
    """
    rg, ree, asph = [], [], []
    for k in range(angles.shape[0]):
        coords = reconstruct_backbone(angles[k, :, 0], angles[k, :, 1])
        ca = coords[:, 1, :]
        rg.append(radius_of_gyration(ca))
        ree.append(end_to_end_distance(ca))
        asph.append(_asphericity(ca))
    return {
        "rg": np.array(rg),
        "ree": np.array(ree),
        "asphericity": np.array(asph),
    }


def observable_distances(
    gen_angles: np.ndarray, ref_angles: np.ndarray
) -> dict[str, float]:
    """Wasserstein-1 distance between generated/reference observable distros."""
    g = ensemble_observables(gen_angles)
    r = ensemble_observables(ref_angles)
    return {k: _wasserstein1(g[k], r[k]) for k in g}


def ensemble_diversity(angles: np.ndarray, max_pairs: int = 2000) -> float:
    """Mean pairwise circular distance within an ensemble (radians).

    Subsamples conformer pairs for efficiency. Higher means a more diverse
    ensemble; used to detect regulariser-induced mode collapse.
    """
    n = angles.shape[0]
    rng = np.random.default_rng(0)
    dists = []
    pair_budget = min(max_pairs, n * (n - 1) // 2)
    for _ in range(pair_budget):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        d = 1.0 - np.cos(angles[i] - angles[j])  # (L, 2)
        dists.append(float(d.mean()))
    return float(np.mean(dists)) if dists else 0.0


def summarize(gen_angles: np.ndarray, ref_angles: np.ndarray) -> dict:
    """One-call summary bundling the headline metrics for a single protein."""
    jsd = per_residue_jsd(gen_angles, ref_angles)
    obs = observable_distances(gen_angles, ref_angles)
    return {
        "jsd_mean": float(jsd.mean()),
        "jsd_median": float(np.median(jsd)),
        "rg_wasserstein": obs["rg"],
        "ree_wasserstein": obs["ree"],
        "asphericity_wasserstein": obs["asphericity"],
        "gen_diversity": ensemble_diversity(gen_angles),
        "ref_diversity": ensemble_diversity(ref_angles),
    }
