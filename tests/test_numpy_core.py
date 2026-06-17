"""Tests for the numpy-only components and core mathematical logic.

These run without torch (CI-friendly offline) and cover the parts most prone to
subtle bugs: the NeRF reconstruction, residue-class assignment, the metrics, and
the circular-geometry invariants that the wrapped diffusion relies on. The
torch-dependent modules are exercised by ``tests/test_torch_smoke.py`` (run on a
machine with torch installed).

Run:  python -m pytest tests/ -v      (or)      python tests/test_numpy_core.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from idpdiff.validation.reconstruct import (
    reconstruct_backbone, radius_of_gyration, end_to_end_distance,
    clash_count, BOND_LENGTHS,
)
from idpdiff.data.torsions import _dihedral, _residue_class_ids
from idpdiff.eval.metrics import per_residue_jsd, observable_distances, ensemble_diversity
from idpdiff.constants import RAMA_CLASS_TO_ID


def _angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def test_nerf_roundtrips_dihedrals():
    """Reconstruction must reproduce the dihedrals it was built from."""
    L = 30
    rng = np.random.default_rng(0)
    phi = rng.uniform(-math.pi, math.pi, L)
    psi = rng.uniform(-math.pi, math.pi, L)
    phi[0] = np.nan
    psi[-1] = np.nan
    coords = reconstruct_backbone(phi, psi)
    assert coords.shape == (L, 3, 3)
    assert np.isfinite(coords).all()
    for i in range(1, L):
        rp = _dihedral(coords[i - 1, 2], coords[i, 0], coords[i, 1], coords[i, 2])
        assert abs(_angdiff(rp, phi[i])) < 1e-6
    for i in range(L - 1):
        rs = _dihedral(coords[i, 0], coords[i, 1], coords[i, 2], coords[i + 1, 0])
        assert abs(_angdiff(rs, psi[i])) < 1e-6


def test_nerf_bond_lengths_ideal():
    L = 20
    rng = np.random.default_rng(1)
    coords = reconstruct_backbone(
        rng.uniform(-math.pi, math.pi, L), rng.uniform(-math.pi, math.pi, L)
    )
    for i in range(1, L):
        assert abs(np.linalg.norm(coords[i, 1] - coords[i, 0]) - BOND_LENGTHS["N_CA"]) < 1e-6
        assert abs(np.linalg.norm(coords[i, 2] - coords[i, 1]) - BOND_LENGTHS["CA_C"]) < 1e-6
        assert abs(np.linalg.norm(coords[i, 0] - coords[i - 1, 2]) - BOND_LENGTHS["C_N"]) < 1e-6


def test_extended_more_extended_than_helix():
    L = 30
    ext = reconstruct_backbone(np.full(L, math.radians(-139)), np.full(L, math.radians(135)))
    hel = reconstruct_backbone(np.full(L, math.radians(-60)), np.full(L, math.radians(-45)))
    assert radius_of_gyration(ext[:, 1, :]) > radius_of_gyration(hel[:, 1, :])


def test_residue_class_assignment():
    seq = "AGPKAP"
    ids = _residue_class_ids(seq)
    M = RAMA_CLASS_TO_ID
    expected = [M["general"], M["glycine"], M["proline"],
                M["general"], M["pre_proline"], M["proline"]]
    assert list(ids) == expected


def test_jsd_bounds():
    rng = np.random.default_rng(1)
    L, n = 20, 200
    A = rng.uniform(-math.pi, math.pi, (n, L, 2))
    assert per_residue_jsd(A, A.copy()).mean() < 0.05  # identical -> ~0
    P = np.full((n, L, 2), -2.0) + rng.normal(0, 0.05, (n, L, 2))
    Q = np.full((n, L, 2), 2.0) + rng.normal(0, 0.05, (n, L, 2))
    assert per_residue_jsd(P, Q).mean() > 0.9  # disjoint -> ~1


def test_diversity_ordering():
    rng = np.random.default_rng(2)
    L, n = 20, 200
    tight = np.full((n, L, 2), 0.3) + rng.normal(0, 0.02, (n, L, 2))
    broad = rng.uniform(-math.pi, math.pi, (n, L, 2))
    assert ensemble_diversity(tight) < ensemble_diversity(broad)


def test_angular_difference_seam():
    """The defining property: 3.0 and -3.0 rad are 0.283 apart, not 6.0."""
    d = _angdiff(np.array([-3.0]), np.array([3.0]))[0]
    assert abs(d - 0.28319) < 1e-4


def test_clash_count_detects_overlap():
    # Two points placed on top of each other (beyond seq sep) must register.
    ca = np.array([[0, 0, 0], [10, 0, 0], [0.1, 0, 0], [20, 0, 0], [0.2, 0, 0]], float)
    assert clash_count(ca, threshold=3.8, sep=2) > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} numpy-core tests passed")
