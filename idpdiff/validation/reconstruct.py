"""
Reconstruct backbone Cartesian coordinates from torsion angles (NeRF).

Torsion-space generation produces (phi, psi) per residue but no 3-D structure.
Every global observable the paper needs -- radius of gyration, end-to-end
distance, SAXS profiles, NMR restraints -- requires Cartesian coordinates, so we
rebuild the N-CA-C backbone using the Natural Extension Reference Frame (NeRF)
algorithm with idealised bond geometry.

What is fixed vs. free
----------------------
* **Free:** the backbone dihedrals phi (C'-N-CA-C') and psi (N-CA-C'-N'), which
  the model generates, plus omega which we fix to the trans value (pi) by
  default (cis-proline is rare and unmodelled; a per-residue omega can be passed
  if wanted).
* **Fixed to ideal values:** bond lengths (N-CA, CA-C, C-N) and bond angles
  (N-CA-C, CA-C-N, C-N-CA), taken from standard engineering-geometry values.
  This is the universal convention for torsion-only reconstruction and is what
  makes phi/psi sufficient to place every backbone atom.

A crucial consequence, stated plainly because it shapes the evaluation: this
reconstruction places only the backbone with ideal local geometry, so it cannot
see side-chain or even backbone *steric clashes*. A chain can have perfectly
reasonable per-residue angles yet fold back through itself. That is exactly why
the evaluation suite includes a clash-rate metric on these reconstructed
coordinates -- the blind spot is measured, not assumed away.

Output is an (n_residues, 3, 3) array of N, CA, C coordinates in angstroms; a
helper adds approximate CB and O if a heavier observable needs them.
"""

from __future__ import annotations

import math

import numpy as np

# Idealised backbone geometry (angstroms / radians). Standard values.
BOND_LENGTHS = {"N_CA": 1.458, "CA_C": 1.525, "C_N": 1.329}
BOND_ANGLES = {  # bond angle at the central atom, radians
    "N_CA_C": math.radians(111.2),
    "CA_C_N": math.radians(116.2),
    "C_N_CA": math.radians(121.7),
}
OMEGA_TRANS = math.pi


def _place_atom(a, b, c, bond_length, bond_angle, dihedral) -> np.ndarray:
    """Place atom D given three predecessors A,B,C and internal coordinates.

    Implements one NeRF step: build the new atom in the local frame defined by
    the C->B and B->A directions, then rotate into the global frame. ``dihedral``
    is the A-B-C-D torsion; ``bond_angle`` is the B-C-D angle.
    """
    bc = c - b
    bc /= np.linalg.norm(bc) + 1e-9
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n) + 1e-9
    m = np.cross(n, bc)

    d2 = np.array(
        [
            -bond_length * math.cos(bond_angle),
            bond_length * math.sin(bond_angle) * math.cos(dihedral),
            bond_length * math.sin(bond_angle) * math.sin(dihedral),
        ]
    )
    basis = np.stack([bc, m, n], axis=1)  # columns are the local frame axes
    return c + basis @ d2


def reconstruct_backbone(
    phi: np.ndarray, psi: np.ndarray, omega: np.ndarray | None = None
) -> np.ndarray:
    """Rebuild N, CA, C coordinates from per-residue dihedrals.

    Parameters
    ----------
    phi, psi : (L,) radians. phi[0] and psi[-1] are undefined (terminal); any
        NaN is treated as the trans/extended default so reconstruction proceeds.
    omega : (L,) radians or None. None -> all-trans (pi).

    Returns
    -------
    (L, 3, 3) array of [N, CA, C] coordinates in angstroms.
    """
    L = len(phi)
    if omega is None:
        omega = np.full(L, OMEGA_TRANS)
    phi = np.nan_to_num(phi, nan=math.radians(-120.0))
    psi = np.nan_to_num(psi, nan=math.radians(120.0))

    coords = np.zeros((L, 3, 3), dtype=np.float64)

    # Seed the first three atoms (residue 0: N, CA, C) in a canonical frame.
    coords[0, 0] = np.array([0.0, 0.0, 0.0])  # N0
    coords[0, 1] = np.array([BOND_LENGTHS["N_CA"], 0.0, 0.0])  # CA0
    theta = BOND_ANGLES["N_CA_C"]
    coords[0, 2] = coords[0, 1] + BOND_LENGTHS["CA_C"] * np.array(
        [math.cos(math.pi - theta), math.sin(math.pi - theta), 0.0]
    )  # C0

    for i in range(1, L):
        n_prev, ca_prev, c_prev = coords[i - 1]
        # N_i: dihedral psi_{i-1} about CA_prev->C_prev
        n_i = _place_atom(
            n_prev, ca_prev, c_prev,
            BOND_LENGTHS["C_N"], BOND_ANGLES["CA_C_N"], psi[i - 1],
        )
        # CA_i: dihedral omega_{i-1} about C_prev->N_i
        ca_i = _place_atom(
            ca_prev, c_prev, n_i,
            BOND_LENGTHS["N_CA"], BOND_ANGLES["C_N_CA"], omega[i - 1],
        )
        # C_i: dihedral phi_i about N_i->CA_i
        c_i = _place_atom(
            c_prev, n_i, ca_i,
            BOND_LENGTHS["CA_C"], BOND_ANGLES["N_CA_C"], phi[i],
        )
        coords[i, 0] = n_i
        coords[i, 1] = ca_i
        coords[i, 2] = c_i

    return coords


def radius_of_gyration(ca_coords: np.ndarray) -> float:
    """Rg (angstrom) from CA coordinates, the standard IDP size observable."""
    centre = ca_coords.mean(axis=0)
    return float(np.sqrt(((ca_coords - centre) ** 2).sum(axis=1).mean()))


def end_to_end_distance(ca_coords: np.ndarray) -> float:
    return float(np.linalg.norm(ca_coords[-1] - ca_coords[0]))


def clash_count(ca_coords: np.ndarray, threshold: float = 3.8, sep: int = 2) -> int:
    """Count CA-CA pairs closer than ``threshold`` A beyond sequence sep ``sep``.

    A coarse but informative steric-validity proxy on the reconstructed chain.
    The 3.8 A default is just below the CA-CA virtual-bond distance, so genuine
    neighbours are excluded by the sequence separation, and anything closer is a
    real overlap. This is the metric that exposes torsion space's blind spot.
    """
    L = len(ca_coords)
    count = 0
    for i in range(L):
        for j in range(i + sep + 1, L):
            if np.linalg.norm(ca_coords[i] - ca_coords[j]) < threshold:
                count += 1
    return count
