"""
Extract backbone torsion angles from PED / PDB structure files.

PED entries are deposited as multi-model PDB or mmCIF files: each entry is one
protein, and the many MODEL records are the members of its conformational
ensemble. The unit we want for training is therefore one (sequence, [angles per
conformer]) record per PED entry, where each conformer contributes a length-L
array of (phi, psi) pairs.

This module wraps Biotite (preferred) with a graceful fallback message if it is
not installed, and exposes:

* :func:`extract_entry` -- parse one structure file into a
  :class:`TorsionEntry` (sequence, per-model angle arrays, residue-class ids).
* :func:`phi_psi_from_chain` -- the low-level dihedral computation for a single
  model/chain.

Residue-class assignment (general / glycine / proline / pre-proline) is done
here so it travels with the data and the Ramachandran regulariser can index it
directly.

Terminal residues have an undefined phi (no preceding C) or psi (no following
N); these are recorded as NaN and the training mask excludes them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..constants import RAMA_CLASS_TO_ID

# Three-letter to one-letter amino acid map (standard 20).
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


@dataclass
class TorsionEntry:
    """One PED entry: a sequence and its ensemble of torsion angles."""

    entry_id: str
    sequence: str  # one-letter, length L
    angles: np.ndarray  # (n_models, L, 2) radians; NaN where undefined
    class_ids: np.ndarray  # (L,) residue-class ids for the Ramachandran KDE

    @property
    def n_models(self) -> int:
        return self.angles.shape[0]

    @property
    def length(self) -> int:
        return len(self.sequence)


def _residue_class_ids(one_letter_seq: str) -> np.ndarray:
    """Assign each residue a Ramachandran class id.

    pre-proline (any residue immediately followed by PRO) is its own class
    because its accessible (phi, psi) basin differs markedly from the general
    case; glycine and proline are likewise distinct.
    """
    ids = np.empty(len(one_letter_seq), dtype=np.int64)
    for i, aa in enumerate(one_letter_seq):
        nxt = one_letter_seq[i + 1] if i + 1 < len(one_letter_seq) else None
        if aa == "G":
            cls = "glycine"
        elif aa == "P":
            cls = "proline"
        elif nxt == "P":
            cls = "pre_proline"
        else:
            cls = "general"
        ids[i] = RAMA_CLASS_TO_ID[cls]
    return ids


def phi_psi_from_chain(coords_by_residue) -> np.ndarray:
    """Compute (phi, psi) for a list of residues' backbone atom coordinates.

    ``coords_by_residue`` is a list of dicts with keys 'N', 'CA', 'C' mapping to
    length-3 numpy arrays. Returns (L, 2) radians with NaN at undefined termini.
    Implemented from first principles (no Biotite dependency) so the dihedral
    convention is explicit and testable.
    """
    L = len(coords_by_residue)
    out = np.full((L, 2), np.nan, dtype=np.float64)
    for i in range(L):
        # phi_i uses C_{i-1}, N_i, CA_i, C_i
        if i > 0:
            out[i, 0] = _dihedral(
                coords_by_residue[i - 1]["C"],
                coords_by_residue[i]["N"],
                coords_by_residue[i]["CA"],
                coords_by_residue[i]["C"],
            )
        # psi_i uses N_i, CA_i, C_i, N_{i+1}
        if i < L - 1:
            out[i, 1] = _dihedral(
                coords_by_residue[i]["N"],
                coords_by_residue[i]["CA"],
                coords_by_residue[i]["C"],
                coords_by_residue[i + 1]["N"],
            )
    return out


def _dihedral(p0, p1, p2, p3) -> float:
    """Signed dihedral angle (radians) about the p1-p2 bond, IUPAC convention."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-8)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return math.atan2(y, x)


def extract_entry(path: str, entry_id: str | None = None) -> TorsionEntry:
    """Parse a multi-model PDB/mmCIF file into a :class:`TorsionEntry`.

    Uses Biotite to read all models and the first chain. Raises a clear error if
    Biotite is unavailable so the environment problem is obvious rather than
    surfacing as a cryptic ImportError deep in a stack trace.
    """
    try:
        import biotite.structure as struc
        import biotite.structure.io as strucio
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "extract_entry needs biotite. Install with `pip install biotite`. "
            "This step runs where you have the PED files and network access."
        ) from exc

    entry_id = entry_id or path.split("/")[-1].split(".")[0]
    stack = strucio.load_structure(path)  # AtomArrayStack for multi-model files

    # Normalise to a stack even if a single model was returned.
    if not hasattr(stack, "stack_depth"):
        stack = struc.stack([stack])

    # Use the first chain present.
    chain_ids = np.unique(stack.chain_id)
    first_chain = chain_ids[0]
    chain_mask = stack.chain_id == first_chain

    # Build sequence + residue ids from the first model.
    model0 = stack[0][chain_mask]
    res_ids = np.unique(model0.res_id)
    seq_three = []
    for rid in res_ids:
        rname = model0.res_name[model0.res_id == rid][0]
        seq_three.append(rname)
    sequence = "".join(THREE_TO_ONE.get(r, "X") for r in seq_three)

    angle_models = []
    for m in range(stack.stack_depth()):
        model = stack[m][chain_mask]
        coords_by_residue = []
        for rid in res_ids:
            sel = model[model.res_id == rid]
            atoms = {a.atom_name: a.coord for a in sel}
            if not all(k in atoms for k in ("N", "CA", "C")):
                # Missing backbone atom -> leave NaN by inserting placeholder.
                coords_by_residue.append(
                    {"N": np.full(3, np.nan), "CA": np.full(3, np.nan), "C": np.full(3, np.nan)}
                )
            else:
                coords_by_residue.append(
                    {"N": atoms["N"], "CA": atoms["CA"], "C": atoms["C"]}
                )
        angle_models.append(phi_psi_from_chain(coords_by_residue))

    angles = np.stack(angle_models, axis=0)  # (n_models, L, 2)
    class_ids = _residue_class_ids(sequence)
    return TorsionEntry(entry_id, sequence, angles, class_ids)
