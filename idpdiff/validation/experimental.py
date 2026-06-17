"""
Experimental-observable validation: SAXS and NMR.

Matching the training distribution's phi-psi histograms only proves the model
memorised marginals. The field demands agreement with the *primary experimental
data* the PED ensembles were derived from. This module is the scaffolding for
that comparison: it defines the contract and the orchestration, and shells out
to the established forward-prediction tools, which must be installed where there
is network access and the experimental data files.

This is deliberately scaffolding, not a finished computation, for an honest
reason: the forward predictors (CRYSOL / Pepsi-SAXS, SPARTA+ / UCBShift) are
external binaries, and the experimental targets (SAXS curves, BMRB chemical
shifts, RDCs, PREs) must be downloaded and matched to each PED entry. None of
that can happen in an offline sandbox. What is provided here is the exact
pipeline shape so that, on a connected machine, filling in the two marked tool
calls makes it run.

Pipeline
--------
generated angles (n_conf, L, 2)
    -> reconstruct full-backbone PDB ensemble (validation.reconstruct + side
       chains via an external tool such as PULCHRA/SCWRL where the predictor
       needs all-atom input)
    -> per-conformer forward prediction (SAXS profile / chemical shifts)
    -> ensemble-average the predicted observable
    -> compare to experiment (reduced chi-square)

Reporting reduced chi-square against SAXS Rg/profile and at least one NMR
observable on *held-out* proteins is the single most load-bearing result for
acceptance.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import numpy as np

from .reconstruct import reconstruct_backbone


@dataclass
class SAXSResult:
    q: np.ndarray
    intensity_pred: np.ndarray
    intensity_exp: np.ndarray | None
    chi_square: float | None


def write_backbone_pdb(angles_one_conf: np.ndarray, path: str, sequence: str) -> None:
    """Write a single reconstructed conformer to a minimal backbone PDB.

    Only N, CA, C are written. For predictors that need all-atom input, pass the
    result through a side-chain builder (PULCHRA/SCWRL) first; that call is one
    of the marked external steps in :func:`run_saxs`.
    """
    coords = reconstruct_backbone(angles_one_conf[:, 0], angles_one_conf[:, 1])
    atom_names = ["N", "CA", "C"]
    with open(path, "w") as fh:
        serial = 1
        for i in range(coords.shape[0]):
            resname = "GLY"  # placeholder; sequence-aware naming optional
            for a, name in enumerate(atom_names):
                x, y, z = coords[i, a]
                fh.write(
                    f"ATOM  {serial:5d}  {name:<3s} {resname} A{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
                )
                serial += 1
        fh.write("END\n")


def run_saxs(
    gen_angles: np.ndarray,
    sequence: str,
    workdir: str,
    exp_profile_path: str | None = None,
    saxs_tool: str = "crysol",
) -> SAXSResult:
    """Predict an ensemble-averaged SAXS profile and compare to experiment.

    The two marked subprocess calls (side-chain build + SAXS predictor) are
    where external tools plug in. Offline, this raises a clear message rather
    than fabricating a curve.
    """
    os.makedirs(workdir, exist_ok=True)
    per_conf_curves = []
    for k in range(gen_angles.shape[0]):
        pdb_path = os.path.join(workdir, f"conf_{k}.pdb")
        write_backbone_pdb(gen_angles[k], pdb_path, sequence)
        # ---- EXTERNAL STEP 1 (optional): all-atom rebuild ----
        #   subprocess.run(["pulchra", pdb_path], check=True)
        # ---- EXTERNAL STEP 2: SAXS forward prediction ----
        #   e.g. crysol/pepsi-saxs producing a (q, I) curve for this conformer
        raise NotImplementedError(
            f"Wire up `{saxs_tool}` here on a machine that has it installed. "
            "Each conformer's PDB is already written to workdir; run the SAXS "
            "predictor per conformer, collect (q, I) curves, then this function "
            "ensemble-averages them and computes reduced chi-square vs experiment."
        )

    # Unreachable offline; shape of the real computation:
    mean_curve = np.mean(per_conf_curves, axis=0)  # noqa: F841 (documentation)


def run_chemical_shift_validation(
    gen_angles: np.ndarray,
    sequence: str,
    workdir: str,
    bmrb_path: str | None = None,
    predictor: str = "ucbshift",
) -> dict:  # pragma: no cover - external tool
    """Predict ensemble-averaged backbone chemical shifts; compare to BMRB.

    Same shape as :func:`run_saxs`: write conformers, run the predictor per
    conformer (UCBShift/SPARTA+), ensemble-average, and compare to deposited
    BMRB shifts by per-nucleus RMSD. Plug in the predictor where available.
    """
    raise NotImplementedError(
        "Wire up the chemical-shift predictor here; the orchestration mirrors "
        "run_saxs. Compare ensemble-averaged predicted shifts to BMRB by "
        "per-nucleus RMSD on held-out proteins."
    )


def reduced_chi_square(
    pred: np.ndarray, exp: np.ndarray, exp_err: np.ndarray, n_params: int = 1
) -> float:
    """Reduced chi-square between predicted and experimental observables."""
    resid = (pred - exp) / np.clip(exp_err, 1e-9, None)
    dof = max(1, len(exp) - n_params)
    return float(np.sum(resid ** 2) / dof)
