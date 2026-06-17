# Torsion-space wrapped diffusion for IDP conformational ensembles

A diffusion model that generates backbone torsion-angle (φ, ψ) ensembles for
intrinsically disordered proteins (IDPs), conditioned on sequence via ESM
embeddings, with a **Ramachandran-KDE regularizer** as the central contribution.

The scientific question the codebase is built to answer:

> Does regularizing a torsion-space generative model toward an empirical,
> residue-type-specific Ramachandran density improve the physical fidelity of
> generated IDP ensembles — and at what cost to ensemble diversity?

The model generates angles on the torus (`T^(2L)`) using **torus-native wrapped
diffusion**, not Euclidean diffusion on sin/cos features, so the periodicity of
the angles is respected exactly.

---

## What is here

```
idpdiff/
  constants.py            torch-free shared constants (residue classes)
  utils/angles.py         circular math: wrapping, wrapped-normal, sin/cos features
  diffusion/
    wrapped.py            torus-native wrapped diffusion (forward + reverse), x0-param
    losses.py             angular reconstruction loss + Ramachandran-KDE regularizer
  models/denoiser.py      Transformer x0-predictor, per-residue ESM + AdaLN timestep
  data/
    torsions.py           PED/PDB -> (phi, psi) extraction; residue-class assignment
    embeddings.py         ESM-2 embedder (cached) + offline MockEmbedder
    dataset.py            per-conformer dataset, collation, PROTEIN-LEVEL splits
  validation/
    reconstruct.py        NeRF: angles -> N,CA,C coordinates; Rg, Ree, clash count
    experimental.py       SAXS / NMR forward-validation scaffolding (external tools)
  eval/
    metrics.py            per-residue phi-psi JSD, observable Wasserstein, diversity
    baselines.py          statistical-coil null (ready) + idpGAN adapter (stub)
  train.py                training loop with annealed regularizer weight

scripts/
  build_ramachandran_kde.py   build per-class KDE from PED
  prepare_data.py             extract angles, build protein-level splits
  train_model.py              train one arm (--lambda-max 0 = without-reg)
  sample_and_eval.py          sample held-out proteins, run full metric suite

tests/
  test_numpy_core.py      numpy-only tests (run anywhere; all passing)
  test_torch_smoke.py     end-to-end torch test (run where torch is installed)

configs/                  with_reg.json / without_reg.json
docs/baselines.md         how to run the statistical-coil and idpGAN baselines
paper/                    paper skeleton (methods written, results placeholders)
```

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch`, `biotite`, and `fair-esm` require network access to install and (for
ESM) to download weights. The numpy-only tests run without any of these:

```bash
python tests/test_numpy_core.py      # passes offline
python tests/test_torch_smoke.py     # run after torch is installed
```

---

## Run order

The ordering matters; in particular, **run the statistical-coil baseline
comparison before investing in the full experimental validation.** If the
learned model cannot beat an independent residue-wise sampler on ensemble
observables, that is the first thing to discover.

1. **Get data.** Download PED ensembles (multi-model PDB/mmCIF) into a directory.
   Each entry is one protein; its MODELs are the ensemble.

2. **Build the Ramachandran KDE** (from the disordered data itself, *not* folded
   proteins):
   ```bash
   python scripts/build_ramachandran_kde.py \
       --ped-dir data/ped --out artifacts/rama_kde.pt --bandwidth-deg 15
   ```

3. **Prepare data + protein-level splits** (optionally cluster-aware):
   ```bash
   python scripts/prepare_data.py \
       --ped-dir data/ped --out artifacts/dataset.pt \
       [--clusters artifacts/clusters.tsv]
   ```

4. **Sanity-check the pipeline offline** with the mock embedder (tiny, fast,
   no ESM download — checks plumbing only, not quality):
   ```bash
   python scripts/train_model.py --dataset artifacts/dataset.pt \
       --out-dir runs/smoke --mock-esm --max-steps 50 --batch-size 4 \
       --width 64 --depth 2 --num-diffusion-steps 50
   ```

5. **Train both arms** (the headline experiment — identical except the flag):
   ```bash
   # with Ramachandran regularisation
   python scripts/train_model.py --dataset artifacts/dataset.pt \
       --rama-kde artifacts/rama_kde.pt --out-dir runs/with_reg --lambda-max 1.0

   # without (ablation)
   python scripts/train_model.py --dataset artifacts/dataset.pt \
       --out-dir runs/without_reg --lambda-max 0.0
   ```

6. **Sample + evaluate** on held-out proteins, including the statistical-coil
   baseline:
   ```bash
   python scripts/sample_and_eval.py --dataset artifacts/dataset.pt \
       --checkpoint runs/with_reg/ema_latest.pt --rama-kde artifacts/rama_kde.pt \
       --out runs/with_reg/eval.json

   python scripts/sample_and_eval.py --dataset artifacts/dataset.pt \
       --checkpoint runs/without_reg/ema_latest.pt --rama-kde artifacts/rama_kde.pt \
       --out runs/without_reg/eval.json
   ```

7. **idpGAN baseline** — see `docs/baselines.md`.

8. **Experimental validation (SAXS / NMR)** — `idpdiff/validation/experimental.py`
   defines the pipeline; plug in CRYSOL/Pepsi-SAXS and UCBShift/SPARTA+ where you
   have them installed. This is the most labour-intensive and most load-bearing
   part of the paper.

---

## Key design decisions (and why)

- **Torus-native wrapped diffusion, x0-parameterisation.** Noise is added as a
  *wrapped* normal intrinsic to the circle; the network predicts the clean angle
  rather than the noise, because the winding number of "the noise" is
  unidentifiable after wrapping. This removes the ±π seam artefacts that a naive
  sin/cos-MSE model suffers from.

- **Residue-type-specific Ramachandran densities.** Glycine, proline, pre-proline
  and general residues occupy distinct basins; collapsing them into one density
  would blur exactly the structure the regulariser is meant to encode.

- **KDE built from disordered data, not folded proteins.** IDP ensembles are
  PPII-enriched and shifted relative to folded-protein Ramachandran maps; seeding
  the prior with folded statistics would bias the model away from the target
  landscape.

- **Protein-level (not conformer-level) splits.** A per-conformer split leaks
  members of the same ensemble across train/test and inflates every metric. The
  whole point of sequence conditioning is generalisation to unseen sequences.

- **Distributional evaluation.** IDPs are defined by the *spread* of their
  ensemble, so global observables are compared as full distributions
  (Wasserstein), not means. Diversity is reported alongside fidelity so the
  regularizer's fidelity/diversity trade-off is explicit, never hidden.

- **Clash rate is measured, not assumed away.** Torsion-space + ideal-geometry
  reconstruction cannot see steric clashes; the evaluation reports the clash rate
  so this known blind spot is quantified.

---

## What this repository does NOT do for you

It does not fabricate results. Producing the paper's numbers requires PED data,
ESM/idpGAN weights, and the external SAXS/NMR predictors — all of which need a
networked machine. The code is structured so that, with those in place, the run
order above produces the figures and tables the paper skeleton expects.
