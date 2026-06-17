# Baselines

This project compares the torsion-space diffusion model against two baselines.
Both must clear the same evaluation pipeline (`idpdiff/eval/metrics.py`) so the
numbers are directly comparable.

## 1. Residue-specific statistical coil (null model)

Implemented and ready in `idpdiff/eval/baselines.py::StatisticalCoilModel`.

This is the field-standard null hypothesis: each residue's (phi, psi) is drawn
**independently** from an empirical, residue-type-specific distribution, with no
inter-residue coupling and no sequence information beyond residue identity. It is
built from the same per-class Ramachandran KDE used by the regulariser, so the
comparison isolates exactly what the learned model adds *on top of* correct
marginals: inter-residue correlations and sequence specificity.

**Run this comparison first.** If the learned model does not beat an independent
sampler on ensemble observables (Rg / Ree distributions, not just per-residue
marginals), that is the most important thing to learn early, before investing in
the full experimental-validation pipeline.

## 2. idpGAN (external learned baseline)

Interface stub in `idpdiff/eval/baselines.py::IDPGANBaseline`. idpGAN
(Janson, Valdes-Garcia, Heo, Feig, *Nat. Commun.* 2023) is the most directly
comparable published generative model for IDP ensembles.

### Setup (on a networked machine)

1. Clone the idpGAN repository and install its dependencies in a separate env.
2. Download the released weights.
3. Generate ensembles for **the same held-out sequences** used to evaluate this
   model (read them from `dataset.pt`'s `test` split so the protein sets match
   exactly).

### Output conversion

idpGAN emits coarse-grained CA traces / its own internal representation, whereas
this repo's metrics consume backbone torsions of shape `(n_conf, L, 2)` in
radians. Implement the conversion inside `IDPGANBaseline.sample_ensemble`:

- If idpGAN gives CA-only coordinates, you cannot recover phi/psi directly
  (phi/psi need N and C). Two options:
  - **(preferred)** Compare idpGAN and this model at the level of **global
    observables** (Rg, Ree, asphericity) and CA-based contact maps, which are
    defined for CA-only ensembles. This keeps the comparison fair and avoids
    inventing backbone atoms idpGAN never modelled.
  - If you need full phi/psi for idpGAN, rebuild a backbone from its CA trace
    with a backbone-reconstruction tool (e.g. PULCHRA), then extract phi/psi.
    Document this clearly as it adds reconstruction error to idpGAN's side.

- If you obtain a full-atom or N-CA-C ensemble, use the same dihedral routine as
  the data pipeline (`idpdiff/data/torsions.py::phi_psi_from_chain`) so both
  models' angles are computed identically.

### Reporting

Report idpGAN and this model on the **same** held-out proteins and the **same**
metrics in one table. Note the conversion path used for idpGAN so the comparison
is reproducible and the reconstruction caveat (if any) is explicit.
