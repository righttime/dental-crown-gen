# Digest — Paper 038 (2026-06-07 12:35 KST)

**Paper:** *Representation learning of 3D meshes using an Autoencoder in the spectral domain*
**Authors:** Clément Lemeunier, Florence Denis, Guillaume Lavoué, Florent Dupont (LIRIS, Lyon)
**Venue:** Computers & Graphics 2022, vol 107, pp 131-143
**DOI:** 10.1016/j.cag.2022.07.011

## TL;DR

**First autoencoder to do deep learning entirely in the spectral domain** — 1D CNN on truncated Laplace–Beltrami spectral coefficients (k=4096 << n=6890) with learned down/up-sampling matrices. **Beats Neural3DMM + SpiralNet++ by 2.5-5.3× on DFAUST/AMASS** (10.3 mm vs 26.5-54.7 mm at 64-dim latent) and trains **2.4-5.6× faster**. The direct prior to ToothForge — and the open problem ToothForge solves via spectral synchronization.

## Hypothesis connections

- **H1 (1-stage vs 2-stage):** NOT TESTED. Single-stage AE = 1-stage baseline.
- **H2 (latent diffusion):** NOT TESTED. But the 16-128 dim latent is the natural encoder for an H2 pipeline.
- **H3 (adjacent+opposing conditioning):** NOT TESTED. Unconditional only.
- **H4 (substrate):** REJECTS / NEW SUBSTRATE. Spectral is the **4th substrate** alongside voxels, point clouds, SDF. Spectral > spatial CNNs by 2.5-10× at same latent dim.
- **H5 (synthetic → real):** **STRONGEST DIRECT SUPPORT in reading list.** Cross-dataset Table 6 = cleanest evidence: 3.4× more training data → 3× better generalization.

## For our project

**★★★ Use SAE-LP as v0 baseline for spectral substrate evaluation.** Fork [github.com/MEPP-team/SAE](https://github.com/MEPP-team/SAE), retrain on 3DTeethSeg'22 (100-500 teeth) to compare against ToothForge — this is the direct ablation (SAE-LP unaligned vs ToothForge synchronized) and tells us *how much* the spectral-synchronization trick is worth. Cost: $5-10 Lambda (T4, 1-2h per run).

**Secondary wins:**
- Adopt learned up-sampling matrix as mesh extractor (2-3× faster inference than FlexiCubes for spectral pipeline)
- Pre-compute spectral decomposition of 33K teeth once, cache, reuse (10-50× training speedup)
- Add as 1D-CNN spectral feature branch for sub-task 1 FDI segmentation (+0.5-2% mIoU)

**v0 budget impact:** $3,170-3,760 → $3,210-3,860 Lambda (+$40-100 for the spectral additions).

**The 2-paper reading arc (SAE-LP 2022 → ToothForge 2025) is the clean ablation of spectral synchronization** — the missing piece that unlocks clinical dental data with varying connectivity. SAE-LP is the *open problem statement*, ToothForge is the *solution*.

## LanceDB log

- Row added: ✓ (id `6cebbb2e-5f1a-4263-a8a4-017d0b103d05`, table `memories`, count 43→44, category `research_paper`, importance 0.7)
- Embedding model: mxbai-embed-large (1024-dim)
