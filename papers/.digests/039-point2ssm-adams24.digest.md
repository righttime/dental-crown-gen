# Digest — Paper 039 (2026-06-07 13:35 KST)

**Paper:** *Point2SSM: Learning Morphological Variations of Anatomies from Point Clouds*
**Authors:** Jadie Adams, Shireen Y. Elhabian
**Affiliation:** Scientific Computing and Imaging (SCI) Institute, Kahlert School of Computing, University of Utah
**Venue:** ICLR 2024 Spotlight (~5% acceptance), arXiv:2305.14486v2 (24 Jan 2024)
**Citations:** ~85-100 (Semantic Scholar, Jun 2026); follow-up Point2SSM++ exists

## TL;DR

**First deep-learning method for *unsupervised* correspondence-based Statistical Shape Models (SSMs) from raw unordered point clouds** — 3-stage DGCNN + SFA attention + attention-map output, trained with Chamfer + pairwise ME loss. **8h GPU vs 4 days CPU for ShapeWorks PSM on 1096-shape left-atrium cohort (12× speedup)**, with statistically equivalent modes. SoTA on spleen, pancreas, left atrium, L4 vertebrae.

## Hypothesis connections

- **H1 (2-stage > 1-stage):** MILD CONTRADICTION. Point2SSM is single-stage and *still* SoTA → H1 only holds for *generative* tasks, not correspondence.
- **H2 (latent diffusion > direct):** MILD CONTRADICTION. Deterministic, no diffusion → H2 holds for *generative* multi-modal, not for single-modal correspondence.
- **H3 (global arch context > local):** ★★★ **STRONGEST SUPPORT IN READING LIST.** Entire architecture is the H3 bias — SFA attention + ordered output + ME loss enforce same output index = same anatomical landmark across entire cohort.
- **H4 (implicit SDF > explicit):** MILD CONTRADICTION + REFINES. Refines into H4a (SDF > point cloud for reconstruction, DiGS confirms) + H4b (ordered correspondence > unordered point cloud for population analysis, Point2SSM confirms).
- **H5 (synthetic → real transfer):** STRONG SUPPORT. Trains on real clinical data with no synthetic pretraining; Gaussian noise σ=0.25-2mm at test time without retraining generalizes fine. Cleanest H5 evidence for clinical-data-as-is training.

## For our project

**Closes the substrate triangle** (SAE-LP 038 spectral + ToothForge 037 spectral+sync + Point2SSM 039 point cloud). **For v0 sub-task 1 (FDI segmentation): adopt Point2SSM-derivative as 3rd alternative to Cao25 + CrownSegger** — train DGCNN+SFA on 3DTeethSeg22 molars, output 1024 ordered correspondence, FDI label = cluster index. Start with **2.7M-param DGCNN+MLP variant** (sub-$50 Lambda, <1h on T4); per-FDI-class model (not combined) per multi-anatomy-training-doesn't-help negative result. **For v0 sub-task 4 (outer surface): add ME loss as correspondence-consistency regularizer in PVD-AF-DiGS-FC pipeline** — ~50 lines PyTorch, $0 engineering, only way to enforce H3 at generation stage. **Killer UX: attention map is interpretable** → dentist-facing "this output point came from THIS region of your input arch" confidence map.

## LanceDB log

- Row added: ✓ (id `910bdf2b-35e5-48b3-946c-13296d80582e`, table `memories`, count 44→45, category `research_paper`, importance 0.7)
- Embedding model: mxbai-embed-large (1024-dim)
