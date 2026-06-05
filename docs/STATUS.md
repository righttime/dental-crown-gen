# Project Status

**Last updated:** 2026-06-06 (third paper read)

## Current state
- [x] Concept defined (intra-oral scan → printable crown/bridge mesh)
- [x] Sub-tasks identified (5 steps)
- [x] Hypotheses drafted (5)
- [x] Architecture draft (training = workstation, inference = TBD)
- [x] Literature survey (Scholar — 3 papers read, see `papers/001-…`, `papers/002-…`, `papers/003-…`)
- [ ] Data acquisition plan
- [ ] Baseline implementation
- [ ] Custom model dev
- [ ] Mesh output validation
- [ ] Clinical fit evaluation

## Open questions
- Data acquisition: public datasets vs synthetic vs scraping?
- Compute: cloud GPU (Lambda, RunPod) vs university cluster?
- 3D format conversion pipeline (OBJ/STL/PLY all in?)

## Scholar weekly digest

- **Week of 2026-06-05:** read paper 001 — *3DTeethSeg'22: 3D Teeth Scan Segmentation and Labeling Challenge* (Ben-Hamadou et al., MICCAI 2022 challenge). Key insight: largest public IOS dataset (1,800 scans / 900 patients / 23,999 FDI-labeled teeth) is now downloadable, and every winning method is 2-stage (centroid detection → per-tooth segmentation) — strong empirical support for **H1**. IGIP team's dental-arch-curve post-processing (Bezier fit through centroids used to correct FDI labels) is a direct analogue of the inductive bias we need for **H3** (conditioning on global arch context for outer-surface generation). Concrete action: download the dataset this week, use it as the public-data backbone for sub-task 1. Note in `papers/001-3dteethseg22.md`.
- **Week of 2026-06-06:** read paper 002 — *DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation* (Park, Florence, Straub, Newcombe, Lovegrove, CVPR 2019). Key insight: **strong support for H4** (implicit SDF > explicit mesh). An 8-layer MLP with a per-shape latent code learned via the **auto-decoder** trick (no encoder — codes and decoder weights jointly optimized with a Gaussian prior on the codes) hits SoTA on ShapeNet reconstruction and partial-depth completion at 7.4 MB with watertight surfaces and analytic normals. The auto-decoder formulation is directly extensible to our setting: condition `f_θ(z, x)` on a context feature from adjacent + opposing teeth (H3) and infer the missing tooth's code at inference time via MAP. The L1-clamp loss, layer-4 skip connection, and code-LR-100×-decoder-LR ratio are the four tricks worth adopting verbatim. Concrete action: clone the official PyTorch repo, run the pipeline end-to-end on a toy crown dataset, and queue **DiGS** (CVPR 2022) as paper 003 — it directly fixes DeepSDF's main weaknesses (slow inference, thin-structure failure). Note in `papers/002-deepsdf.md`.
- **Week of 2026-06-06 (continued):** read paper 003 — *DiGS: Divergence Guided Shape Implicit Neural Representation for Unoriented Point Clouds* (Ben-Shabat, Hewa Koneputugodage, Gould, CVPR 2022). Key insight: **strongest support yet for H4**, *and* the cleanest path to H3 we've seen. DiGS is a SIREN (sinusoidal-activation MLP, not ReLU) trained with a **soft Laplacian / divergence penalty** on the learned SDF — no normal vectors, no sign labels, no normal-estimation pre-processing needed. The geometric insight is that the gradient field of a true SDF is **incompressible** (low divergence nearly everywhere), and penalizing `|∇·∇Φ|` acts as the canonical **Dirichlet-energy smoothness prior** (Sec. 5.3 derives this formally). The architectural price is a non-negotiable `C²` activation, hence SIREN. Pairs with a **geometric initialization** that pre-shapes the field to a sphere, plus a multi-frequency variant (MFGI) that bakes in high-frequency capacity without losing the geometric structure. SoTA on SRB and ShapeNet among **unoriented** methods (Table 2: dC 0.19 vs. SIREN-wo-n's 0.42; Table 3: IoU 0.939 on 260 shapes, beating every normal-supervised method). **The crown jewel for us: the auto-decoder extension to DFAUST (Sec. 6.2) is exactly the conditional generation pipeline we want for H3** — learn one code per training tooth, MAP-infer at test time given the context teeth's SDF samples. **Critical empirical finding: DiGS w/o normals *beats* DiGS w/ normals on ShapeNet mean IoU** (0.939 vs. 0.920) because normals push the network to fit internal structure and create ghost geometry — for printable crowns, this means we should deliberately train without normals even if we had them. **Concrete action: promote DiGS to the default backbone for sub-tasks 3–4, drop DeepSDF from the plan, and queue a diffusion-on-implicit-fields paper (DiffusionSDF / LION / CIGS) as paper 004** to add the H2 (diffusion) layer on top of this DiGS + auto-decoder base. Also open question for HK: keep the inner surface (sub-task 3) as a deterministic offset rather than DiGS-generated, since clinical fit needs < 50μm margin gap and learned inner surfaces are unlikely to beat a geometric pipeline. Note in `papers/003-digs.md`.
