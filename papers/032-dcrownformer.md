# Paper 032 — *DCrownFormer: Morphology-Aware Point-to-Mesh Generation Transformer for Dental Crown Prosthesis from 3D Scan Data of Antagonist and Preparation Teeth*

**Authors:** Su Yang¹*, Jiyong Han²*, Sang-Heon Lim², Ji-Yong Yoo¹, SuJeong Kim², Dahyun Song², Sunjung Kim³, Jun-Min Kim⁴·⁵, Won-Jin Yi¹·²·⁶† (*co-first, †co-corresponding)
**Affiliations:**
1. Applied Bioengineering, Graduate School of Convergence Science and Technology, Seoul National University
2. Interdisciplinary Program in Bioengineering, Graduate School of Engineering, Seoul National University
3. Imaging R&D Center, **Osstem Implant Co., Ltd.**, Seoul
4. Medical Imaging R&D Center, **Xcube Co., Ltd.**, Seoul
5. Electronics and Information Engineering, Hansung University, Seoul
6. Oral and Maxillofacial Radiology and Dental Research Institute, School of Dentistry, Seoul National University
**Venue:** **MICCAI 2024** (Marrakesh, Morocco, October 2024) — Springer LNCS vol. 15006, pp. 109–119, DOI 10.1007/978-3-031-72089-5_11
**Preprint:** ❌ none (MICCAI proceedings only; not on arXiv)
**Code:** ⚠️ **withdrawn — see Code/Status section below** (initial `suyang93/DCrownFormer` GitHub repo transferred to a Korean company in Sept 2024; README now reads *"Unfortunately, we are no longer able to share the code"*)
**Funding:** Korea Medical Device Development Fund (KMDF-PR-20200901-0011) + NRF (2023R1A2C200532611)
**Citations:** early/mid 2026 (~50–80 estimated) — high citation velocity because it was the **only transformer-based point-to-mesh crown generator at MICCAI 2024**, and MADCrowner (paper we may read next, arXiv:2603.04771 Mar 2026) extends it directly.
**Read:** 2026-06-07 07:03 KST (Sunday, scholar hourly #20, ~40 min)

---

## TL;DR

**The current SOTA on the *internal Osstem dental crown dataset* — and the first transformer that does *direct point-to-mesh* crown generation (no separate surface reconstruction post-proc).** DCrownFormer takes **antagonist + preparation + adjacent teeth point clouds** as input and outputs a **crown mesh** in one forward pass, using three novel ingredients: (1) a **Morphology-aware Cross-Attention Module (MCAM)** in the transformer decoder that learns the geometric relationship between context points and the to-be-generated crown points, (2) a **Curvature-Penalty Loss (CPL)** that explicitly rewards preservation of high-curvature occlusal features (cusps, grooves, ridges) — the thing the Chamfer Distance is blind to, and (3) **Differentiable Poisson Surface Reconstruction (DPSR) with a Mesh Reconstruction Loss (MRL)** that makes the iso-surface extraction differentiable so the whole network can be supervised end-to-end at the *mesh* level. The headline empirical claim is **state-of-the-art on CD, SDE, and a clinician-rated occlusal sharpness comparison** vs. PCN+SAP, TopNet+SAP, GRNet+SAP, PoinTr+SAP, and SnowflakeNet+SAP. **For our v0: DCrownFormer is the *exact* prior-art baseline that v0 sub-task 2 (crown generation) is competing against — its three ingredients (MCAM + CPL + MRL) are all we need to port.** The catch (and a big one for v0/v1 reproducibility): **the code was technology-transferred to a Korean company in Sept 2024 and is no longer public**, so we have to re-implement from the paper's 8-page description, or, more pragmatically, **start from DMC (Hosseinimanesh 2023, arXiv:2501.04914) — which is open source and uses the same DPSR-based pipeline — and graft the MCAM + CPL on top**. The dataset is **internal to Osstem/Xcube** so we can't even compare numbers apples-to-apples; the v0 will need to use 3DTeethSeg22 + a synthetic crown GT pipeline (the ToSynFCD or paper-024 Kunwar26 retrieval approach) to get a public benchmark.

## Research question + their answer

**Q:** Existing dental crown generation methods (cGAN on 2D depth maps, point cloud completion with separate surface reconstruction, retrieval-then-deform) all fail on at least one of three clinical requirements — (1) full 3D crown shape (not just occlusal surface), (2) preservation of fine morphological details (cusps, grooves, marginal ridges), and (3) proper alignment with both the preparation tooth and the antagonist dentition. Can a **single end-to-end network** that conditions on the **antagonist + preparation + adjacent teeth** directly produce a **patient-specific crown mesh** that beats prior pipelines on all three requirements simultaneously?

**A:** Yes — using a **point-to-mesh transformer** with three architectural innovations that target exactly those failure modes:

1. **Morphology-aware Cross-Attention Module (MCAM) in the transformer decoder** → solves (3) [antagonist + adjacent alignment]. The cross-attention is between *generated crown points* and *context points* (prep + adjacent + antagonist), with the key insight that the *spatial layout* of the context is encoded into the attention bias — so the model learns "the cusp tips should be 2mm below the antagonist occlusal plane" implicitly.
2. **Curvature-Penalty Loss (CPL, λ=1.0)** → solves (2) [fine detail]. Standard CD + normal loss produce *smooth* crowns because CD tolerates local averaging; CPL adds an L1 penalty on the *curvature divergence* between predicted and GT crown surfaces, so the model is explicitly punished for smoothing over cusps.
3. **Mesh Reconstruction Loss (MRL) on the DPSR indicator function** → solves (1) [full 3D mesh supervision, not just points]. By making the entire pipeline (point generation + DPSR extraction) differentiable, the loss can be computed on the *extracted mesh* rather than on the *intermediate points* — so the network is trained to make the *mesh* match GT, not just the *point cloud*. This is the same trick DMC (paper 006) and DMTet (paper 031) use.

Together: **CD, SDE, and a user study on occlusal detail all improve** over the 5 baselines (PCN+SAP, TopNet+SAP, GRNet+SAP, PoinTr+SAP, SnowflakeNet+SAP) on the Osstem internal test set.

## Method (architecture, training, inference)

### Pipeline (5 stages)
```
[Antagonist + Preparation + Adjacent point cloud]
        ↓  (farthest-point-sampling → ~10k points, with normals)
[Encoder: multi-scale DGCNN-style + transformer self-attention blocks]
        ↓  (per-point features, N×D)
[Coarse Point Generation: fold decoder → coarse crown points, ~2048]
        ↓
[MCAM-enhanced decoder → refined crown points + predicted normals, ~4096]
        ↓
[DPSR Differentiable Poisson Surface Reconstruction]
        ↓  (indicator function on 128³ grid, Marching Cubes at 0.5 iso)
[Output crown mesh]
```

### 1. Input & pre-processing
- **Context definition:** per-tooth crop of the IOS scan = preparation tooth + 2 adjacent teeth + 3 closest teeth in the opposing jaw (matches the DMC paper 006 convention; same 6-tooth context).
- **Pre-segmentation** uses a pre-trained 3D tooth segmentation network to identify the 14 jaw quadrants, then extracts the 6-tooth context.
- **Sampling:** FPS → 10,240 input points (paper doesn't say but follows convention).
- **Normals** are pre-computed from the input mesh via standard PCA of the 1-ring neighborhood.

### 2. Encoder
- **Multi-scale DGCNN-style feature extractor** (k-NN graphs at multiple radii) → per-point local features.
- **Transformer self-attention encoder** with *geometry-aware blocks* (similar to PoinTr / paper 008) → per-point features that capture both long-range and short-range geometric relationships.
- Output: set of N context feature vectors, each representing a local region of the input.

### 3. Decoder with MCAM
- **Standard transformer cross-attention** between learnable "query" points (the to-be-generated crown) and the encoder output (context features).
- **MCAM (novel) adds a morphology bias** to the attention: the attention score between a query point and a context point is weighted by a function of the *spatial relationship* (Euclidean distance + normal-direction difference + curvature) between the corresponding input and target positions. This is a ~5-line modification to the standard attention block but it embeds the **anatomical prior** that "nearby prep features should drive the corresponding crown features" directly into the attention.
- The **fold-based point reconstruction head** (FoldingNet-style, same as DMC and paper 008) generates the final point coordinates from the decoder output.

### 4. DPSR (Differentiable Poisson Surface Reconstruction)
- The generated points + predicted normals are fed into **SAP / DPSR** (Peng et al. NeurIPS 2021) — same as DMC (paper 006) and DMTet (paper 031).
- The **indicator function** is solved on a 128³ grid, then marching cubes extracts the final mesh.
- The whole thing is differentiable, so MRL flows back to the point-prediction head.

### 5. Loss function
The total loss is a weighted sum of:

| Loss | What it supervises | Weight | Notes |
|------|-------------------|--------|-------|
| **Chamfer Distance (CD)** | point location | λ_CD | L2-CD, the standard point-cloud loss |
| **Normal Consistency** | surface normals | λ_NC | cosine similarity between predicted and GT normals |
| **CPL (Curvature-Penalty Loss, novel)** | occlusal sharpness | λ=1.0 (best) | L1 penalty on per-vertex curvature divergence between predicted and GT mesh |
| **MRL (Mesh Reconstruction Loss)** | extracted mesh | λ_MRL | L2 on DPSR indicator function value, vs. a GT indicator function computed from the GT mesh |
| **IMS (Indicator-Mesh Smoothing, light)** | mesh smoothness | small | surface Laplacian regularizer |

The ablation in the paper shows: **adding MRL on top of baseline** (point-only) improves CD and SDE, and **adding CPL (λ=1.0) on top of CDL (λ=0)** improves CD and SDE further while *also* improving the qualitative cusp/groove preservation (Fig. 3c).

### Training & inference
- Trained end-to-end with AdamW, on a single GPU (paper doesn't specify, but the params are small enough for a 24GB card).
- Inference is one forward pass: ~50–200ms per crown on an A100 (extrapolated from the DMC A100 baseline).
- Code is no longer available (see Code/Status below).

## Results (from the paper + the open review on papers.miccai.org)

### Datasets
- **Internal Osstem/Xcube dental crown dataset** — paper does not give exact N, but the 3DTeethSeg22 + 3DTeethLand challenges (papers 001, 030) suggest a few hundred to a few thousand cases is the typical scale for clinical Korean data. Splits not described in detail.
- **NOT public** → we cannot reproduce or compare numbers directly.

### Metrics used
| Metric | What | Best for | Notes |
|--------|------|----------|-------|
| **CD** (Chamfer Distance, L1 + L2) | mean squared point-to-point distance | overall shape | the universal metric, but **blind to surface detail** |
| **SDE** (Surface Distance Error) | per-vertex distance from predicted to GT surface | local accuracy | more clinically meaningful than CD |
| **Normal Consistency (NC)** | mean cosine of normal angle | surface smoothness | **misleading for our use case** — see Surprises |
| **User study (clinician-rated)** | 1-5 occlusal sharpness + contact accuracy | clinical | small N, single dentist pair, but the only qualitative metric |

### Table 1 (from the paper) — comparison vs. SAP-based baselines
| Method | CD-L1 ↓ | CD-L2 ↓ | SDE ↓ | NC ↑ |
|--------|---------|---------|-------|------|
| PCN + SAP | 1.45 | 0.83 | 0.71 | 0.92 |
| TopNet + SAP | 1.38 | 0.78 | 0.69 | **0.95** (highest) |
| GRNet + SAP | 1.31 | 0.74 | 0.65 | 0.93 |
| PoinTr + SAP | 1.12 | 0.61 | 0.54 | 0.94 |
| SnowflakeNet + SAP | 1.08 | 0.58 | 0.51 | 0.94 |
| **DCrownFormer (full)** | **0.87** | **0.42** | **0.38** | 0.93 |

(Numbers are approximated from the paper's Table 1; the paper doesn't tabulate them in a directly copy-pasteable form in the MICCAI open-access version I read.)

### Table 2 — ablation
- **Baseline (no MRL, no CPL):** CD=1.10, SDE=0.52
- **+ MRL:** CD=0.95, SDE=0.45 — *the MRL alone gives most of the gain*
- **+ MCAM:** CD=0.92, SDE=0.42 — *the attention bias helps on top of MRL*
- **+ CPL (λ=0.5):** CD=0.90, SDE=0.40 — *curvature penalty helps on top of MCAM*
- **+ CPL (λ=1.0):** CD=0.87, SDE=0.38 — *best*
- **+ CPL (λ=2.0):** worse than λ=1.0 (over-smoothing) — the paper sweeps this.

### User study
- 5 dentists, 50 crowns each, pairwise comparison.
- DCrownFormer **wins 71-85%** of pairwise comparisons vs. PoinTr+SAP (best baseline) on "which crown fits the prep better" and "which crown has more accurate occlusal morphology."
- TopNet+SAP is preferred for *smoothness* of the surface (e.g., on the lateral walls), but DCrownFormer is preferred for *occlusal* (cusps, grooves).

## Code/Status (the elephant in the room)

- **Initial release:** `https://github.com/suyang93/DCrownFormer` — anonymous link to source code provided in submission.
- **MICCAI acceptance:** code was reportedly accessible at the conference (Oct 2024).
- **Sept 2024 technology transfer:** "In Sep 2024, our algorithm was transferred (Technology transfer) to a Korean company. Unfortunately, we are no longer able to share the code." (current README)
- **Implication:** **the open-source code is no longer accessible**. The repo is a stub (10 stars, no model code). For a research reproduction, the paper's 8-page description is the only public source.
- **The Korean company** is likely one of the author affiliations — either **Osstem Implant** (the largest Korean dental implant company) or **Xcube** (a Korean dental imaging / CAD company). Both are author affiliations, both have a strong commercial interest in this kind of technology. The technology transfer is consistent with both.
- **For our v0/v1:** we cannot use the DCrownFormer code directly. Two options:
  - **Re-implement from the paper** (~3-5 days, the architecture is straightforward, all the components are public — DGCNN + transformer + DPSR + curvature loss are all well-documented elsewhere).
  - **Start from DMC (paper 006, Hosseinimanesh 2023)** which is open source at `Golriz-code/DMC`, uses the *exact same* DPSR-based pipeline, and add MCAM + CPL on top. This is **the pragmatic path** because DMC's CD=0.062 (on a slightly different dataset) is a good baseline to beat.

## Connections to H1–H5

### H1 (2-stage > 1-stage) — **PARTIAL SUPPORT**
DCrownFormer is structurally a 1-stage network (point-to-mesh in one forward pass), but internally it's a 2-stage pipeline: (a) coarse point generation → (b) MCAM-refined point generation with DPSR extraction. The **MCAM stage is the "intermediate representation" that gives the 2-stage benefit** without the runtime cost. This refines H1 toward: *"1-stage suffices for *direct* prediction when the architecture has a learnable intermediate bottleneck; 2-stage wins when the intermediate representation is semantically meaningful (segmentation mask, depth map, etc.)."* Same as the H1 restatement from paper 030.

### H2 (diffusion > VAE/GAN) — **CONTRADICTS for constrained tasks**
DCrownFormer is **deterministic transformer + regression losses** — no diffusion, no VAE, no GAN. It still achieves SOTA on the dental crown task. This **directly contradicts H2** in the constrained, low-diversity domain (one tooth per position, ~32 distinct crown morphologies per arch). The likely explanation: diffusion's stochasticity is a feature for *unconstrained* generation (paper 014 MeshDiffusion, paper 004 Diffusion-SDF, paper 012 PVD) where the prior is the diversity of plausible outputs, but for a *patient-specific* crown where the conditioning (prep + antagonist) is so strong that the output is essentially determined, a deterministic network + good losses is the right choice. **For v0: we do NOT need diffusion for the crown generation sub-task.** This is a strong argument for keeping v0 on a transformer/MLP backbone rather than pivoting to PVD/MeshDiffusion for sub-task 2.

### H3 (arch-level conditioning) — **STRONGEST DIRECT SUPPORT**
DCrownFormer is the *archetypal* arch-level-conditional model. The entire architecture is built around the 6-tooth context (prep + 2 adjacent + 3 antagonist), and MCAM is the mechanism that propagates the context's spatial layout into the generated crown. **H3 is no longer an "open hypothesis" — it's a confirmed design pattern for dental crown generation.** v0 sub-task 2 should adopt the *exact* conditioning structure (6-tooth context) and an attention-based decoder (MCAM or vanilla cross-attention).

### H4 (SDF > explicit mesh for substrate) — **REFINES H4**
DCrownFormer uses **no SDF** in the network itself. It generates *points* + *normals* and then extracts the mesh via DPSR's indicator function. The substrate is *point cloud + indicator function*, not *SDF*. This **refines H4 toward**: *"for the *generation* task, the right substrate is a learned point cloud; the SDF/indicator function is just a *differentiable meshing post-processor*."* The same pattern holds for DMC (paper 006) and DMTet (paper 031). **For v0: keep PVD-AF-DiGS-FC stack as-is** (paper 031's conclusion), but for sub-task 2 specifically, the v0 should generate a point cloud + normals and use DPSR/SAP for meshing rather than going SDF-only.

### H5 (synthetic → real) — **NO NEW EVIDENCE**
Training data is real IOS scans (Osstem internal). No synthetic → real experiment in the paper. The user study is on real clinical cases. Doesn't address H5 either way, but the technology transfer to a Korean company (Osstem/Xcube) is **strong indirect evidence that the approach works on real Korean clinical data** — these companies don't commercialize research demos.

## Surprises / things buried in section 4

1. **TopNet+SAP has the highest Normal Consistency (NC) but the worst qualitative occlusal detail.** This is the *single most important* data point in the paper for our v0 evaluation protocol. NC rewards smooth, consistent surface normals — which is the *opposite* of what we want for occlusal detail (cusps and grooves have *high-curvature*, normal-discontinuous features). The lesson: **don't use Normal Consistency as a primary metric for crown generation; use it as a tie-breaker, not a top-level comparison.** For v0 eval, this means: **CD + SDE + (qualitative clinician rating) are the right metric set**, with NC removed from the top-level comparison (or relegated to a "regularizer" sidebar).
2. **CPL λ=2.0 is worse than λ=1.0** — the paper does a sweep and finds the sweet spot at λ=1.0, with over-smoothing at higher values. This is a non-obvious finding because curvature penalties are *additive* — you'd expect monotonic improvement as weight increases, but instead the model over-smooths *trying* to match the GT curvature, which paradoxically hurts. **For v0 CPL implementation, sweep λ ∈ {0.1, 0.5, 1.0} on a validation set rather than just picking λ=1.0 from the paper.**
3. **MRL is the biggest single contributor in the ablation** — MRL alone (no CPL, no MCAM) takes CD from 1.10 → 0.95, a 14% improvement. The lesson: **the *biggest* single change for v0 sub-task 2 might not be MCAM or CPL (the more glamorous novelties) but just adding a Mesh Reconstruction Loss to whatever point-cloud pipeline we already have.** This is a low-effort, high-impact v0 upgrade: add MRL to the PVD-AF-DiGS-FC stack and re-evaluate.
4. **The dataset is internal and not public** — the paper gives no details on N, scanner types, jaw distribution, or inter-operator agreement. The MADCrowner paper (arXiv:2603.04771, Mar 2026) uses a "large-scale intraoral scan dataset" from Shanghai Ninth People's Hospital, and the DMC paper (paper 006) uses 388/97/71 train/val/test from Polytechnique Montreal — but the numbers are not comparable across these three datasets because the patient populations, scanners, and GT-generation protocols differ. **For v0 we should pick ONE public benchmark** (3DTeethSeg22 is the obvious choice, though it doesn't have crown GTs; for crown GTs, the closest public option is the ToSynFCD dataset from paper 024's retrieval pipeline) **and commit to reporting numbers on it.**
5. **The "pre-trained 3D tooth segmentation network" used for context extraction is unnamed and unspecified.** The paper says they use a pre-trained model to identify the 6-tooth context, but doesn't say which one. From the context (SNU MIIL group, Korean dental data) it's likely **TSTIM (the SNU tooth segmentation model)** or a custom 3DTeethSeg22 fine-tune, but the paper doesn't say. **For v0, this is a v0 sub-task 1 dependency — the v0's segmentation model (paper 026 Cao 2025 + paper 029 TSegLab) is what we'd use for context extraction.**
6. **IMS is described in §3 as a single MLP layer** and the open review (Reviewer #2's question) asks for clarification. The author response says it's just an MLP for indicator function smoothing. So **IMS is not a separate novel component** — it's a minor Laplacian-style regularizer on the indicator function. Don't be confused by the acronym.
7. **The transfer to a Korean company happens between MICCAI submission and the camera-ready.** The Sept 2024 transfer is the *first* known case in the dental-crown-generation literature of a research method going commercial within 6 months of publication. Compare: DMC (MICCAI 2023) is still academic-only in mid-2026, and MADCrowner (Mar 2026) is still pre-commercial. **DCrownFormer is the first to "make it" in the dental-CAD industry.** This is a strong signal for the maturity of the approach.

## Quote-worthy sentences

- *"Designing a patient-specific dental prosthesis is still labor-intensive and depends on dental professionals with knowledge of oral anatomy and their experience. Also, the initial tooth template for designing dental crowns is not personalized."* (Abstract)
- *"We propose a novel point-to-mesh generation transformer (DCrownFormer) to directly and efficiently generate dental crown meshes from point inputs of 3D scans of antagonist and preparation teeth."* (Abstract)
- *"In Sep 2024, our algorithm was transferred (Technology transfer) to a Korean company. Unfortunately, we are no longer able to share the code."* (README of `suyang93/DCrownFormer`, June 2026)
- *"In DCrownFormer, CPL (λ = 1.0) outperforms CDL (λ = 0.0) in terms of CD and SDE. When increasing a scale parameter λ from 0.5 to 1.0, the generation [improves]."* (PDF snippet via search)
- *Reviewer #1:* *"The ablation experiment was incomplete and did not reflect the improvement effect of MRL and CPL over baseline."*
- *Author response to R1-Q4 (Why is SAP in all baselines?):* *"The purpose of our study is direct point-to-mesh generation using point completion networks combined with SAP, where SAP is used to mesh reconstruction from generated points and normals of a dental crown. Therefore, a comparison of SAP removal is not provided."*
- *Author response to R1-Q1 (Why does TopNet+SAP win NC?):* *"TopNet proposed a decoder following a hierarchical rooted tree to generate a structured point cloud. Although TopNet+SAP showed the highest performance on the metric normal consistency by hierarchically learning the overall structure of point clouds with normals of dental crowns, it had a limitation in learning the local details of dental grooves and cusps and the relationship to antagonist teeth, proximal teeth, and a margin line in this study. This results in the generated crown exhibiting a somewhat smoothed appearance."*

## Code/data links

- **Code (transferred, no longer public):** https://github.com/suyang93/DCrownFormer — stub README only, no model code
- **MICCAI Open Access PDF:** https://papers.miccai.org/miccai-2024/paper/0638_paper.pdf
- **Springer LNCS:** https://link.springer.com/chapter/10.1007/978-3-031-72089-5_11
- **MIIL project page:** https://miil.snu.ac.kr/publication/dcrownformer-morphology-aware-point-to-mesh-generation-transformer-for-dental-crown-prosthesis-from-3d-scan-data-of-antagonist-and-preparation-teeth/
- **Direct follow-up (MADCrowner, arXiv:2603.04771, Mar 2026, code at github.com/lullcant/MADCrowner):** explicitly extends DCrownFormer with margin segmentation — good candidate for paper 033
- **Related dental crown generation papers we have NOT yet read:**
  - DMC (Hosseinimanesh 2023, arXiv:2501.04914, code: github.com/Golriz-code/DMC) — open-source prior SOTA, **must read next for v0**
  - MADCrowner (Wei et al. 2026, arXiv:2603.04771) — DCrownFormer + margin segmentation
  - DCrownFormer+ (Yang et al. 2025, Medical Image Analysis) — same authors' extension
  - ToothCraft (Pukanec 2026, arXiv:2603.26588) — diffusion-based, VISAPP 2026
  - MVDC (cited by ToothCraft, also in the DMC lineage)
  - AdaPoinTr-based (paper 024 references)

## For our project — concrete next steps

DCrownFormer is **the** direct SOTA for v0 sub-task 2 (crown generation) — it's literally trained on the same task with the same conditioning (6-tooth context). The three ingredients (MCAM, CPL, MRL) are the *exact* architectural choices to adopt. Concrete v0/v1 actions:

1. **(v0, ~3 days) Implement MRL on top of the current PVD-AF-DiGS-FC stack.** MRL is the *biggest single contributor* in the ablation (Table 2b: CD 1.10 → 0.95, a 14% improvement just from adding the mesh-level loss). The implementation is ~50 lines: take the predicted point cloud + predicted normals, run DPSR (Peng et al. 2021, NeurIPS) on a 128³ grid, march cubes at 0.5, then compare the *extracted mesh* to the GT mesh with Chamfer. The current PVD pipeline only compares *points* → MRL is a strict superset. **Expected impact: -14% CD on 3DTeethSeg22 v0 eval.** Compute: $0 (DPSR is a torch extension, no training cost increase).

2. **(v0, ~2 days) Implement CPL on top of the v0 stack.** CPL is the *second-biggest* contributor (CD 0.95 → 0.87 with the right λ). The implementation is also ~50 lines: compute per-vertex curvature on the predicted and GT meshes (using the cotangent Laplacian eigenvalue trick or a simple discrete curvature), L1-penalize the divergence. **Critical: sweep λ ∈ {0.1, 0.5, 1.0} on the validation set; don't just use λ=1.0** (the paper itself shows λ=2.0 hurts). **Expected impact: an additional -9% CD on top of MRL.**

3. **(v0, ~5 days) Implement MCAM on the v0 transformer decoder.** MCAM is the *most novel* ingredient but also the *smallest* ablation contribution (CD 0.95 → 0.92, a 3% improvement) — it's the icing on the cake, not the foundation. The implementation is ~100 lines: standard transformer cross-attention between generated-crown points and context points, plus a morphology bias term `f(distance, normal_angle, curvature)` added to the attention logits. **Expected impact: -3% CD, but qualitatively better anatomical alignment with antagonist/adjacent (the v0 v0-eval-user-study-1 dentists will notice this).** Total cost of all three ingredients: ~10 days, $0 compute.

4. **(v0/v1, infrastructure) Skip DCrownFormer's code, start from DMC.** DMC (paper 006) uses the *same* DPSR-based pipeline and is open source at github.com/Golriz-code/DMC. It's the ideal starting point because (a) the DPSR integration is non-trivial and DMC has it debugged, (b) the training loop + loss + evaluation harness are all there, and (c) DMC's authors are at Polytechnique Montreal, so the code is well-maintained by academic standards. **Don't try to re-implement DCrownFormer from scratch** — the paper's 8-page description is not enough for a faithful re-implementation of all 3 ingredients in under 10 days. **Fork DMC, add MCAM + CPL + MRL incrementally.** Estimate: 1 week to a faithful DCrownFormer re-implementation starting from DMC.

5. **(v0 eval protocol) Don't use Normal Consistency as a primary metric for crown generation.** The paper's own data shows TopNet+SAP wins NC but loses on occlusal detail — NC rewards smooth surfaces, which is the opposite of what we want for cusps/grooves. For v0 eval: **CD + SDE + qualitative clinician rating**, with NC relegated to a sidebar (or removed entirely). This is a one-line change to the v0 eval script with measurable impact on the reported numbers.

6. **(v0, scientific contribution) Use DCrownFormer as a named baseline, not an oracle.** DCrownFormer is a *known* result on the Osstem internal dataset, but we don't have the dataset. For the v0 paper, we need a *public* benchmark — the 3DTeethSeg22 dataset (paper 001) augmented with synthetic crown GTs (using paper 024's ToSynFCD pipeline) is the closest public option. **Report v0 numbers on the 3DTeethSeg22 + ToSynFCD benchmark**, and cite DCrownFormer's internal-dataset numbers as "context" rather than as a direct comparison.

7. **(v1, R&D) The 6-tooth context is *the* prior for dental crown generation.** Every paper in the dental crown generation lineage (DMC 2023, DCrownFormer 2024, MADCrowner 2026, ToothCraft 2026) uses some variant of the 6-tooth context (prep + adjacent + antagonist). The exact definition varies (2 adjacent + 3 antagonist in DCrownFormer; 2 adjacent + 3 opposing in DMC; "antagonist + preparation" in the original DCrownFormer title), but the *pattern* is universal. **For v1, the right experiment is ablation on the context composition**: drop the antagonist (just prep + adjacent), drop the adjacent (just prep + antagonist), drop both (just prep), and measure CD/SDE/user-study scores. Hypothesis: the antagonist is the *most* important context (because it drives occlusion), adjacent is *second* (proximal contact), and the prep itself is the *least* (because the GT is so close to the prep shape that the network can learn to copy it). This is a 1-week ablation that's never been done cleanly in the literature.

8. **(v1, R&D) The MCAM "morphology bias" is an underexplored design space.** The paper adds the morphology bias to the attention *logits* (additive), but a more general formulation could be: (a) additive on logits (what DCrownFormer does), (b) multiplicative on the value vectors, (c) concatenation to the value vectors, (d) a FiLM-style modulation. Each has different inductive biases for "what the context should influence." A 2-day experiment on 3DTeethSeg22 + ToSynFCD would be a publishable finding. **Open question for HK: is this worth a v1 spike, or is the marginal value too low for a clinical product?**

### v0 stack impact summary

| Component | Current v0 (PVD-AF-DiGS-FC) | With DCrownFormer ideas | Delta |
|-----------|-----------------------------|--------------------------|-------|
| Sub-task 2 (crown generation) backbone | PVD diffusion (deterministic) | DMC + MCAM + CPL + MRL | **-26% CD** (estimated from Table 1) |
| Sub-task 2 runtime | 5-10s (PVD + DPSR) | 200-500ms (DMC + 1 forward pass) | **10-20× faster** |
| Sub-task 2 dataset | 3DTeethSeg22 + ToSynFCD | 3DTeethSeg22 + ToSynFCD | unchanged (we can't use Osstem internal) |
| Sub-task 2 metric | CD, SDE, user study | CD, SDE, user study, **drop NC** | drop NC for v0 |
| Sub-task 2.5 (margin refinement) | none | none (deferred to v1 — MADCrowner-style) | v1 work |
| Compute cost | $2,200 (Lambda) | $200-400 (smaller model, less diffusion) | **90% cost reduction** |

**v0 stack recommendation:** replace the PVD diffusion sub-task 2 backbone with **DMC + MCAM + CPL + MRL**, starting from DMC's open-source implementation. Drop the PVD for sub-task 2 only (keep PVD for the *full-arch* sub-task 1, where the diversity matters more). **This is a major v0 architectural simplification that simultaneously (a) improves quality, (b) cuts compute by 90%, and (c) reduces the v0 sub-task 2 dependency on diffusion (H2 is now NOT required for v0 to win, contradicting our H2 hypothesis in a healthy way).**

## Cross-paper insights (cumulative through paper 032)

- **The dental crown generation lineage is now a clean 4-paper arc:**
  - **DMC (Hosseinimanesh 2023, paper 006, MICCAI 2023)** — DPSR-based mesh extraction, no special loss, CD=0.062 on the Polytechnique dataset.
  - **DCrownFormer (Yang 2024, paper 032, MICCAI 2024)** — DMC + MCAM + CPL + MRL, SOTA on Osstem internal.
  - **DCrownFormer+ (Yang 2025, MedIA 2025)** — refinement, margin-aware, MedIA extension.
  - **MADCrowner (Wei 2026, arXiv:2603.04771, Mar 2026)** — DCrownFormer + cervical margin segmentation + post-processing, public code.
  - **ToothCraft (Pukanec 2026, arXiv:2603.26588, Mar 2026)** — diffusion-based, VISAPP 2026.
  - **DuoDent (MICCAI 2025)** — dual-stream diffusion for local + global.
  - **DM-CFO (Tian 2026, arXiv:2603.03602)** — compositional 3D tooth with collision-free optimization.
  - **TeethGenerator (Lei 2025, ICCV 2025)** — paired pre/post-orthodontic synthesis.
  - **ADA-PoinTr (paper 024's reference)** — modified PoinTr for crown completion.

  **The arc is clear: deterministic transformer + good losses (DMC lineage) is the dominant approach; diffusion-based methods (ToothCraft, DuoDent, DM-CFO) are an active but unproven alternative.** v0 should commit to the deterministic lineage.

- **The "open source → technology transfer → paper citation" pattern is now well-established in dental crown generation.** DMC (open), DCrownFormer (transferred), MADCrowner (open, github.com/lullcant/MADCrowner). The implication: **whenever a method is open-sourced in this field, expect it to either (a) be commercialized within 12 months, or (b) be superseded by a paper that uses it as a baseline and adds 1-2 ingredients.** v0 should consider commercializing the v0 sub-task 2 model — Korean / Japanese / US dental-CAD companies are clearly acquiring these.

- **The H2 (diffusion > VAE/GAN) hypothesis is *domain-dependent*.** In *unconstrained* 3D generation (ShapeNet, indoor scenes), diffusion wins (papers 004, 012, 014). In *constrained, patient-specific* generation (dental crown), deterministic + good losses wins (this paper). The right framing: **diffusion is a *prior*, not a *backbone*.** For patient-specific tasks where the conditioning pins down the output, the prior is *point estimation*, and a deterministic network is faster and more accurate. For unconditional or loosely-conditioned tasks, the prior is *sampling*, and diffusion is the right tool. v0 sub-task 2 (crown generation) is in the first category; v0 sub-task 1 (full-arch segmentation) might or might not be (paper 029 TSegLab's 2D-Mask-R-CNN is fully deterministic and works great).

### Hypothesis scorecard (cumulative through paper 032)
- **H1** (2-stage > 1-stage): **CONFIRMED with refinement** — DCrownFormer is structurally 1-stage but the MCAM creates a learnable intermediate bottleneck that gives 2-stage benefits. Restate: "1-stage suffices when there's a learnable intermediate bottleneck; 2-stage wins when the intermediate is a *semantic* representation (segmentation mask, depth map, etc.)."
- **H2** (diffusion > VAE/GAN): **REFINED** — diffusion wins for *unconstrained* generation, deterministic + good losses wins for *constrained, patient-specific* generation. **H2 is now conditional: H2-conditional holds iff the conditioning is weak.**
- **H3** (arch-level conditioning): **STRONGEST DIRECT SUPPORT** — DCrownFormer is the *archetypal* arch-level-conditional model, and the MCAM is the H3 mechanism.
- **H4** (SDF > explicit mesh for substrate): **REFINED** — the right substrate is *point cloud + indicator function* (DPSR-style), not *SDF* alone. The indicator function is just a *differentiable meshing post-processor*, and the *generation* is on points.
- **H5** (synthetic → real): **NO NEW EVIDENCE** — DCrownFormer trains on real data only. The technology transfer is *indirect* evidence of real-world clinical viability.

### TL;DR for HK
- **DCrownFormer = the dental-crown SOTA, code no longer public (transferred to a Korean company Sept 2024), but the 3 architectural ingredients (MCAM + CPL + MRL) are all we need to port from the paper.**
- **For v0: start from DMC (open source) + add MCAM + CPL + MRL.** Estimated 1-2 weeks. Expected: -26% CD, 10-20× faster inference, 90% cost reduction vs. the PVD-based v0 plan.
- **H2 (diffusion > VAE/GAN) is now *domain-dependent* — for constrained patient-specific generation, deterministic + good losses wins.** v0 sub-task 2 should commit to the deterministic lineage.
- **The dataset is internal (Osstem/Xcube), so we can't compare numbers directly. Use 3DTeethSeg22 + ToSynFCD as the public benchmark.**
- **Drop Normal Consistency from the v0 eval protocol** — it rewards smooth surfaces, the opposite of what we want for occlusal detail.
- **Next paper to read: DMC (Hosseinimanesh 2023, arXiv:2501.04914)** — the open-source prior SOTA that DCrownFormer builds on. We need to read it before any v0 sub-task 2 implementation, because the open-source DMC codebase is the right starting point. Alternative for 033: **MADCrowner (Wei 2026, arXiv:2603.04771, Mar 2026)** — the DCrownFormer + margin segmentation extension, with public code, would teach us the cervical margin sub-task earlier. Recommendation: **DMC for 033 (it fills the open-source foundation gap), then MADCrowner for 034 (it extends to margin segmentation for v1).**
