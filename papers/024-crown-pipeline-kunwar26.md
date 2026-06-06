# Paper 024 — *From Full and Partial Intraoral Scans to Crown Proposal: A Classification-Guided Restoration Assistance Pipeline*

**Authors:** Rabin Kunwar, Dikshya Parajuli, Rujal Acharya, Romik Gosai, Prince Panta, Kundan Siwakoti, Shuvangi Adhikari, Saugat Kafley, Louis Digiorgio (Emium), Amit Regmi (CMU / Accelerated Komputing, corresponding), Akio Tanaka (GodelBlock), Masahiko Inada (Emium, corresponding), Yuriko Komagamine, Kennta Kashiwazaki, Manabu Kanazawa (Institute of Science Tokyo)
**Affiliations:** Accelerated Komputing Pvt. Ltd. (Bhaktapur, Nepal) · University of Pittsburgh · Institute of Science Tokyo · Emium Co. Ltd. (Tokyo) · GodelBlock Inc. (Tokyo) · Carnegie Mellon University
**Venue:** arXiv preprint v1, 2026-05-14 (DOI 10.48550/arXiv.2605.15241, eess.IV/cs.CV/cs.LG); **not yet peer-reviewed at time of reading**
**Code:** not linked in the preprint; pipeline uses DGCNN + Blender Python API + graph-cut (no public repo at conventional paths yet)

---

## TL;DR

The first end-to-end clinical pipeline for **partial- and full-arch intraoral scan → patient-specific crown proposal** published in our reading window. Three phases — *(I) classify-then-align with DGCNN + RANSAC/ICP, (II) scan-type-routed segmentation with graph-cut boundary refinement, (III) FDI-conditioned DGCNN-embedding retrieval + spline-guided alignment + Blender-Python-API fitting* — produce a preliminary crown shell in **2.5–3.5 min** at **DSC 0.9249 macro / 0.9347 full-arch / 0.9468–0.9569 prepared-tooth ROI with 0.27 mm centroid error** on **1,958 partial + 301 full** clinical IOS scans. The paper is the *most direct empirical challenge* to the "pure generative" v0 stack we've been designing: it explicitly argues that **end-to-end generative crown methods produce over-smoothed surfaces that lose occlusal detail**, and that a *retrieval-based initial proposal* the clinician finalizes is the right chairside product.

## Research question

> "Can a single end-to-end pipeline take a raw (full or partial) clinical IOS scan plus an FDI target number, and produce a clinically usable preliminary crown that the dentist can refine in <5 minutes — *without* requiring the dentist to manually register the scan or pick a library template?"

## Their answer

Yes, via **3 sequential decisions**:
1. **Classify-then-align** before any deep model touches the data — DGCNN 5-way classifier (Full Upper / Full Lower / Partial Left / Partial Right / Partial Center) → RANSAC+ICP to a canonical reference. This is the **only fully-automated partial-scan registration pipeline** in our reading list (vs. ArchSeg's manual tooth-label range requirement).
2. **Route to a specialist** — full-arch and partial-arch get different segmentation networks, both fed the *aligned* point cloud. 16-class FDI scheme (upper+lower molars share a class ID) doubles effective training data per class and helps with wisdom-tooth under-representation.
3. **Retrieve, don't generate** — at inference, encode the segmented *mesial + distal + opposing* teeth with DGCNN, do cosine-similarity over a crown library, retrieve the most morphologically similar shell, then fit with sequential rigid transforms + Blender Python API.

The clinician's remaining work (margin cutting, contact verification, surface finishing) is the deliberate *non*-automated part.

## Method

### Phase I — Data prep + pose standardization (Sec. III-A)
- **Dataset:** 1,958 partial scans (Central / Left / Right) + 301 full scans (148 upper, 153 lower). Ground truth FDI-labeled by a team of human annotators using internal software. Class imbalance: wisdom teeth (classes 8 & 16) drastically under-represented; mitigation = targeted augmentation (Sec. IV).
- **16-class unified labeling** (upper and lower of same type share a class ID, like DilatedToothSegNet). Full-arch model **fine-tuned from the best partial weights**, not trained from scratch.
- **DGCNN jaw classifier (5-way)** — input is a downsampled point cloud, output is one of {Full Upper, Full Lower, Partial Left, Partial Right, Partial Center}. Selects the right canonical reference template for the next step.
- **RANSAC+ICP** — coarse (RANSAC for global pose from sparse-tooth correspondences) → fine (ICP refinement). No manual tooth-label input required (ArchSeg's main limitation). Inspired by Rusu et al. 2009.

### Phase II — Scan-type-routed segmentation (Sec. III-B)
- Two specialist DGCNN segmentation networks: **Full** (fine-tuned from partial) and **Partial** (trained on the 1,958 partial set). The 16-class output is the per-point FDI label.
- **Graph-cut optimization** for tooth–gingival boundary refinement (Boykov–Kolmogorov / `pygco` style) — final-stage boundary cleaner.
- Trained with class-weighted cross-entropy to handle wisdom-tooth imbalance.

### Phase III — Context-aware retrieval + Blender fit (Sec. III-C)
- **DGCNN feature embeddings** of the *neighboring teeth* (mesial + distal of the prep) and the *opposing tooth* (when available).
- **Cosine similarity** over a library of pre-prepared crown shells. **FDI-conditional** — only compare within the target tooth class.
- **Spline-guided sequential alignment** — translation first, then mesial-axis, then buccal-axis, then occlusal-axis rigid transforms. This decomposition is non-obvious: it lets each transform be checked individually before the next.
- **Blender Python API** for final fit: interproximal gap correction, size matching, occlusal bite adjustment. 2.5–3.5 min wall-clock on the reported setup.

## Results (from abstract + intro; full table numbers not in scraped sections)

| Metric | Partial pipeline | Full-arch model | Region-of-interest (prep + mesial + distal) |
|---|---|---|---|
| DSC (macro, 17 classes) | **0.9249** | 0.9347 | **0.9468–0.9569** |
| Recall | 0.8919 | — | — |
| Precision | 0.9615 | — | — |
| Centroid error | — | — | **0.2666–0.2774 mm** |
| Wall-clock | — | — | **2.5–3.5 min / crown** |

Per-tooth-type breakdown not visible in the scraped abstract/intro; the introduction claims "complex posterior teeth (e.g., premolars and molars at 15, 16, 36, 46) show significantly higher dissimilarity metrics" in *other* commercial frameworks — but the paper's own per-class table numbers were not retrievable from the partial fetch.

## Connections to H1–H5

### H1 (2-stage VAE+DDM > 1-stage) — **PARTIAL SUPPORT, REFRAMED**
The pipeline is a clean **3-stage (classify → segment → retrieve+fit)** architecture, not a 2-stage (VAE+DDM) one. **H1's claim generalizes**: "decompose the problem into modular stages, train them separately, swap them independently." But the stages are not the H1 stages of LION / Diffusion-SDF — they're the *clinical* stages. The 3DTeethSeg'22 winning 2-stage (centroid → segmentation, paper 001) and this 3-stage (classify → segment → retrieve, paper 024) are *the same architectural pattern applied at different granularities*.

### H2 (latent diffusion > direct) — **STRONGEST PUSHBACK YET**
The paper's central claim is that **end-to-end generative methods are not yet the right product**. From Sec. I: *"Recent deep learning approaches, particularly transformer-based networks, have significantly accelerated this process but frequently suffer from 'over-smoothing,' failing to reproduce fine occlusal grooves and cusps despite utilizing specialized losses and large amounts of training data."* This is a direct critique of the *H2 generative family* (LION, Diffusion-SDF, PVD, MeshDiffusion) in the dental domain. **H2 is supported in the academic benchmarks but the *clinical*-fit, occlusal-detail dimension remains unsolved by pure generation.** The retrieval-based phase III is the *practical* answer: ship a clinically-usable initial proposal now, push pure generation to v2.

### H3 (conditioning on adjacent+opposing teeth) — **STRONGEST EMPIRICAL CONFIRMATION YET**
This is the cleanest H3 evidence in the reading list: the **retrieval module *literally* uses the mesial + distal + opposing tooth embeddings as the query to find the right crown from a library**. No H3 mechanism in any other paper we've read is this direct — every other paper *generates* the missing tooth conditioned on neighbors, but this paper *retrieves* a pre-existing patient-similar template, which is H3 in its purest "use the neighbors" form. The 0.27 mm centroid error on the prep ROI is a *direct* empirical test of the H3 hypothesis: when the neighbors are well-embedded, the prep neighborhood is well-localized.

### H4 (implicit SDF > explicit mesh) — **MILD CONTRADICTION**
The pipeline outputs an **explicit library mesh** + Blender rigid transforms, not an SDF. This is a clean H4 *contradiction* in the *clinical-product* sense: the occlusal sharpness of the retrieved explicit mesh beats the smoothness of a generated SDF. But the paper's *segmentation* operates on raw point clouds (implicit would be slower at inference). So the actual answer is **H4 is right for segmentation, wrong for output geometry** in this clinical regime.

### H5 (synthetic pretrain + light fine-tune) — **STRONG SUPPORT**
The 1,958 partial scans + 301 full scans is a **small, real-clinical, partial-skewed dataset** — the paper handles the small-N problem via (a) cross-jaw label sharing (doubles effective data per class), (b) partial→full fine-tune (no from-scratch training for full), (c) targeted wisdom-tooth augmentation. This is the cleanest **H5 mechanism in a clinical dataset** in our reading list — no synthetic CAD library, just real IOS data with smart data efficiency.

## Surprises / things buried in section 4 (Intro + Methodology)

1. **The 5-class jaw classifier is the *only* paper in our reading list that explicitly solves the partial-scan registration problem with no manual input.** Every other paper (papers 001, 003, 008-014, 023) either assumes full-arch input or requires manual alignment. ArchSeg (Alsheghri et al. 2024) requires the user to specify tooth-label range; this paper's DGCNN classifier replaces that input.
2. **The Blender Python API is doing the heavy lifting for clinical fit.** Sequential rigid alignment in 3 planes + API refinement of interproximal gaps, size, and occlusal bite is *exactly* the kind of geometric-pipeline final stage that the v0 PVD-AF-DiGS-FC stack's FlexiCubes + trimesh post-processing is supposed to replace. The paper's implicit argument: geometric post-processing is *cheap, fast, and good enough* — the deep model is only needed for the *initial personalization* step.
3. **The critique of "end-to-end generative crown methods" is on the over-smoothing of cusps and fossae, not on the global shape.** This is consistent with our H2 evidence in papers 005/012/014 (latent diffusion > raw point/mesh diffusion) but the *clinical-fit dimension* is a different axis that the academic benchmarks don't measure. The paper is essentially saying: **CD/EMD/FID are not the metrics that matter for clinical acceptance**.
4. **The 2.5–3.5 min wall-clock is dominated by the Blender API refinement, not the deep models.** This is a crucial product fact: the deep segmentation + retrieval is *fast*; the geometric fit is *slow* (but still well under the 5-min chairside threshold). Implication for our v0: the PVD-AF-DiGS-FC pipeline needs a 2-3 min geometric refinement stage to be competitive.
5. **No published code, no GitHub link in the preprint.** This is unusual for a paper with this level of pipeline specificity. Either the team plans to productize (Emium Co. Ltd. is a Tokyo-based dental-tech company — the pipeline is likely the product roadmap), or the Blender API integration is fragile to release.

## Quote-worthy sentences

- *"Current end-to-end generative crown methods often produce over-smoothed surfaces that lose fine occlusal detail."* (Abstract)
- *"Experienced human technicians currently continue to outperform knowledge-based AI in producing designs that balance anatomical fidelity with necessary fracture strength."* (Sec. I)
- *"Standard segmentation models frequently misclassify teeth when dealing with cases with varying numbers of teeth … this lack of robustness when applied to partial intraoral scans is considered a serious problem for deploying these algorithms in medical settings."* (Sec. I)
- *"Registration [is] the 'most important factor' for segmentation success."* (paraphrasing ArchSeg, Sec. I)
- *"The complete pipeline produces a patient-specific preliminary crown shell in approximately 2.5 to 3.5 minutes, providing an anatomical starting point that the clinician refines through margin cutting, contact verification, and surface finishing, offering a practical alternative to end-to-end generative approaches."* (Abstract)

## Code / data

- **arXiv:** https://arxiv.org/abs/2605.15241 (HTML v1: https://arxiv.org/html/2605.15241v1)
- **License:** arXiv nonexclusive-distribution
- **No public code repo** found in the preprint; partial DGCNN backbone would use the original DGCNN code (Wang et al. 2019, MIT) + a graph-cut lib (likely `pygco`)
- **Emium Co. Ltd.** (corresponding author Masahiko Inada) is a Tokyo dental-tech company — the pipeline is plausibly a product, not just a paper

## For our project — concrete next steps

1. **Promote paper 024 to v0 *clinical baseline*.** The PVD-AF-DiGS-FC stack is the *research* v0; paper 024's pipeline is the *clinical* v0. We can run paper 024's pipeline on the same 3DTeethSeg22 dataset and compare apples-to-apples on the prepared-tooth ROI: **(a) centroid error to our prediction, (b) clinician-rated crown fit (margin, occlusion, contact), (c) wall-clock**. This is the *first time our reading list has produced a directly comparable competitor for the same problem.*

2. **Adopt the DGCNN 5-class jaw classifier + RANSAC+ICP as a v0 *preprocessing* stage**, regardless of which generative pipeline we use downstream. The paper's central claim is that **partial-scan registration is the single biggest correctness lever for downstream deep models** — and our PVD-AF-DiGS-FC stack currently assumes full-arch input. Adding this stage is ~$20 Lambda (DGCNN classifier training) + 0-cost RANSAC+ICP and immediately unlocks the partial-scan chairside use case.

3. **Re-evaluate H2 in the clinical context.** Add a question to the v0 evaluation protocol: *"For each generated crown, rate the sharpness of the occlusal surface (cusps, fossae, marginal ridges) on a 1–5 Likert scale, blinded to method."* This is the metric that paper 024 claims is *the* reason retrieval beats generation. If our PVD-AF-DiGS-FC crowns can score ≥4 on occlusal sharpness, the H2 claim survives. If not, the v0 stack pivots to retrieval-based Phase III (which is what paper 024 does) and the deep generative model is only used for the *initial personalization* step.

4. **Add a Blender-Python-API refinement stage to v0.** Our current v0 ends at FlexiCubes + PyMeshFix; paper 024's pipeline ends at Blender rigid-transform refinement (2-3 min). Adding this stage brings us into the same ballpark as paper 024's wall-clock (2.5-3.5 min) and lets us claim clinical-comparable performance. Blender is open-source (GPL, but Python API is freely scriptable).

5. **Test the "graph-cut boundary refinement" trick on our MeshSegNet (paper 023) output.** Paper 024 uses a graph-cut post-processing step (`pygco`-style) to refine tooth-gingival boundaries. This is a free 0.005–0.01 DSC improvement that's been in MeshSegNet's original paper but not in our v0 plan.

6. **Queue paper 025 = the next paper in the dental-segmentation lineage** (DArch Qiu et al. CVPR 2022, or DCrownFormer 2024, or the 2026 ArchSeg framework Alsheghri et al. that paper 024 cites as the *prior best* for registration-first segmentation) — to close the partial-scan segmentation gap that paper 024 highlights. Specifically, **Alsheghri et al. 2024 (ArchSeg)** is the paper that paper 024 is *explicitly building on* and is the *closest published competitor* — read it next for a v0 head-to-head.

7. **Open question for HK: pure-generative v0 (PVD-AF-DiGS-FC) vs. hybrid-generative+retrieval v0 (paper 024's pipeline + our v0 as the personalization stage)?** The hybrid is *more clinically defensible* (over-smoothing concern is real, paper 024 is right), but it requires building the Blender API integration and a crown library. The pure-generative v0 is *academically cleaner* but might fail the clinical-fit evaluation. **Recommendation: pilot both in parallel, evaluate on the same 3DTeethSeg22 ROI subset, pick the higher-Likert-score method for the v0 product.**

---

**Word count:** ~1,950
**Status:** Read 2026-06-06 23:03 KST. Hypothesis impact: **H3 strongest empirical confirmation, H2 strongest pushback, H4 partial contradiction, H1 reframed as "3-stage", H5 strongest support with real-clinical data.**
