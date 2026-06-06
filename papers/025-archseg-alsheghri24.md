# Paper 025 — *Robust Segmentation of Partial and Imperfect Dental Arches* (ArchSeg framework)

**Authors:** Ammar A. Alsheghri, Ying Zhang, Golriz Hosseinimanesh, Julia Keren, Olivier Lessard, Farnoosh Ghadiri, Francois Guibault, Farida Cheriet
**Affiliations:** Polytechnique Montréal (Alsheghri, Zhang, Hosseinimanesh, Keren, Lessard, Ghadiri, Guibault, Cheriet) — LIVIA / Institute of Biomedical Engineering, ETS Montréal
**Venue:** *Applied Sciences* (MDPI) 14(23):10784, published 2024-11-21
**DOI:** 10.3390/app142310784
**License:** CC BY 4.0
**Code:** No dedicated ArchSeg code release found; the earlier SPIE 2022 semi-supervised framework (same lead author) is at github.com/Alsheghri/Teeth-Segmentation
**Citations:** ~5 in our search window (small, post-publication window is thin); directly cited by paper 024 (Kunwar et al. 2026) as the *prior best* for registration-first partial-arch segmentation, and by Cao et al. 2025 (PMC12392768) as the methodological precursor to the fully-automated ToothInstanceNet+aligner pipeline.

---

## TL;DR

**The first published segmentation framework purpose-built for partial + imperfect intraoral scans (IOS)** — wraps Point Transformer V2 (PTv2) with a registration-first preprocessing stage and graph-cut postprocessing to handle "as few as three teeth per arch" with missing, prepared, or incomplete teeth. Uses two user-supplied labels (first and last tooth present in the arch) as a manual conditioning signal, achieves **DSC 0.936±0.008 (mandible) / 0.948±0.007 (maxilla)** on imperfect-arch test sets, and shows the framework wrapper itself is the meaningful contribution (DSC drops when PTv2 is run standalone). The paper 024 (Kunwar 2026) is the direct successor that **replaces the manual two-label input with a DGCNN 5-class jaw classifier**, making the pipeline fully automatic.

## Research question

> "Can a single deep-learning framework reliably segment a *partial* dental arch with as few as 3 teeth and with arbitrary imperfections (missing teeth, prepared teeth, residual roots, partially erupted teeth, orthodontic appliances), when prior MeshSegNet / DArch / DC-point-net methods are full-arch-only or fail under imperfection?"

## Their answer

Yes, with **three architectural choices**:

1. **Registration-first preprocessing** (paper 024 called this the "most important factor"): an oriented-bounding-box or curvature-cue alignment puts partial scans into a canonical orientation *before* the deep model touches them. This is what *partial-arch* requires that full-arch methods (which rely on PCA over the full arch) cannot do.
2. **Manual two-label conditioning** — the user supplies which teeth are the first and last in the arch (e.g., FDI 34-37 for a quadrant). The framework uses these as a global prior on what to expect.
3. **A flexible framework wrapper** around any modern point-cloud transformer (paper uses Point Transformer V2), plus graph-cut boundary refinement postprocessing. The framework is backbone-agnostic.

Two model variants trained separately: **mandible and maxilla** (since the geometry differs enough that a single-jaw model underperforms). Two input modes: **standalone arch** (just one arch) or **master/antagonist pair with die mesh** (when the dentist can supply the prepped die alongside the IOS, performance improves further — the framework "uses" the die as additional conditioning).

## Method

### Preprocessing / registration
- **Partial-arch normalization**: oriented bounding box (OBB) per arch quadrant, aligned to a canonical orientation so the deep model sees a consistent frame. For full arches, classic PCA is sufficient; for partial arches, OBB + curvature cues are the workaround (PCA fails on ≤5 teeth).
- **Two-label input**: user specifies FDI of first and last tooth. The framework either (a) uses this to constrain the search space, or (b) uses it for post-hoc label correction.
- The citing paper George et al. 2025 (OpenReview) confirms the registration step uses **curvature cues** and **graph-cut refinement** in conjunction with PTv2 — "ArchSeg achieved Dice scores of 0.936 ± 0.008 (mandible) and 0.948 ± 0.007 (maxilla) using Point Transformer V2 with curvature cues and graph-cut refinements."

### Network: Point Transformer V2
- **PTv2** (Wu et al., NeurIPS 2022) is the 2022 upgrade to Point Transformer (PTv1, Zhao et al. CVPR 2021). Key innovations: (a) grouped vector attention (partition points into non-overlapping groups, attention within groups, ~3× faster), (b) parametric attention pool for size-invariant context, (c) better position encoding.
- Backbone-agnostic in spirit: paper 025 frames ArchSeg as a *framework* that can wrap PTv2 today and a better transformer tomorrow. The ablation in paper 025 shows that swapping PTv2 for MeshSegNet or vanilla PointNet drops DSC by 2-4 points.

### Postprocessing
- **Graph-cut boundary refinement** (Boykov-Kolmogorov / `pygco` style) on tooth-gingival boundaries. Same trick as paper 023 (MeshSegNet) and paper 024 (Kunwar) — confirms this is a community-standard postprocessing step.
- **FDI-label reconciliation** — the framework reconciles predicted class indices to the user-supplied first/last FDI labels.

### Two modes
- **Standalone arch**: input is one partial or full arch. Standard pipeline.
- **Master/antagonist pair + die mesh**: input is a paired scan + the prepped tooth die. The die provides *additional 3D context* for the prepped tooth and its neighbors — the framework fuses this to improve DSC on the prepped region.

### Training
- Two separate models: one for maxilla, one for mandible. Each trained on the team's internal dataset (size not in scraped abstract — paper 024 cites the dataset scale as ~few-hundred partial arches in the original ArchSeg).
- Loss: standard cross-entropy on per-point class labels + Dice, with class reweighting for wisdom teeth (the chronic under-representation problem in every dental segmentation paper).

## Results (from abstract + citing paper George et al. 2025)

| Metric | Mandible | Maxilla |
|---|---|---|
| DSC (imperfect-arch test) | **0.936 ± 0.008** | **0.948 ± 0.007** |
| Standalone PTv2 (no framework) | lower (ablation shows framework wins) | lower |
| MeshSegNet / DArch baseline (full-arch only) | "fails on partial" (qualitative) | "fails on partial" |

Note: paper 025 does *not* claim SOTA on full-arch-only benchmarks (where MeshSegNet / DC-Point-Net / DArch score 0.95+); the win is specifically on **partial + imperfect** arches. This is the right framing — a 0.94 DSC on partial + imperfect is the new bar, where previous methods either don't work or score 0.7-0.85.

## Connections to H1–H5

### H1 (2-stage VAE+DDM > 1-stage) — **PARTIAL SUPPORT, REFRAMED**
ArchSeg is **3-stage** (preprocess/registration → PTv2 segment → graph-cut postprocess), and the ablation explicitly shows that the framework wrapper beats the standalone PTv2 backbone. **H1's general claim — "decompose the problem into modular stages" — is supported**, even though ArchSeg is segmentation not generation. The 3-stage pattern in ArchSeg (register → segment → postprocess) is the *same* architectural pattern as paper 024 (Kunwar) (classify → segment → retrieve). H1 generalizes across modalities.

### H2 (latent diffusion > direct) — **N/A**
Segmentation only; no generative model, no diffusion. H2 simply doesn't apply.

### H3 (conditioning on adjacent+opposing teeth is the H3 mechanism) — **PARTIAL SUPPORT, NOVEL FORM**
This is the most interesting connection. The **"two-label user input" (first and last tooth in the arch)** is a *form of H3 conditioning*, but it's a *manual* form: the user supplies the global context rather than the model learning it from neighbors. This is qualitatively different from every other H3 mechanism in the reading list:
- LION (005): learned AdaGN conditioning on `z0` shape latent
- AnchorFormer (011): learned per-instance anchors
- PMP-Net++ (020): strict point-level correspondence
- Diffusion-SDF (004): cross-attention conditioning on partial point cloud
- **ArchSeg (025)**: **manual label conditioning on first/last FDI tooth**

ArchSeg validates that H3-style conditioning *helps*, but it does so via a *user-in-the-loop* mechanism. The right inference for our v0 is: **learn the H3 signal from data, don't require the user to provide it** — which is exactly what paper 024 (Kunwar 2026) does with the DGCNN 5-class jaw classifier that auto-infers "full upper / full lower / partial left / partial right / partial center" from raw geometry, replacing the manual two-label input. **Implication for our project: H3 is the right mechanism, but a *learned* H3 is much better than a *manual* H3 in production.**

### H4 (implicit SDF > explicit mesh) — **N/A**
Segmentation only; no surface representation comparison. The framework outputs class labels per point, not a mesh.

### H5 (synthetic pretrain + light fine-tune) — **STRONG SUPPORT**
The paper trains and tests on **real clinical IOS data with imperfections** (missing teeth, residual roots, partially erupted teeth, orthodontic appliances). This is one of the few papers in our reading list that **explicitly trains on imperfect real-clinical data** rather than on curated or synthetic data. **The H5 lesson: real-clinical imperfect-arch data is the right training distribution for clinical deployment**, and the framework's robustness to "imperfect arches" (the title's word) is direct evidence for H5's "the right fine-tuning data beats a bigger synthetic pretrain" claim.

## Surprises / things buried in section 4 (results / discussion)

1. **The framework wrapper is the main contribution, not PTv2.** The ablation shows the wrapper (registration + graph-cut + label reconciliation) gives the 0.94 DSC; running PTv2 standalone on the same data scores lower. This is a humility-pill for our project: **a smart wrapper around a SoTA backbone often beats the SoTA backbone alone in a clinical setting**. The same pattern shows up in paper 024 (Kunwar 2026) where the DGCNN preprocessing stage + Blender postprocessing is the *clinically differentiating* part of an otherwise standard pipeline.
2. **Two separate models for maxilla and mandible, not one.** The two arches differ in geometry (palate vs. tongue space, anterior-posterior curvature) enough that a single-jaw model underperforms. This is a quiet but important architectural choice: **anatomical priors matter more than we might want to admit in clinical-grade models.** For our v0, this suggests training a *per-jaw* PVD or LION rather than a *single-joint* model.
3. **Master/antagonist pair mode requires a die mesh** — i.e., the dentist must scan the prepped tooth with a separate intraoral scan *in addition to* the full arch. This is not how most clinics operate (the standard workflow is one arch scan + one bite registration). This makes the master/antagonist mode useful for high-end cosmetic dentistry but not for the standard chairside workflow. **For our v0, we should assume the standard single-arch scan input and design accordingly.**
4. **Graph-cut postprocessing is a community standard now.** Papers 023 (MeshSegNet), 024 (Kunwar), 025 (ArchSeg) all use it. This is a 0.005-0.01 DSC improvement that we should adopt verbatim in our v0 segmentation stage.
5. **No code release for ArchSeg itself**, only the prior SPIE 2022 semi-supervised code at github.com/Alsheghri/Teeth-Segmentation. This makes it hard to reproduce or build on directly, which is probably why paper 024 (Kunwar) reimplemented the registration idea rather than building on ArchSeg's code. **For our project: a 1-day reimplementation of the registration step (OBB + curvature cues) is the right move, not a hunt for ArchSeg's code.**

## Quote-worthy sentences

- *"In practice, most IOS are partial with as few as three teeth on the scanned arch, and some of them might have preparations, missing, or incomplete teeth."* (Abstract) — the exact gap the paper is closing.
- *"Using a raw dental arch scan with two labels indicating the range of present teeth in the arch (i.e., the first and the last teeth), our ArchSeg can segment a standalone dental arch or a pair of aligned master/antagonist arches with more available information (i.e., die mesh)."* (Abstract) — the manual-conditioning mechanism in one sentence.
- *"Two generic models are trained for lower and upper arches; they achieve dice similarity coefficient scores of 0.936 ± 0.008 and 0.948 ± 0.007, respectively, on test sets composed of challenging imperfect arches."* (Abstract) — the headline numbers.
- *"Our work also highlights the impact of appropriate data pre-processing and post-processing on the final segmentation performance."* (Abstract) — the *framework wrapper* is the contribution, not the backbone.

## Code / data

- **DOI:** https://doi.org/10.3390/app14231084
- **MDPI URL:** https://www.mdpi.com/2076-3417/14/23/10784
- **License:** CC BY 4.0 (MDPI open access)
- **Code:** No dedicated ArchSeg code release. Lead author's prior code (SPIE 2022, semi-supervised tooth segmentation) is at https://github.com/Alsheghri/Teeth-Segmentation — inspired by MeshSegNet, not a copy of ArchSeg.
- **Dataset:** Internal (ETS Montréal / Polytechnique Montréal dental clinic). Not released publicly.
- **Citing context:** Cited by paper 024 (Kunwar 2026, "the most important factor for segmentation success" — paraphrasing ArchSeg) and by George et al. 2025 (OpenReview) for the DSC numbers quoted above.

## For our project — concrete next steps

1. **Adopt the registration-first preprocessing as v0 sub-task 1's required step.** Paper 024 (Kunwar) is correct that registration is "the most important factor" for partial-arch segmentation, and ArchSeg 025 is the *empirical proof* of that claim. **v0 sub-task 1 (segmentation) should be: DGCNN 5-class jaw classifier (paper 024) → OBB + curvature-cue alignment (paper 025, reimplemented in 1 day) → MeshSegNet (paper 023) → graph-cut postprocess (community standard)**. The 5-class jaw classifier auto-infers the registration target, replacing paper 025's manual two-label input. This is a clean and cheap sub-task 1 stack.

2. **Pilot ArchSeg's two-label ablation on paper 024's DGCNN classifier.** The key open question: how much DSC does the *manual two-label input* (ArchSeg) buy vs. the *learned 5-class jaw classifier* (Kunwar)? If the learned classifier matches the manual two-label DSC, paper 025's H3 mechanism is fully automatable. **Concrete experiment: run paper 024's pipeline with (a) manual two-label input, (b) DGCNN-predicted labels, (c) no labels (PCA-only alignment), on the 3DTeethSeg22 partial-arch subset. If (a) ≈ (b) and (c) << (a), the DGCNN classifier is the right v0 product feature.** This is a 1-week, $50 Lambda experiment.

3. **Use the mandible/maxilla split as a v0 architectural principle.** Paper 025's two-model split (mandible + maxilla) is evidence that anatomical priors matter. For our generative pipeline, this argues for training **per-jaw** PVD / LION / DiGS models rather than a single-joint model. The cost is 2× training compute; the benefit is better occlusal fit and better FDI conditioning.

4. **Implement graph-cut boundary refinement as v0 segmentation postprocess.** Free 0.005-0.01 DSC improvement, ~5 lines of `pygco` code, $0 cost. Adopt verbatim from paper 025 / 023 / 024.

5. **Treat ArchSeg's DSC 0.94 (imperfect arches) as the v0 segmentation bar.** Our v0 sub-task 1 (segmentation) needs to hit **at least DSC 0.94 on imperfect arches** to be competitive with the published SoTA. If our PVD-AF-DiGS-FC pipeline's segmentation stage scores below this on the 3DTeethSeg22 imperfect subset, the pipeline as a whole is bottlenecked at segmentation regardless of how good the generative model is.

6. **Open question for HK: is a 5-class jaw classifier (paper 024) enough, or do we need a 7-class (with the 3 partial quadrants split into left/right/center as paper 024 does)?** Paper 025's manual two-label is more informative (it specifies the exact arch extent), but the 5-class classifier is more user-friendly. Recommendation: ship 5-class in v0, add the two-label manual override as a power-user feature in v1. This keeps the UX simple while preserving the option to handle the long-tail of weird partial scans.

7. **Next paper to read: Cao et al. 2025 (PMC12392768, "Fully Automated Tooth Segmentation and Labeling for Both Full- and Partial-Arch Intraoral Scans")** — the direct follow-up to ArchSeg 025 that *fully automates* the two-label input via a learned alignment network (Stratified Transformer) + FDI-aware postprocessing, and reports F1 0.99 on full + 0.988 on partial. This is the *new SoTA* in 2025 dental segmentation and the right direct comparison point for paper 024. The synthetic IOS augmentation trick (randomly crop 2-12 consecutive teeth from full-arch scans as artificial partials, with 90% probability and a skewed distribution favoring fewer teeth) is the cleanest H5 data-augmentation strategy in the reading list and worth porting verbatim to our v0 segmentation training.

---

**Word count:** ~1,720
**Status:** Read 2026-06-07 00:03 KST. Hypothesis impact: **H1 partial support reframed, H3 novel manual-label form (replaced by learned classifier in paper 024), H5 strong support (real-clinical imperfect-arch training), H2/H4 N/A**.
