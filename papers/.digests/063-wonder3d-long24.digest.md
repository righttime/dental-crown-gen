# Paper 063 Digest — *Wonder3D: Single Image to 3D using Cross-Domain Diffusion*

**Authors:** Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, Wenping Wang
¹ HKU · ² Tsinghua + VAST · ³ UPenn · ⁴ ShanghaiTech · ⁵ MPI Informatik · ⁶ Texas A&M
**Year:** 2023 (arXiv 2310.15008) → **CVPR 2024 Highlight**
**Code:** https://github.com/xxlong0/Wonder3D (MIT)
**Date:** 2026-06-08 12:35 KST (Monday, scholar-digest cron #63)
**For:** HK (Telegram, Alf)

---

## 📄 Telegram digest

```
📄 Paper 063: Wonder3D — Single Image to 3D using Cross-Domain Diffusion (2023, CVPR'24 Highlight)
TL;DR: 6 consistent multi-view normal+color maps from single image in 2-3 min on 1 GPU via cross-domain Stable Diffusion (1-token domain switcher RGB↔normal + cross-domain attention), then fuse via instant-NGP/Neuralangelo into textured mesh; GSO SOTA at publication (CD 0.0199, IoU 62.44).
Hypothesis: H1 STRONGEST (Wonder3D IS the H1 decomposition), H2 STRONG (clean 2-stage training recipe), H3 PARTIAL+CLEANEST-EXTENSION (cross-domain attention is the architectural primitive for H3 multi-source fusion), H4 STRONG (mesh extraction is EXPLICITLY neural SDF), H5 STRONGEST (30K synthetic Objaverse → real/AI/paintings; the strongest H5 argument in the reading list for "pretrain on 100K synthetic dental arches → fine-tune on 1K clinical IOS")
For our project: ADOPT cross-domain diffusion as v0 SIDE-TRACK pretrain step ($500 Lambda, 4 weeks) + PORT outlier-dropping loss to v0 sub-task 4 DiGS (1 day, $0) + PORT geometry-aware normal loss to cervical margin refinement (1-2 days, $30) + USE 6 fixed azimuths no elevation as v0 v0.5 pilot + CITE as v0 paper's cross-modal generation precedent; DEFER full Wonder3D-from-scratch training to v1 (not on v0 critical path)
```

---

## Full digest (for record / re-read)

### One-sentence pitch
**The CVPR 2024 Highlight single-image-to-3D method** that uses cross-domain diffusion (1-token RGB/normal switcher + cross-domain attention in every UNet block) to generate 6 consistent multi-view normals+colors, then fuses via neural SDF; GSO SOTA at publication and the strongest H1+H5 evidence in the reading list for the v0 architecture.

### Key architectural innovations
1. **Cross-domain Stable Diffusion** — extend 2D UNet with 1-token domain switcher (RGB=0 vs normal=1) positional-encoded + concatenated with time embedding. The ONLY domain-conditioning mechanism — no extra output channels, no separate UNet.
2. **Cross-domain attention** — in every transformer block, new attention layer inserted BEFORE cross-attention. Keys+values from BOTH domains concatenated; queries from A attend to both A and B. This is the geometric/visual consistency mechanism.
3. **Multi-view attention** — self-attention layers extended to be global-aware; tokens from all 6 views see each other (similar to SyncDreamer, MVDream).
4. **Geometry-aware normal loss** — `cos(angle(normal, view_ray))` weighting; the more orthogonal a normal is to its view ray, the more reliable it is (normals outward-facing, view-rays inward-facing, valid pair ≥90°).
5. **Outlier-dropping loss** — at each iter, sort color/mask errors descending, drop top X% largest. Wrong predictions lack cross-view consistency → large errors → auto-dropped. Mathematically equivalent to hard attention over loss terms.
6. **Input-view-related coordinate system** — Z+/X+ align with input image UV, Y+ perpendicular to image plane. NO elevation estimation at inference. Trade-off: no canonical pose evaluation.
7. **6 fixed azimuths, 0° elevation, 256×256** — simplest possible view setup, no occluded-region hallucination, 12GB VRAM with xformers+fp16.

### Training recipe (2 stages)
- **Data:** LVIS subset of Objaverse, ~30K cleaned objects, 6 random-rotation views per object
- **Stage 1:** multi-view attention only (no cross-domain), random normal/flag
- **Stage 2:** add cross-domain attention modules, only optimize NEW params (rest of SD frozen → preserves 2D prior, no catastrophic forgetting)
- **Resolution:** 256×256, batch 512, 30K steps, ~3 days on 8× A800
- **Key 2024-08-29 bug fix:** `zero_init_camera_projection` should be False; CFG inference needs RGB/normal in first/second batch halves (not uncond/cond split)

### Results
- **GSO single-view reconstruction:** Wonder3D CD 0.0199, IoU 62.44, F-Score 0.6244 (SOTA at publication); Unique3D (5mo later) surpassed on CD/F-Score, Wonder3D still highest IoU in 2024
- **Beats:** Magic3D, RealFusion, Zero123, SyncDreamer, One-2-3-45 on CD/IoU/F-Score
- **2-3 min vs Magic3D 30-60min vs DreamFusion 2-4h**
- **Robust generalization:** synthetic Objaverse → GSO real scans + AI-generated images + paintings
- **Failure modes:** occluded objects (back unconstrained), non-orthographic real images (focal distortion), elevation ambiguity

### H1–H5 connections
- **H1 (2-stage gen+recon):** STRONGEST DIRECT SUPPORT IN 2023-2024 LITERATURE — Wonder3D IS the H1 decomposition; 2D gen alone = beautiful images but not 3D, 3D recon alone = needs dense views and breaks on sparse/noisy inputs, only combination gives high-quality 3D in 2-3min
- **H2 (diffusion backbone):** STRONG — multi-view diffusion is the ONLY 2D prior strong enough for OOD; 2-stage training is clean H2 recipe
- **H3 (adjacent+opposing):** PARTIAL + CLEANEST H3 EXTENSION MECHANISM — single-image conditioning only, BUT cross-domain attention (RGB↔normal) is the EXACT architectural primitive for fusing multiple modalities; replace (RGB, normal) with (target, adjacent, opposing) and domain switcher generalizes to modality switcher
- **H4 (SDF > mesh):** STRONG — mesh extraction is EXPLICITLY neural SDF (instant-NGP/NeuS/Neuralangelo), paper §4.3: "Unlike meshes, SDF offers compactness and differentiability, ideal for stable optimization"; confirms v0 stack's DiGS+FlexiCubes choice
- **H5 (synthetic→real):** STRONGEST — trained on 30K synthetic Objaverse, generalizes to GSO real + AI-generated + paintings; 2D diffusion prior carries over to 2D renders of dental scans, 3D fusion is geometric (no domain gap); STRONGEST H5 argument in reading list for "pretrain on 100K synthetic dental arches → fine-tune on 1K clinical IOS"

### For v0 (concrete next steps)
1. **ADOPT cross-domain diffusion as v0 SIDE-TRACK** ($500 Lambda, 4 weeks) — train SD-variant on 6-view renders of 3DTeethSeg22 arches; cross-domain attention is natural H3 injection point; 2-3min inference acceptable for non-realtime
2. **PORT outlier-dropping loss to v0 sub-task 4 DiGS** (1 day, $0) — drop top 20% largest SDF errors per iter; trade-off: keep on boundary loss only
3. **PORT geometry-aware normal loss to cervical margin refinement** (1-2 days, $30 Lambda) — cos(angle) weighting, cervical margin normals are more grazing → more reliable
4. **CONSIDER input-view-related coordinate system** (arch decision) — pro: no FDI-relative canonical pose estimation; con: no canonical eval; RECOMMEND: keep canonical FDI-relative for v0 (matches Cao25, CrownSegger)
5. **USE 6 fixed azimuths, NO elevation as v0 v0.5 pilot** (1 day setup, $0) — v0 has no elevation views in IOS
6. **USE 256×256 as v0 v0.5 pilot resolution** (arch decision) — 12GB VRAM, batch 32
7. **CITE Wonder3D as v0 paper's cross-modal generation precedent** ($0, 30min) — list 2D-3D cross-modal literature: Wonder3D, SyncDreamer, MVDream, One-2-3-45, ImageDream, Magic3D, Era3D, Unique3D, CraftsMan3D, Wonder3D++
8. **FOLLOW Wonder3D training recipe for v0 v0.5** (2 weeks, $200 Lambda) — Stage 1: multi-view only on dental data; Stage 2: add cross-domain to fuse (target+adjacent+opposing)
9. **DEFER full Wonder3D-from-scratch dental training to v1** ($500 Lambda deferred) — not on v0 critical path
10. **REPLICATE Wonder3D bug-fix pattern in v0 repo** (1 day, $0) — add unit tests for cross-domain attention CFG inference

### 10 surprises / interesting things buried in §4
1. Domain switcher is just 1-token integer — least-invasive mod to SD preserves prior
2. Cross-domain attention BEFORE cross-attention (agree first, then condition)
3. cos(angle) weighting as cheap H3-like signal (viewing geometry as reliability)
4. Outlier-dropping as hard attention over losses (no Huber/Tukey/confidence)
5. 6 fixed azimuths no elevation (occlusal via top-down view, not elevation)
6. Input-view-related NOT canonical (no elevation inference, strong on unreal images)
7. 30K Objaverse sufficient with right inductive biases
8. 2024-08-29 bug fix is deployment lesson (CFG batch halves)
9. Channel expansion tried and rejected ("low convergence, poor generalization, catastrophic forgetting")
10. SDF chosen over mesh: "compactness and differentiability, ideal for stable optimization"

### v0 stack updated
- **sub-task 4 (outer crown surface):** += cross-domain attention + outlier-dropping loss + geometry-aware normal loss
- **H3 toolkit:** += cross-domain attention (063) | DITA (058) | OCM (044) | point-curvature (045) | PGM offset (046) | O_cp/O_ce/O_cr (059) | 2D-projection-consistency (060) | 2D-depth+gap (061) | per-point ConcatSquashLinear (062)
- **v0 compute:** unchanged (Wonder3D-derived v0 v0.5 = +$700 Lambda, deferred to v1)

### Open questions for HK
(i) pilot cross-domain diffusion as v0 side-track? (ii) port outlier-dropping loss to DiGS? (iii) port geometry-aware normal loss to cervical margin? (iv) use 6 fixed azimuths in v0 v0.5? (v) cite as v0 paper's cross-modal precedent? — **recommend YES on all 5, defer full Wonder3D training to v1**

### Next paper (064)
TBD

---

**LanceDB row:** ✅ `0d435568-509d-40fa-8354-92ce3afa694c` (memories table, row 71, category `research_paper`, importance 0.7, mxbai-embed-large)
**Digest sent:** this file (Telegram will pick up the 📄 block above)
**Errors:** none
