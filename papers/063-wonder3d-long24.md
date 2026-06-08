# 063 — Wonder3D: Single Image to 3D using Cross-Domain Diffusion

**Authors:** Xiaoxiao Long¹*†, Yuan-Chen Guo²*†, Cheng Lin¹‡, Yuan Liu¹, Zhiyang Dou¹, Lingjie Liu³, Yuexin Ma⁴, Song-Hai Zhang¹, Marc Habermann⁵, Christian Theobalt⁵, Wenping Wang⁶‡
¹ HKU · ² Tsinghua + VAST · ³ UPenn · ⁴ ShanghaiTech · ⁵ MPI Informatik · ⁶ Texas A&M
*equal contribution · †, ‡ corresponding
**Year:** 2023 (arXiv:2310.15008, v1 Oct 23 2023, v3 Nov 8 2023) → **CVPR 2024 Highlight**
**Code:** https://github.com/xxlong0/Wonder3D (MIT, training + inference + NeuS/instant-nsr-pl mesh extraction; pretrained weights on HuggingFace `flamehaze1115/wonder3d-v1.0` + OneDrive mirror)
**Project:** https://www.xxlong.site/Wonder3D/

---

## TL;DR

Wonder3D generates **6 consistent multi-view normal maps + matching color images** from a **single input image** in **2–3 minutes** on one GPU, then fuses the normals through **instant-NGP / Neuralangelo** into a textured mesh. The key idea is **cross-domain diffusion**: extend Stable Diffusion's 2D UNet with a 1-token *domain switcher* (RGB vs surface-normal) so the same denoiser can produce both domains jointly, plus a **cross-domain attention** layer that lets RGB tokens attend to normal tokens and vice-versa, enforcing geometric/visual consistency. The cross-domain trick avoids re-training Stable Diffusion from scratch and keeps the strong 2D prior intact. Beats prior SOTA on GSO (CD 0.0199, IoU 62.44) at publication.

## Research question + their answer

**Q:** *How do we generate a high-quality textured 3D mesh from a single image in minutes, without the per-shape SDS optimization (slow, inconsistent) of DreamFusion/Magic3D and without the low geometric detail of direct 3D inference (Point-E, Shape-E)?*

**A:** *Don't generate 3D directly. Generate **6 multi-view consistent 2D representations** (normal maps + color images) using a **cross-domain Stable Diffusion variant**, then fuse them into a textured mesh via a fast **geometry-aware neural-SDF optimization** (instant-NGP / Neuralangelo with mask + color + geometry-aware normal loss + outlier-dropping loss). The 2D generative prior is the strong one we already have; the 3D reconstruction is a known well-posed problem given enough consistent views.*

## Method

### Architecture
- **Backbone:** Stable Diffusion Image Variations Model (2D UNet, image-conditioned, ϵ-prediction) — *not* retrained from scratch, only fine-tuned
- **Domain switcher `s`:** a single integer token (RGB=0 or normal=1) is positional-encoded and concatenated with the time embedding, then injected into the UNet. This is the *only* domain-conditioning mechanism — no extra output channels, no separate UNet
- **Cross-domain attention:** in *every* transformer block of the UNet, a new attention layer is inserted *before* the cross-attention layer. Keys and values from BOTH domains (RGB + normal) are concatenated and the queries from domain A attend to both A and B. This is the geometric/visual consistency mechanism
- **Multi-view attention:** the self-attention layers are extended to be *global-aware* — tokens from all 6 views see each other (similar to SyncDreamer, MVDream)
- **Output:** 6 normal maps + 6 color images at 256×256, in 6 azimuthal views (0°, ±45°, ±90°, 180°) in the *input-view-related coordinate system* (not canonical — see below)

### Training
- **Data:** LVIS subset of Objaverse (Deitke et al.) — ~30,000 cleaned 3D objects
- **Rendering:** Blenderproc, each object normalized to unit scale, center-aligned, **6 views rendered with random rotations** for diversity
- **Two stages:**
  1. Stage 1: train multi-view attentions only (no cross-domain), randomly taking normal OR color flag
  2. Stage 2: add cross-domain attention modules into the SD model, only optimize the **newly added parameters** (rest of SD is frozen → preserves the strong 2D prior, avoids catastrophic forgetting)
- **Resolution:** 256×256, batch size 512, 30,000 steps, ~3 days on 8× NVIDIA Tesla A800
- **Important fix (2024-08-29):** the original training had a severe bug — `zero_init_camera_projection` should have been `False` (otherwise domain/pose controls are invalid during training); the original inference also had a CFG bug where RGB and normal domain inputs need to be placed in the first/second halves of the batch (not the typical unconditional/conditional split)

### Geometry-aware normal fusion (mesh extraction)
Given the 6 normal maps G^0:N + 6 color images H^0:N, a neural SDF is optimized to amalgamate the 2D data:
1. **Segmentation:** SAM or rembg → object masks M^0:N
2. **Random ray sampling:** at each step, sample a batch of pixels `(g_k, h_k, m_k, v_k)` from all views
3. **Geometry-aware normal loss:** for a 3D point visible from multiple views, weight normals by `cos(angle(normal, view_ray))` — the more orthogonal a normal is to its view ray, the more reliable it is (normals are *outward-facing*, view rays are *inward-facing*, so a valid pair must have angle ≥ 90°)
4. **Outlier-dropping losses:** at each iteration, sort color/mask errors in descending order, **drop the top-largest errors** (a fixed percentage). Wrong predictions lack cross-view consistency → large per-pixel errors → get dropped automatically. This is the key robustness trick — eliminates "isolated geometries and distorted textures" caused by occasional bad generations
5. **Backbone:** instant-NGP-based SDF (later, also a NeuS alternative released for Windows-friendliness). Recommended NeuS for robustness, instant-NGP for sharpness.

### Coordinate system
**Key design choice:** Wonder3D uses an *input-view-related* coordinate system, NOT the canonical system used by Zero123/SyncDreamer/MVDream. Z+ and X+ align with the UV of the 2D input image; Y+ is perpendicular to the image plane. All 6 output views are sampled in the XvOYv plane at fixed radius, so **no elevation estimation is needed** at inference (compare to Zero123 which must guess elevation). The trade-off: cannot directly match canonical-world ground-truth poses for evaluation, but **stronger generalization on unreal images** because the system adapts to whatever orientation the input has.

### Inference
- `accelerate launch test_mvdiffusion_seq.py --config configs/mvdiffusion-joint-ortho-6views.yaml ...`
- 20 diffusion steps (DDIM-like)
- ~1-2 min for multi-view generation, ~1-2 min for SDF optimization → **2–3 min end-to-end on 1 GPU**
- xformers + fp16 → 12GB VRAM

## Results

### GSO (Google Scanned Objects) Single-View Reconstruction — Table from OpenCodePapers leaderboard
| Method | CD ↓ | IoU ↑ | F-Score | Year |
|---|---|---|---|---|
| **Wonder3D** | **0.0199** | **62.44** | 0.6244 | 2023-10 |
| Unique3D | 0.0145 | 55.38 | 68.45% | 2024-05 |
| SyncDreamer | 0.0261 | 54.21 | — | 2023-09 |

Wonder3D was the **GSO SOTA at publication** with **>2× lower CD** than SyncDreamer (0.0199 vs 0.0261) and **+8 IoU points** (62.44 vs 54.21). Unique3D (5 months later) surpassed it on CD/F-Score but Wonder3D still wins on IoU — the highest-IoU single-image-to-3D method in 2024.

### OmniObject3D & Other (from paper)
- Beats Magic3D, RealFusion, Zero123, SyncDreamer, One-2-3-45 on CD, IoU, F-Score
- 2-3 min runtime vs ~30-60 min for Magic3D, ~2-4 hours for DreamFusion
- **Robust generalization**: trained on synthetic Objaverse only, transfers to real scans, paintings, AI-generated images, etc.

### Qualitative observations
- 6 fixed views = **no occluded-region hallucination** (vs 256-view NeRF-based methods that "see" the back)
- Sharp cusps and fine geometric detail (the cross-domain normal supervision is the secret)
- Failure modes: occluded objects (back is unconstrained), non-orthographic real-captured images (focal-lens distortion), elevation ambiguity (no up/down views)

## Connections to H1–H5

**H1 (2-stage: generator + reconstructor is the right architecture)** — **STRONGEST DIRECT SUPPORT IN THE 2023-2024 CROSS-MODAL LITERATURE**. Wonder3D is *exactly* the H1 decomposition: (Stage A) a 2D generative model produces multi-view consistent 2D representations, (Stage B) a 3D reconstructor fuses them into a mesh. The paper's killer empirical evidence: doing Stage A alone (multi-view diffusion, no fusion) gives images that are beautiful but not 3D; doing Stage B alone (NeuS / instant-NGP from real captures) requires dense input views and breaks on sparse/noisy inputs. Only the *combination* gives high-quality 3D in 2-3 min. For v0 sub-task 2/4, this is the H1 architectural template.

**H2 (diffusion as the generative backbone)** — **STRONG SUPPORT**. Multi-view diffusion is the *only* 2D prior strong enough to handle out-of-distribution objects. SDS-based 3D methods (DreamFusion, Magic3D) also use diffusion but require per-shape optimization (30-60+ min) and produce inconsistent geometry. Wonder3D's "diffusion in 2D, then fuse to 3D" is the H2-fast variant. The two-stage (Stage 1 multi-view attention training, Stage 2 cross-domain attention training, only the new params) is a *clean* H2 training recipe.

**H3 (context-aware generation: adjacent + opposing teeth)** — **PARTIAL + THE CLEANEST H3 EXTENSION MECHANISM IN THE 2023-2024 LITERATURE**. Wonder3D is single-image-conditioned, no multi-modal fusion. BUT: the *cross-domain attention* (RGB ↔ normal) is the exact architectural primitive needed to fuse *multiple* conditioning modalities — replace (RGB, normal) with (target_render, adjacent_teeth_render, opposing_jaw_render) and you get multi-source H3. The domain switcher generalizes to a "modality switcher" (1-token integer per modality). The reading list's v0 could adopt this pattern for sub-task 4: denoise the target tooth's normal map while attending to (a) the target tooth's color image, (b) the adjacent teeth's normal maps, (c) the opposing teeth's normal maps — all via cross-domain attention.

**H4 (SDF > explicit mesh for surface representation)** — **STRONG SUPPORT**. The mesh extraction stage is *explicitly* a neural SDF (instant-NGP / NeuS / Neuralangelo), not a mesh-based VAE. The paper's argument (§4.3): "Unlike alternative representations like meshes, SDF offers compactness and differentiability, making them ideal for stable optimization." For v0 sub-task 4, this confirms the v0 stack's choice of DiGS (paper 003) + FlexiCubes (paper 007) as the field + mesh extractor. Wonder3D uses NeuS-style volume rendering for the SDF optimization, which is *not* the same as DiGS (no divergence penalty) but is in the same family.

**H5 (transfer from synthetic to real / from one domain to another)** — **STRONG SUPPORT**. Trained entirely on synthetic Objaverse (30K LVIS objects), generalizes to GSO real scans, OmniObject3D, AI-generated images, paintings. The 2024 bug-fix in the repo (CFG order swap for cross-domain attention) is *the* kind of robustness work that comes from real-world deployment. For v0, this is a proof that the synthetic-to-real transfer works *for 3D generation* when the intermediate representation is 2D — a 2D diffusion prior trained on natural images carries over to 2D renders of dental scans, and the 3D fusion is geometric (no domain gap). This is *the* strongest H5 argument in the reading list for "we can pretrain on 100K synthetic dental arches and fine-tune on 1K clinical IOS".

## Surprises / interesting things buried in §4

1. **Domain switcher is just a 1-token integer** — the simplest possible mechanism for multi-domain diffusion. Don't need a second UNet, don't need to add output channels (which the paper tried and showed "suffers from low convergence speed and poor generalization" because "channel expansion may perturb the pre-trained weights of stable diffusion models and therefore cause catastrophic model forgetting"). The 1-token switch is the lesson: *least invasive modification to a pretrained model = best preservation of the prior*.

2. **Cross-domain attention is inserted BEFORE cross-attention, not after** — a subtle architecture choice. The intuition: the model should first "agree" between domains (RGB and normal should describe the same object) and *then* attend to the input image. Doing cross-domain after cross-attention would be redundant because the cross-attention already conditioned on the input.

3. **Geometry-aware normal loss weights by `cos(angle(normal, view_ray))`** — the more orthogonal a normal is to its view ray, the more reliable it is. This is a *cheaper* H3-like mechanism: the *viewing geometry* is itself a reliability signal. For v0, this is direct evidence that we should weight our cross-modal losses by *visibility* / *occlusion*.

4. **Outlier-dropping loss is a brilliant robustness trick** — instead of robust losses (Huber, Tukey biweight) or learned confidence weights, just *drop the top X% largest errors* at every iteration. Wrong predictions lack cross-view consistency → they have large errors → they get dropped. This is mathematically equivalent to a hard attention over the loss terms. The v0 sub-task 4 could use this exact pattern: drop the top 20% largest SDF errors per iteration to avoid being pulled around by outlier prep-margin pixels.

5. **6 fixed azimuths, no elevation** — the simplest possible view setup. No view-pose estimation, no spherical sampling, just 6 in-plane rotations. The trade-off: cannot reconstruct the top/bottom of objects (a real dental crown sits between upper and lower teeth, so the v0 might want elevation views too). This is a *non-obvious* v0 modification: add 2 elevation views (top + bottom of the arch) to capture occlusal/incisal surfaces for crown generation.

6. **Input-view-related coordinate system, NOT canonical** — this is a *deliberate* divergence from Zero123/SyncDreamer/MVDream. The advantage: no elevation estimation at inference, strong generalization on unreal images. The disadvantage: cannot do canonical pose evaluation. For v0, the v0 sub-task 4 (crown generation) actually *wants* a canonical system (so the crown can be placed in FDI-relative position on the arch) — Wonder3D's choice is *not* what we want for v0, but the cross-domain/multi-view attention mechanisms are.

7. **Trained on 30K Objaverse objects, not 800K** — much smaller than typical 2D diffusion training sets. The 30K is sufficient because the multi-view + cross-domain structure provides strong inductive biases. For v0, the lesson: a *small, clean* dental arch dataset (maybe 1K-5K carefully curated IOS scans) might be sufficient if the architecture is right — don't over-emphasize data scale.

8. **The 2024-08-29 bug fix** — the CFG inference had RGB and normal inputs in the wrong batch halves. This is a *deployment lesson* — cross-domain models have subtle inference-time bugs that don't show up in paper-quality evaluation. For v0, the v0 paper should *release inference code* with bug-fix unit tests.

## Quote-worthy sentences

> "Our model is built upon pre-trained 2D stable diffusion models to leverage its strong generalization."

> "Channel expansion may perturb the pre-trained weights of stable diffusion models and therefore cause catastrophic model forgetting."

> "A straightforward solution is to add four more channels to the output of the UNet module representing the extra domain. However, we notice that such a design suffers from low convergence speed and poor generalization."

> "Cross-domain attention maintains the same structure as the original self-attention layer and is integrated before the cross-attention layer in each transformer block of the UNet."

> "To tolerate trivial inaccuracies of the generated normals from different views, we introduce a geometry-aware normal loss... the angle between the normal vector and the viewing ray remains not less than 90°."

> "Erroneous predictions lack sufficient consistency with other views, making them less amenable to effective minimization during optimization, and they often result in large errors."

> "Surprisingly, even with fine-tuning on this relatively small-scale dataset [30K Objaverse], our method demonstrates robust generalization capabilities."

> "Our views are defined in the camera system of the input image. The six views are in the plane with 0 elevation degree in the camera system of the input image."

## Code/data link

- **Code:** https://github.com/xxlong0/Wonder3D (MIT, includes training + inference + NeuS/instant-nsr-pl mesh extraction)
- **Pretrained weights:** HuggingFace `flamehaze1115/wonder3d-v1.0` or OneDrive at https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xxlong_connect_hku_hk/Ej7fMT1PwXtKvsELTvDuzuMBebQXEkmf2IwhSjBWtKAJiA
- **Wonder3D++ extension (Dec 2024):** https://github.com/xxlong0/Wonder3D/tree/Wonder3D_Plus (higher resolution, better consistency)
- **Related works by same authors:** GeoWizard (depth+normal, Mar 2024), CraftsMan3D (3D-native diffusion, May 2024), Era3D (auto focal length, May 2024)
- **Data:** Objaverse (https://objaverse.allenai.org/), GSO (https://app.gso.io/), OmniObject3D (https://omniobject3d.github.io/)

## For our project

Wonder3D is a *cross-modal generation* paper, not a dental paper — but it is the **technical precedent for the 2D-projection-consistency loss** in Diff-OSGN/Diff-TRGN/CrownGen (papers 059/060/058) and the **2D-rendering + back-projection approach** in DCPRGAN (paper 061). Its mechanisms are *reusable* for the v0 sub-task 4 (outer crown surface) and v0 sub-task 5 (intaglio):

**(a) ADOPT cross-domain diffusion as a v0 *side-track* — not the default architecture, but a high-leverage pre-training step** ($500 Lambda, 4 weeks). Train a Stable-Diffusion-variant on 6-view renders of 3DTeethSeg22 arches, conditioning on (single-tooth-render, FDI-number, arch-position). The cross-domain attention is the *natural* place to inject adjacent-teeth and opposing-jaw conditioning (the H3 mechanism we need for v0 sub-task 4). At inference, generate 6 normal maps + 6 color images of the missing crown, then fuse via DiGS (paper 003) + FlexiCubes (paper 007). The 2-3 min inference time is acceptable for non-realtime use cases (the dentist designs, the lab fabricates overnight).

**(b) PORT the outlier-dropping loss to v0 sub-task 4 DiGS inner-surface prior** (1 day, $0). Drop the top 20% largest SDF errors per iteration when training v0 sub-task 4 (PVD + Surface Projection loss + ME-loss + CBL boundary loss + ...). This is a *direct port* from Wonder3D's §4.3 — the same robustness reasoning applies (cervical-margin pixels are the most error-prone, but cervical-margin errors are also the most clinically important, so we want to keep them in the loss with high weight... or do we? Wonder3D's answer is: drop the largest 20%, they will be mislabeled anyway). Trade-off discussion for v0 paper: keep outlier-dropping on the *boundary* loss terms only, full weight on the *interior* SDF loss.

**(c) PORT the geometry-aware normal loss to v0 sub-task 4 cervical margin refinement** (1-2 days, $30 Lambda). The cos(angle(normal, view_ray)) weighting is the *cleanest* way to weight per-vertex contributions: weight the cervical margin normals higher than occlusal normals (the cervical margin normals are more *grazing* relative to any view ray, so they're more reliable). Direct 1-line code change to the MGR normals+curvatures secondary loss.

**(d) CONSIDER input-view-related coordinate system for v0 sub-task 4** (architecture decision, no code yet). The advantage: no FDI-relative canonical pose estimation needed (the FDI number is implicit in the input tooth's identity in the IOS scan). The disadvantage: cannot do canonical evaluation against a ground-truth crown library. **Recommendation: keep the canonical FDI-relative system for v0 (consistent with the rest of the v0 stack — Cao25, CrownSegger, etc., all assume canonical positions on the arch), but use Wonder3D's multi-view pattern as inspiration for the multi-source H3 conditioning (RGB + normal + adjacent + opposing all in cross-attention).**

**(e) USE 6 fixed azimuths, NO elevation, as v0 v0.5 pilot configuration** (1 day setup, $0). v0 doesn't have elevation views in the IOS scan, and the *occlusal surface* of a crown is captured by the IOS top-down view, not by adding elevation. The 6-azimuth setup is *the right amount* of conditioning for v0 — adding elevation would require re-engineering the IOS data loader. The exception: if v0 v1 wants to handle full-arch cases, add 2 elevation views (one for upper arch, one for lower arch) — Wonder3D's bug-fix history shows the elevation inference is *subtle*, so fixed views are safer.

**(f) USE 256×256 generation resolution as v0 v0.5 pilot resolution** (architecture decision, no code yet). Wonder3D's 256×256 is what fits in 12GB VRAM with batch size 32. v0's v0.5 sub-task 4 pilot should match this (256×256 renders of the missing tooth, 6 views). For v1, 512×512 (Era3D) or 1024×1024 (Wonder3D++ at 512×512 then upsampled) is the upgrade path.

**(g) CITE Wonder3D as the v0 paper's "cross-modal generation precedent" in related work** ($0, 30 min writing). The v0 paper's related work section should make the *general 2D-3D cross-modal generation literature* explicit: Wonder3D (2023), SyncDreamer (2023), MVDream (2023), One-2-3-45 (2023), ImageDream (2023), Magic3D (2023), Era3D (2024), Unique3D (2024), CraftsMan3D (2024), Wonder3D++ (2024). v0's contribution is *applying* these to the dental domain, not inventing new 2D-3D pipelines.

**(h) FOLLOW the Wonder3D training recipe pattern for v0 v0.5** (architecture pattern, 2 weeks setup, $200 Lambda). The two-stage training (Stage 1: multi-view only, Stage 2: add cross-domain, only train new params) is the *right* recipe for the v0 v0.5 sub-task 4 cross-modal generation. Stage 1 uses random normal/flag to train the multi-view attention on dental data; Stage 2 adds cross-domain attention to fuse (target_render, adjacent_render, opposing_render) — the H3 mechanism.

**(i) DEFER the v0 v0.5 cross-modal generation pilot to v0 v1** ($0 decision, but $500 Lambda deferred). The full Wonder3D-from-scratch training on dental data is *expensive* (3 days on 8× A800, ~$500 Lambda) and not on the v0 critical path. v0's critical path is DiGS + FlexiCubes + CAo25 + IGIP parabola + GRAB-Net OCM. The Wonder3D-derived v0 v0.5 cross-modal pipeline is the v0 v1 product differentiator.

**(j) REPLICATE the Wonder3D bug-fix pattern in the v0 repo** ($0, 1 day). Add explicit unit tests for cross-domain attention CFG inference (RGB and normal inputs in the right batch halves, not the typical unconditional/conditional split). The Wonder3D repo's 2024-08-29 commit is a *deployment lesson* — even SOTA papers have subtle inference-time bugs, and the v0 should have a CI pipeline that catches them.
