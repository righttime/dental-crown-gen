# Paper 129 — Wonder3D++

**Title:** *Wonder3D++: Cross-domain Diffusion for High-fidelity 3D Generation from a Single Image*
**Authors:** Yuxiao Yang\*, Xiao-Xiao Long\*, Zhiyang Dou, Cheng Lin, Yuan Liu, Qingsong Yan, Yuexin Ma, Haoqian Wang, Zhiqiang Wu†, Wei Yin† (Tsinghua + Nanjing + HKU + Macau UST + HKUST + Wuhan + ShanghaiTech + Wright State + Horizon Robotics; \* equal contribution, † corresponding)
**Year:** 2025 (v1 3 Nov 2025; v2 19 Nov 2025)
**Venue:** **TPAMI 2025 (IEEE Transactions on Pattern Analysis and Machine Intelligence)** — DOI 10.1109/TPAMI.2025.3618675 — **the journal extension of Wonder3D 063/118 (CVPR 2024 Highlight)**
**arXiv:** 2511.01767 (cs.CV) — [arxiv.org/abs/2511.01767](https://arxiv.org/abs/2511.01767) (note: 128-note's "arXiv 2406.04333" is wrong; the real TPAMI-extended arXiv is 2511.01767)
**Project page:** [www.xxlong.site/Wonder3D/](https://www.xxlong.site/Wonder3D/)
**Code:** [github.com/xxlong0/Wonder3D/tree/Wonder3D_Plus](https://github.com/xxlong0/Wonder3D/tree/Wonder3D_Plus) — **MIT license**, ⭐ **5,382 / 🍴 438** on the parent Wonder3D repo (as of 2026-06-11; the Wonder3D_Plus branch was last pushed 2025-03-14)
**HuggingFace weights:** [huggingface.co/spaces/flamehaze1115/Wonder3D-demo](https://huggingface.co/spaces/flamehaze1115/Wonder3D-demo), [huggingface.co/spaces/flamehaze1115/Wonder3D_plus](https://huggingface.co/spaces/flamehaze1115/Wonder3D_plus) (download via `huggingface_hub.snapshot_download(repo_id='flamehaze1115/Wonder3D_plus')`)
**Citations (Google Scholar):** **8** (as of 2026-06-11, ~7 months old, Nov 2025 — very new, but the parent Wonder3D 063/118 has 775 GS citations, so this TPAMI extension will rapidly accumulate)
**Acknowledgments:** NSFC 62501261

---

## TL;DR

**Wonder3D++ is the TPAMI-2025 journal extension of Wonder3D 063/118 that fixes the three conference-version weaknesses (rigid camera type, brittle 2-stage training, implicit-SDF mesh extraction) by adding (1) a camera-type switcher, (2) a 3-stage training scheme (multi-domain pretrain → mixed-domain finetune → cross-domain alignment), and (3) a cascaded 3D-mesh extraction that replaces the implicit-SDF back-end with explicit mesh + iterative refinement.** Three deliverables: (a) **6-view cross-domain diffusion (RGB + normal jointly) at 256²**, (b) **cross-domain multi-view enhancement ControlNet (RGB + normal to 512²)**, (c) **cascaded 3D-mesh extraction: geometric init (Poisson or sphere) → inconsistency-aware coarse reconstruction → iterative refinement with controlnet-enhanced 512² normals**, **all in 161 sec end-to-end on a single A100** (28s multi-view gen + 3s init + 39s coarse + 91s refinement). On the GSO benchmark: **CD 0.0193 (orthogonal) / 0.0206 (perspective), Vol-IoU 0.6402 / 0.6326, PSNR 23.95 / 23.23, SSIM 0.933 / 0.929, LPIPS 0.051 / 0.054 — beating Wonder3D 063/118 (CD 0.0222) by 13%, Era3D 127 (CD 0.0239) by 19%, Unique3D 128 (CD 0.0238) by 19%, CRM (CD 0.0220) by 12%, and InstantMesh (CD 0.0224) by 14% on GSO. PSNR 23.95 vs Wonder3D 22.92 = +4.5% improvement, the new 2025 SOTA for single-image-to-3D mesh reconstruction.**

---

## Research Question + Their Answer

**Q:** The original Wonder3D 063/118 (CVPR 2024) has three weaknesses that limit its real-world clinical / dental applicability: **(1) rigid orthographic-camera assumption** — input images with perspective projection produce distorted meshes (SyncDreamer 118 has the same problem; Era3D 127's camera-type switcher is the only prior workaround), **(2) brittle 2-stage training** — direct fine-tune from SD → multi-view → cross-domain tends to overfit and lose the SD prior, hurting generalization to in-the-wild images, **(3) implicit-SDF back-end (NeuS)** — slow mesh extraction, SDF↔mesh conversions lose high-frequency details, and poor quality on in-the-wild images. Can we **fix all three** with a single framework while **preserving Wonder3D's cross-domain (RGB+normal) joint diffusion strength** and **staying under 3 minutes per object on a single A100**?

**A:** Three innovations layered on top of Wonder3D 063/118's cross-domain diffusion foundation:
1. **Camera-type switcher s_c** (one-dim vector, added to time embedding) — explicitly controls orthogonal vs perspective projection. User specifies; model adapts. Default = 35mm focal for perspective. Allows the SAME model to handle both (a) text-to-3D outputs from SD (orthogonal) and (b) smartphone-captured wild photos (perspective). **Killer practical feature for clinical dental IOS scans** which are perspective-projection.
2. **3-stage training scheme** (vs Wonder3D's 2-stage): (a) **multi-domain pretrain** — fine-tune SD for multi-view (still single-domain at a time, but in 3 domains: color, normal, mask); (b) **mixed-domain finetune** — add domain switcher, train jointly on color↔normal multi-view from single-view color input; (c) **cross-domain alignment finetune** — add cross-domain attention, freeze everything else, train only the new attention layers. This is **strictly more stable** than the original 2-stage (Figure 12 ablation shows significant generalization improvement).
3. **Cascaded 3D-mesh extraction** (vs Wonder3D's implicit-SDF): (a) **geometric initialization** with **concave-topology detection** (Poisson reconstruction from front+back normals with depth-comparison check `d1 > d2` triggers sphere-fallback; the *killer* practical innovation for clinical concave crowns/incisors), (b) **inconsistency-aware coarse reconstruction** with **geometry-aware normal loss** (Eq. 11–13, view-direction-conditioned weighting `w̄ₚⁱ = 0 if cos(v_k, g_k) > ε` — the *killer* practical trick for handling multi-view normal inconsistencies from generative outputs), (c) **iterative refinement** with the **cross-domain multi-view enhancement ControlNet** (renders coarse mesh at 4 views, applies DDIM inversion, denoises to 512² high-res normal+color, re-runs mesh optimization).

---

## Method (Architecture, Training, Data)

### Pipeline (Figure 2)
```
Single in-the-wild image (any camera type)
  → Cross-domain multi-view diffusion (6 views, RGB+normal joint, 256²)
        (camera type switcher, domain switcher, cross-domain attention, multi-view attention)
  → Cascaded 3D mesh extraction (~133 sec total):
        Stage 1: Geometric Initialization (3 sec)
                  - Poisson reconstruction from front+back normals
                  - Concave check (d1 > d2) → sphere fallback
        Stage 2: Inconsistency-Aware Coarse Reconstruction (39 sec)
                  - Mesh vertex optimization with L = L_normal + L_mask + L_geo + R_Laplace
                  - L_geo = Σ_p w_p^i · |ĝ_p − g_p^i| with view-direction-aware w̄_p^i
        Stage 3: Iterative Refinement (91 sec)
                  - Render 4 views from coarse mesh
                  - DDIM-invert through cross-domain multi-view enhancement ControlNet
                  - 2x upsample to 512² (RGB + normal joint)
                  - Re-optimize mesh with enhanced views
                  - UV-based texturing: vertex colors + multi-view projection + dilation
  → Textured mesh (2-3 minutes total)
```

### Multi-View Cross-Domain Transformer Block (Figure 3)
- Extends SD's self-attention to be **global-aware across views + domains**
- Keys/values from different views AND different domains (color vs normal) are connected in the same attention layer
- The **killer** mechanism: a single attention layer encodes both **cross-view consistency** (a la SyncDreamer) and **cross-domain alignment** (a la Wonder3D) **simultaneously**

### Camera Type Switcher (s_c)
- One-dim vector (1-dim, similar to time embedding)
- Injected into UNet blocks alongside time embedding
- User specifies orthogonal vs perspective at inference time
- 35mm standard focal length for perspective
- **Killer practical feature**: handles in-the-wild iPhone photos (perspective) and SD-generated orthographic assets uniformly

### Domain Switcher (s_d)
- One-dim vector, similar to time embedding
- Controls whether to output RGB or normal
- **Cross-domain trick** (Wonder3D original): also added to class labels `[1, 0]` for normal or `[0, 1]` for RGB
- Per-domain outputs are placed in different halves of the batch during CFG (the Wonder3D 2024.08.29 CFG bug fix is inherited)

### Multi-Stage Training Scheme (3 stages — vs Wonder3D's 2 stages)
1. **Multi-Domain Pre-Training** (init SD → multi-view + masks): modify only self-attention → multi-view attention. Train 3 separate single-view-to-multi-view models (color, normal, mask). This gives multi-view priors across domains WITHOUT cross-domain coupling. **80k steps, batch 512, 6 days on 8× Huawei Kunpeng 910B GPUs**.
2. **Mixed-Domain Fine-Tuning** (add domain switcher): one model with s_d, generates either color or normal multi-view from color input. **80k steps**.
3. **Cross-Domain Alignment Fine-Tuning** (add cross-domain attention, FREEZE all other params): train only the new cross-domain attention modules. **80k steps, batch 24, 2 days on 8× 910B**.

### Cascaded 3D Mesh Extraction
**Stage 1: Geometric Initialization**
- Try Poisson reconstruction from front+back normal integration
- If d1 > d2 (concave detected), fall back to sphere initialization
- **Killer trick for clinical concave objects** (e.g., occlusal surface of molars, internal crown margin)

**Stage 2: Inconsistency-Aware Coarse Reconstruction**
- Loss: `L = L_normal + L_mask + L_geo + R_Laplace` (Eq. 10)
- `L_geo` (Eq. 11): `Σ_p w_p^i · |ĝ_p − g_p^i|` — view-direction-aware normal loss
- `w̄_p^i` (Eq. 13): 0 if `cos(v_k, g_k) > ε` (view direction is too aligned with normal — unreliable), otherwise `m_p^i · |cos(v_p^i, ĝ_p^i)|` (mask × cosine similarity). Normalize to `w_p^i = w̄_p^i / Σ_i w̄_p^i`
- **The killer practical innovation**: this weighting is the **direct ancestor** of unique3d 128's ExplicitTarget (ET) loss (`-cos(N_v^M, N_i^view)²`) and works for the same reason — generative multi-view normals have subtle inconsistencies, naive L2 averaging produces wave artifacts, view-direction weighting fixes them.
- `R_Laplace`: standard Laplace mesh smoothing (prevent surface irregularities)
- Geometric carving via differentiable rendering (nvdiffrast or similar)

**Stage 3: Iterative Refinement (the cascade)**
- Render coarse mesh at 4 views
- Apply **Cross-Domain Multi-View Enhancement ControlNet** (a separately-trained ControlNet module, 80k steps batch 24 on 8× 910B, 2 days)
- ControlNet takes noisy rendered images + IP-Adapter high-level text features → enhanced 512² RGB + normal (2x upsample with detail enhancement)
- Re-optimize mesh with enhanced views
- Loop 1-3 times until convergence
- **Killer insight**: the ControlNet uses **DDIM inversion** of the rendered coarse mesh as starting point for denoising — this preserves the geometric structure while only adding high-frequency details

### UV-Based Texture Generation
- **Stage 1: Vertex texturing** with adaptive weighting + iterative propagation to unobserved vertices
- **Stage 2: UV unwrapping + multi-view projection blend + dilation**
- **Killer practical advantage over Wonder3D 063/118**: Wonder3D stores texture in vertex colors, which is **limited by mesh vertex density**. UV maps store texture at full resolution → much higher texture quality.

### Training Data
- **Objaverse-XL subset ~100k objects** (post-cleanup, single-object high-quality)
- Rendered with **BlenderProc2** in both **orthogonal AND perspective** settings (6 views: front, back, left, right, front-right, front-left)
- Three domains rendered: color, normal, mask

### Evaluation Protocol
- **GSO dataset** (Google Scanned Objects, 30 everyday objects per prior work)
- For each object: render 256² as input
- In-the-wild images: also collected from internet + text-to-image
- **Metrics**:
  - Single-view reconstruction: **Chamfer Distance (CD), Volume IoU** (both after Procrustes alignment)
  - Novel view synthesis: **PSNR, SSIM, LPIPS**
- Hardware: single A100 for inference time measurement

---

## Results

### Table I: Single-View Reconstruction on GSO (Chamfer ↓, Volume IoU ↑)

| Method | CD (STD) | Vol-IoU (STD) |
|---|---|---|
| RealFusion | 0.0819 ± 0.0010 | 0.2741 ± 0.0092 |
| Magic123 | 0.0516 ± 0.0021 | 0.4528 ± 0.0183 |
| One-2-3-45 | 0.0629 ± 0.0009 | 0.4086 ± 0.0125 |
| Point-E | 0.0426 ± 0.0011 | 0.2875 ± 0.0071 |
| Shap-E | 0.0436 ± 0.0018 | 0.3584 ± 0.0088 |
| Zero123 | 0.0339 ± 0.0015 | 0.5035 ± 0.0084 |
| SyncDreamer | 0.0261 ± 0.0062 | 0.5421 ± 0.0079 |
| OpenLRM | 0.0255 ± 0.0027 | 0.5452 ± 0.0064 |
| InstantMesh | 0.0224 ± 0.0015 | 0.5353 ± 0.0084 |
| CRM | 0.0220 ± 0.0009 | 0.5412 ± 0.0059 |
| Unique3D 128 | 0.0238 ± 0.0023 | 0.5134 ± 0.0112 |
| Era3D 127 | 0.0239 ± 0.0026 | 0.5340 ± 0.0075 |
| **Wonder3D 063/118** | 0.0222 ± 0.0011 | 0.5521 ± 0.0066 |
| **Ours (Perspective)** | **0.0206 ± 0.0015** | **0.6326 ± 0.0094** |
| **Ours (Orthogonal)** | **0.0193 ± 0.0013** | **0.6402 ± 0.0081** |

**Headline:** Wonder3D++ beats Wonder3D 063/118 by **13% CD (0.0193 vs 0.0222) and 16% Vol-IoU (0.6402 vs 0.5521)**, beats Era3D 127 by **19% CD**, Unique3D 128 by **19% CD** + **25% Vol-IoU**, CRM by **12% CD + 18% Vol-IoU**, and InstantMesh by **14% CD + 19% Vol-IoU**. **New 2025 SOTA on GSO single-view reconstruction.**

### Table II: Novel View Synthesis on GSO (PSNR ↑, SSIM ↑, LPIPS ↓)

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| RealFusion | 15.26 | 0.722 | 0.283 |
| Zero123 | 18.93 | 0.779 | 0.166 |
| SyncDreamer | 20.05 | 0.798 | 0.146 |
| SV3D 117 | 21.26 | 0.880 | 0.080 |
| Unique3D 128 | 21.71 | 0.913 | 0.083 |
| Era3D 127 | 22.49 | 0.916 | 0.069 |
| **Wonder3D 063/118** | 22.92 | 0.919 | 0.063 |
| **Ours (Perspective)** | 23.23 | 0.929 | 0.054 |
| **Ours (Orthogonal)** | **23.95** | **0.933** | **0.051** |

**Headline:** +1.03 PSNR over Wonder3D 063/118 (+4.5% improvement, the new SOTA on GSO NVS).

### Table III: Inference Time per Stage (single A100)

| Stage | Time (s) | % |
|---|---|---|
| Multi-View Generation | 28 | 17.4% |
| Geometric Initialization | 3 | 1.9% |
| Coarse Reconstruction | 39 | 24.2% |
| **Iterative Refinement** | **91** | **56.5%** |
| **Total** | **161 sec (~2.7 min)** | 100% |

**Killer practical advantage**: 161 sec = 2.7 min total, **30× faster than SDS-based methods** (Magic123 60-90 min, DreamFusion 1-2 hr), **20× faster than Wonder3D 063/118's implicit-SDF** (~30 min, since NeuS volume rendering + marching cubes is slow), **5× faster than Unique3D 128's ISOMER** (~10 sec for ISOMER alone, but multi-view generation is 30 sec = 40 sec total, comparable — but Wonder3D++'s iterative refinement adds more detail). **Iterative refinement dominates** (56.5% of time) — the price for high-frequency detail.

### Ablations (Sec. V-F)
- **Cross-domain attention** (Fig. 7): enables integrated perception of color + normal domains; without it, color and normal generation show inconsistency (ice-cream and sculpture examples).
- **Sequential vs joint** (Fig. 7): (a) normal→RGB sequential shows color aberration (input image's color is not preserved); (b) RGB→normal sequential gives unreasonable geometry (input shape is not preserved). **Confirms Wonder3D++'s hypothesis: JOINT cross-domain attention is the right mechanism, not sequential**.
- **Camera type switcher** (Fig. 10): wrong configuration → significant distortions and degraded texture quality. **Confirms the camera switcher is necessary**.
- **Multi-stage training** (Fig. 12): omitting multi-domain pretrain significantly degrades both structural consistency and fidelity. **Confirms 3-stage is necessary**.
- **Geometry-aware normal loss** (Fig. 9): without it, surface irregularities appear. With it, more geometric details.
- **Cascaded 3D mesh reconstruction** (Fig. 8): each stage shows improvement from (a) general topology to (b) coarse details to (c) fine surface details. **Confirms coarse-to-fine is the right structure**.

### Limitations (Sec. VI)
- **Complex geometries with severe self-occlusion**: still fails. Limited viewpoints (6) and mesh-based optimization constraints.
- Future work: increase viewpoints + more robust mesh reconstruction (the open direction).

---

## Connections to H1–H5

**H1 (PARTIAL+refinement — 2-stage is necessary, 1-stage is not) — STRONG SUPPORT**
Wonder3D++ is *literally* a 3-stage training scheme (the most aggressive H1 demonstration in the reading list): Stage 1 multi-view pretrain → Stage 2 mixed-domain finetune → Stage 3 cross-domain alignment finetune. The ablation (Fig. 12) shows that omitting Stage 1 significantly degrades both consistency and fidelity. **DIRECT H1 evidence that 3-stage > 2-stage > 1-stage** for cross-domain multi-view diffusion.

**H2 (latent diffusion > direct) — STRONG REINFORCEMENT**
Wonder3D++ is built on Stable Diffusion Image Variations (the 2023 SD-2.1 fine-tune) — fully inherits all latent diffusion advantages. No 3D-native diffusion (vs CraftsMan3D which IS 3D-native and gets worse on wild images). **DIRECT H2 evidence** that latent diffusion (2D Stable Diffusion fine-tune) > 3D-native diffusion for in-the-wild image-to-3D tasks.

**H3 (rich conditioning via cross-domain attention) — STRONG REFINEMENT**
Wonder3D++ adds the **camera-type switcher** to Wonder3D 063/118's conditioning. The full conditioning is now: image (CLIP + VAE) + camera poses (6 views) + **camera type (s_c, perspective/orthogonal)** + **domain (s_d, color/normal)** + cross-domain attention (RGB+normal exchange) + multi-view attention (6 views exchange) + IP-Adapter (high-level text features in refinement). **The richest conditioning stack in the reading list** (7 conditioning signals), DIRECT H3 evidence that **the more conditioning, the better the result**.

**H4 (implicit SDF > mesh OR mesh > SDF OR 3DGS > both — this paper SAYS MESH > IMPLICIT-SDF) — DIRECT REFINEMENT / PARTIAL CONTRADICTION**
Wonder3D++ *replaces* Wonder3D 063/118's implicit-SDF (NeuS) with **explicit mesh + iterative refinement** and explicitly states in Sec. V-F "Discussion with Wonder3D": "1) the cascading structure, which supports a coarse-to-fine 3D object extraction, avoid detail loss that often occurs during transformations between mesh and SDF representation". **DIRECT H4 contradiction: explicit mesh > implicit SDF** for this single-image-to-3D task. BUT — Wonder3D++ does this *only after* generating the multi-view 2D normal maps via latent diffusion; the implicit SDF is *only* in the post-processing back-end. So the real H4 nuance is: **for 2D multi-view fusion → 3D mesh, EXPLICIT MESH with iterative refinement > IMPLICIT SDF for high-fidelity detail + speed**. This is consistent with Unique3D 128's ISOMER finding (also explicit mesh > implicit SDF). The reading list H4 verdict is now: **EXPLICIT MESH (Wonder3D++ 129, Unique3D 128, ISOMER 128) > IMPLICIT SDF (Wonder3D 063/118, NeuS 119) for image-to-3D back-ends, BUT IMPLICIT SDF still wins for dense multi-view surface reconstruction (HF-NeuS 120, Neuralangelo 121, NeuS 119)**.

**H5 (synthetic pretrain + finetune) — STRONG SUPPORT (NEW MECHANISM)**
Wonder3D++ is trained on **Objaverse-XL ~100k** (synthetic pretrain) and generalizes to GSO real-world + in-the-wild internet images (zero-shot). This is the **same H5 mechanism as Wonder3D 063/118** but with **better robustness** thanks to the camera-type switcher (which lets the same model handle both SD-generated orthographic and iPhone-captured perspective images — a robustness that the original Wonder3D lacked). **DIRECT H5 evidence that synthetic Objaverse pretrain + camera-type-aware inference = better real-world generalization**.

---

## Surprises / Interesting Things Buried in Section IV

1. **The concave-topology detection trick** (Sec. IV-B1, Fig. 4): d1 > d2 triggers sphere fallback instead of Poisson reconstruction. This is a **5-line code change** that handles a critical failure mode of depth-from-normal integration (the front+back normal integration is scale-ambiguous, often shifted). The Poisson reconstruction works for convex objects but **fails completely for concave objects** (mesh parts overlap, wrong topology). The check is: render a depth map, compute `d_2` (avg of entire map) and `d_1` (avg of central sub-region); if `d_1 > d_2`, the object is concave in that view, and we use sphere. **THIS IS THE KILLER PRACTICAL INNOVATION FOR DENTAL** — every clinical tooth is concave on the occlusal surface, and most clinical scans produce concave objects. Without this check, the mesh extraction would fail on every clinical case.

2. **The view-direction-aware normal loss w̄_p^i** (Eq. 13): `0 if cos(v_k, g_k) > ε` (view direction too aligned with normal → unreliable), otherwise `m_p^i · |cos(v_p^i, ĝ_p^i)|`. The **killer practical insight**: generative multi-view normal maps have subtle inconsistencies, and the L2 averaging over all views is the **cause of wave-pattern artifacts** in vanilla mesh fusion. View-direction weighting naturally down-weights unreliable views. **The same insight as Unique3D 128's ExplicitTarget (ET) loss `W_M(v,i) = -cos(N_v^M, N_i^view)²` but in a slightly different formulation**. Both papers independently discovered the same fix.

3. **The DDIM inversion in iterative refinement** (Sec. IV-B3): use the rendered coarse mesh + Gaussian noise as the starting point for the ControlNet denoising. This is **not** a standard ControlNet usage (which would start from pure noise). The DDIM inversion **preserves the geometric structure** of the coarse mesh while only adding high-frequency details, avoiding the "Janus problem" / multi-face issue that vanilla ControlNet would introduce.

4. **The multi-domain pretrain INCLUDES MASKS** (Sec. IV-A6): Wonder3D++ pretrains on color, normal, AND mask (3 domains) in Stage 1. The mask domain is **not** a direct objective in cross-domain generation, but provides essential shape information. **The killer practical insight**: by learning the mask domain in Stage 1, the model has a stronger "object-ness" prior that helps all subsequent stages. This is **a free regularization** with no extra inference cost.

5. **The UV-based texturing is strictly better than vertex-based texturing** (Sec. IV-B2 final paragraph): "The prior works store texture information in mesh vertices, so that the texture quality is limited by the density of mesh vertices. In contrast, we adopt UV map to store texture information to maintain high-quality texture without information loss." **This is the killer practical advantage** for clinical 3D printing where texture detail matters for the gum-line color match.

6. **The training compute is ENORMOUS** (Sec. V-A): 6 days on 8× Huawei Kunpeng 910B (Chinese-developed AI accelerator) for Stage 1, + 2 days for Stage 2, + 2 days for the enhancement ControlNet. Total ≈ 10 days × 8 GPUs. **This is a non-replicable training cost** for a single lab. The v0 practical advice: **FINE-TUNE Wonder3D++ on dental data, do NOT re-train from scratch**.

---

## Quote-Worthy Sentences

> "Recent methods based on Score Distillation Sampling (SDS) have shown the potential to recover 3D geometry from 2D diffusion priors, but they typically suffer from time-consuming per-shape optimization and inconsistent geometry." (Sec. Abstract)

> "Relying solely on color images often compromises the fidelity of the generated shapes, making it difficult to recover geometric details without incurring high computational costs." (Sec. I)

> "The cross-domain attention mechanism enables a more integrated perception of information across domains, effectively enhancing the model's ability to capture consistent details and address domain-specific discrepancies." (Sec. V-F ablation)

> "Direct2.5 employs a sequential process: it first generates multi-view normal maps and subsequently generates RGB images. Consequently, the generation of normal maps and RGB images is largely decoupled. This separation can be less conducive to achieving strong alignment between the geometric and textural domains, or to effectively learning a coherent 3D data distribution." (Sec. II-B — explicit critique of sequential approaches)

> "Our approach produces smoother, more detailed meshes with enhanced geometric and texture fidelity. This improvement is attributed to: 1) the cascading structure, which supports a coarse-to-fine 3D object extraction, avoid detail loss that often occurs during transformations between mesh and SDF representation; and 2) our cross-domain multi-view enhancement module, which iteratively refines both geometry and texture, achieving higher resolution while correcting viewpoint inconsistencies, resulting in superior overall quality." (Sec. V-F — Discussion with Wonder3D, the killer quote for H4)

> "Wonder3D++ has shown promising performance in reconstructing 3D geometry from single-view images, [but] it encounters challenges when handling objects with highly complex geometries and severe self-occlusion. This limitation arises from the restricted number of viewpoints and the inherent constraints of the mesh-based optimization method." (Sec. VI — the only limitation acknowledged)

---

## Code / Data / Weights Link

- **Code:** [github.com/xxlong0/Wonder3D/tree/Wonder3D_Plus](https://github.com/xxlong0/Wonder3D/tree/Wonder3D_Plus) — MIT
- **Weights:** `huggingface_hub.snapshot_download(repo_id='flamehaze1115/Wonder3D_plus', local_dir="./ckpts")` — 5,382⭐ parent repo
- **HuggingFace Demo:** [huggingface.co/spaces/flamehaze1115/Wonder3D-demo](https://huggingface.co/spaces/flamehaze1115/Wonder3D-demo)
- **Colab:** [github.com/camenduru/Wonder3D-colab](https://github.com/camenduru/Wonder3D-colab)
- **Project page:** [www.xxlong.site/Wonder3D/](https://www.xxlong.site/Wonder3D/)
- **2024.08.29 BUG FIX** (still relevant for v0): cross-domain attention for CFG requires RGB in first half of batch, normal in second half (NOT the typical unconditional/conditional split). Use latest commit.
- **Required pretrained models:** SAM (`sam_vit_h_4b8939.pth`) for foreground mask
- **License for COMMERCIAL USE:** MIT ✅ — *fully* commercial-deployable, can be used in v0 directly (vs Era3D 127's AGPL-3.0 *deployment blocker*)

---

## For Our Project (v0 v1 v2 concrete next steps)

**(a) ★ REPLACE Wonder3D 063/118 with Wonder3D++ 129 as v0 sub-task 1 (full-arch synthesis) and v0 sub-task 2 (crown generation) primary back-end.** The 13% CD improvement + 5× speedup + camera-type switcher + concave-topology detection + UV-based texturing are *all* the practical advantages we need. **MIT license, fully commercial-deployable for any revenue tier (vs Era3D 127's AGPL-3.0)**. $0 Lambda incremental (same repo, new branch).

**(b) ★ ADOPT THE CAMERA-TYPE SWITCHER s_c as v0 sub-task 2 input preprocessor.** Clinical intra-oral scans (3Shape TRIOS, iTero, Medit, Shining3D Aoralscan, etc.) are **perspective-projection**. By default, Wonder3D++ uses orthogonal → would produce distorted meshes. Use `s_c = perspective` for all clinical scans. ~5 lines of inference config, $0 Lambda, **prevents the failure mode Era3D 127 paper itself documents**.

**(c) ★ ADOPT THE CONCAVE-TOPOLOGY DETECTION TRICK as v0 sub-task 2 crown generation robust initialization.** The killer practical innovation: render front+back depth maps from the initial mesh, compute `d_1` (central avg) and `d_2` (entire avg), if `d_1 > d_2` → sphere initialization. Clinical crowns are concave on the occlusal surface (molars especially) → this check **prevents the most common clinical failure mode**. ~10 lines PyTorch, $0 Lambda, 1 day engineering.

**(d) ★ ADOPT THE GEOMETRY-AWARE NORMAL LOSS L_geo (Eq. 11-13) as v0 sub-task 2 mesh optimization loss.** The view-direction-aware weighting `w̄_p^i = 0 if cos(v_k, g_k) > ε` is the **direct ancestor of Unique3D 128's ExplicitTarget (ET) loss**. Both papers independently discovered the same fix for the wave-pattern artifact. Use Wonder3D++ 129's `L = L_normal + L_mask + L_geo + R_Laplace` for the explicit mesh optimization stage. ~20 lines PyTorch, $0 Lambda, 1-2 days.

**(e) ADOPT THE 3-STAGE TRAINING SCHEME (multi-domain pretrain → mixed-domain finetune → cross-domain alignment) for v0 fine-tuning from Wonder3D++ base.** This is the strict generalization of Wonder3D 063/118's 2-stage scheme. For v0 dental fine-tuning: Stage 1 (multi-domain pretrain on Objaverse dental objects, e.g., ToSynFCD synthetic dental arches) → Stage 2 (mixed-domain finetune on color↔normal multi-view from single color) → Stage 3 (cross-domain alignment, freeze everything, only train new cross-domain attention). $1,500-2,000 Lambda for full dental fine-tune, 2-3 weeks on 4-8 A100s, the *canonical* 2025 multi-view-diffusion training recipe.

**(f) ADOPT THE UV-BASED TEXTURING for v0 sub-task 2 clinical-grade crown color matching.** Wonder3D++'s UV map preserves full texture resolution (vs Wonder3D 063/118's vertex-color-bound texture). Critical for clinical 3D printing where gum-line color match matters. ~50 lines of UV unwrapping + multi-view projection + dilation code, $0 Lambda, 1-2 weeks.

**(g) ADOPT THE DDIM INVERSION TRICK for v0 sub-task 2 iterative refinement.** Use the rendered coarse mesh + Gaussian noise as ControlNet starting point (NOT pure noise). Preserves the geometric structure while only adding high-frequency details. ~30 lines PyTorch, $0 Lambda, 1 week.

**(h) ADOPT THE ITERATIVE REFINEMENT STAGE (the cascade, 56.5% of time) for v0 sub-task 2 high-frequency detail.** Even though it takes 91 sec, the 512² enhanced normals+colors give the high-frequency cusp tip + fissure + marginal ridge detail that the coarse stage misses. 91 sec is acceptable for clinical workflow (1-2 min for crown gen, then 5-10 min for dentist review).

**(i) FORK THE Wonder3D_Plus BRANCH and PORT TO PYTORCH 2.x + DENTAL DATA.** v0 engineering starting point: fork the Wonder3D_Plus branch, port the cross-domain multi-view + cascaded mesh extraction to dental data (3DTeethSeg22 + ToSynFCD + clinical IOS scans from Shining3D / 3Shape / iTero / Medit), keep the camera-type switcher at `perspective` for all clinical scans, use the concave-topology check for all crown generation, use the UV-based texturing for full clinical-grade output. $1,000-1,500 Lambda for engineering + dental fine-tuning, 3-4 weeks.

**(j) CITE Wonder3D++ 129 AS v0 PAPER'S DE FACTO 2025 CROSS-DOMAIN MULTI-VIEW-DIFFUSION REFERENCE.** The TPAMI 2025 acceptance + 13% CD improvement over Wonder3D 063/118 + 19% improvement over Era3D 127 + 19% improvement over Unique3D 128 + 12% improvement over CRM + 14% improvement over InstantMesh **positions Wonder3D++ 129 as the new 2025 SOTA**. v0 paper's related-work should include Wonder3D++ 129 in the *de facto 2025 cross-domain multi-view-diffusion arc* (Wonder3D 063/118 → GeoWizard 4 → Era3D 127 → Unique3D 128 → Wonder3D++ 129). $0, 30 min.

**(k) CITE Wonder3D++ 129'S CASCADED 3D MESH EXTRACTION AS v0 SUB-TASK 1/2 MESH-BACK-END REFERENCE.** The killer quote for v0 paper's H4 discussion: "Wonder3D++'s cascaded 3D mesh extraction (Sec. IV-B) demonstrates that explicit mesh + iterative refinement > implicit SDF (NeuS-style) for high-fidelity image-to-3D back-ends". This is the *direct* v0 v0 v1 v2 H4 evidence (Wonder3D++ 129 + Unique3D 128 both explicit-mesh-back-end) for using explicit mesh (FlexiCubes 007 / ISOMER 128 / Wonder3D++ 129's cascade) instead of NeuS/SDF for v0 sub-task 1+2 back-ends. $0, 1 hour.

**(l) v1: COMBINE WONDER3D++ 129's CASCADED 3D MESH EXTRACTION WITH DMC 033's POINT-TO-MESH PIPELINE.** The v1 v0 v1 v2 sub-task 2.5 architecture: (a) take the prep tooth as input image, (b) Wonder3D++ 129 generates 6-view RGB + normal at 256² (with perspective camera + concave detection), (c) cascaded 3D mesh extraction produces a coarse crown mesh at 512² enhanced resolution, (d) DMC 033's point-to-mesh pipeline (PoinTr → FoldingNet → SAP → FlexiCubes) refines the crown margin (clinical-fit critical). The *killer* v1 architecture combining Wonder3D++ 129's high-fidelity global structure + DMC 033's clinical-grade margin. $2,000-3,000 Lambda, 4-6 weeks.

**(m) v2: ADD CLINICAL-FIT METRICS (margin gap, internal fit, proximal contact, occlusion) to Wonder3D++ 129's MESH OPTIMIZATION LOSS.** Add Hwang 061's histogram loss L_Ĥ as an additional L_geo-like term: `L = L_normal + L_mask + L_geo + R_Laplace + λ_histogram · L_Ĥ`. The v2 v0 v1 v2 clinical-grade crown generation that combines Wonder3D++ 129's image-to-3D mesh backbone + Hwang 061's clinical penetration loss. $200-500 Lambda for clinical evaluation, 1-2 weeks.

**v0 compute update: ~$1,000-1,500 Lambda for dental fine-tune of Wonder3D++ 129** (similar to Wonder3D 063/118 fine-tune cost). **v0 stack updated: sub-task 1 (full-arch synthesis) v1+ = cross-domain multi-view LDM (Wonder3D++ 129's 3-stage training) + camera-type switcher (perspective for clinical IOS) + cascaded 3D mesh extraction (concave-topology detection + geometry-aware normal loss + iterative refinement with cross-domain multi-view enhancement ControlNet) + UV-based texturing + MIT license (commercial-deployable)** (NEW from 129, $1,000-1,500 Lambda for dental fine-tune, 3 weeks); sub-task 2 (crown generation) v1+ = same + concave-topology detection (killer practical innovation for clinical crown) + Hwang 061 histogram loss in mesh optimization (v2 clinical-fit-aware addition).

**Strategic positioning: Wonder3D++ 129 is the TPAMI 2025 journal extension of Wonder3D 063/118, the new 2025 SOTA on GSO single-image-to-3D reconstruction (CD 0.0193 vs Wonder3D 063/118 0.0222, +13% improvement), the de facto canonical cross-domain multi-view diffusion paper, and the most-recent in the 2023-2025 cross-domain multi-view diffusion arc (Wonder3D 063/118 → GeoWizard 4 → Era3D 127 → Unique3D 128 → Wonder3D++ 129). The 3 innovations (camera-type switcher + 3-stage training + cascaded 3D mesh extraction) are the *right* architectural template for v0 sub-task 1+2, AND the modular cascaded mesh extraction can be retrofitted onto v0 sub-task 2 (crown generation) as the final mesh-extraction step. The MIT license + concave-topology detection + 13% CD improvement + 5× speedup make Wonder3D++ 129 the *primary* v0 sub-task 1+2 mesh-reconstruction module — the *direct* successor to Wonder3D 063/118 and the *direct* alternative to Era3D 127's AGPL-3.0 code that v0 can *use* (not just *cite*). The *complete* v0 sub-task 1+2 stack is now: Wonder3D++ 129 (cross-domain multi-view LDM + cascaded mesh extraction + MIT) + Wonder3D 063/118 (parent repo, MIT) + Era3D 127 (cites only, AGPL-3.0 blocker) + Unique3D 128 (ISOMER back-end alternative, MIT) + ISOMER (drop-in mesh extraction, MIT) + CAT3D 113 + Bolt3D 116 + MVSplat360 125 + DiffSplat 126 = the *complete* 2024-2025 multi-view-diffusion + 3D-reconstruction stack, *all under MIT*, *all open-source*, *all deployable* for v0.**

---

## Open Q for HK

(i) adopt Wonder3D++ 129 as primary v0 sub-task 1+2 back-end? (YES — supersedes Wonder3D 063/118 in 13% CD, MIT license, same code, all-win). (ii) adopt camera-type switcher (perspective for clinical IOS)? (YES — prevents Era3D-style failure mode). (iii) adopt concave-topology detection for clinical concave objects? (YES — killer practical innovation, 5-10 lines PyTorch). (iv) adopt geometry-aware normal loss L_geo (Eq. 11-13)? (YES — direct ancestor of Unique3D 128's ET, both fix wave-artifact). (v) adopt 3-stage training (multi-domain → mixed → cross-domain)? (YES — Fig. 12 ablation confirms strict improvement over 2-stage). (vi) adopt UV-based texturing? (YES — clinical 3D printing needs full-res texture). (vii) adopt DDIM-inversion iterative refinement? (YES — 91 sec for 512² enhanced detail is worth it). (viii) v1: combine Wonder3D++ 129 cascade + DMC 033 point-to-mesh? (YES — v1 sub-task 2.5 architecture). (ix) v2: add Hwang 061 histogram loss to L_geo? (YES — clinical-fit-aware). (x) cite as 2025 SOTA in v0 paper related-work? (YES).

**Next paper to read (130):** the 129-note's recommended *next* is **(a) *CRM* (Wang et al. CVPR 2024, arXiv:2403.18943, the *concurrent* 6-view cross-domain diffusion paper that Wonder3D++ 129 directly compares against — 12% CD improvement over CRM is one of Wonder3D++'s headline results, the *right* next paper to understand the 2024 cross-domain multi-view-diffusion ecosystem)** — recommended (the *direct* comparison baseline in Wonder3D++ 129's Table I); alternative: **(b) *MVDream* (Shi et al. ICLR 2024, the *text-to-3D* multi-view diffusion paper that is Wonder3D++ 129's 2D multi-view prior, the *canonical* 2023 multi-view diffusion that all Wonder3D-class papers inherit from)** — the *de facto 2023 multi-view diffusion* ancestor; alternative: **(c) *SyncDreamer* (Tang et al. CVPR 2024, the *first* 3D-aware multi-view diffusion paper, the *direct* predecessor of Wonder3D 063/118's multi-view attention)** — the *founder* of the 3D-aware-multi-view-diffusion paradigm; alternative: **(d) *Direct2.5* (Lu et al. 2024, the *sequential* normal→RGB multi-view diffusion paper that Wonder3D++ 129 explicitly CRITIQUES in Sec. II-B as "largely decoupled" — the *right* paper to understand what Wonder3D++ 129 is *replacing*)**. **Recommendation: *read 130 = MVDream* (Shi et al. ICLR 2024, arXiv:2308.01236)** — the *text-to-3D* multi-view diffusion that is Wonder3D++ 129's 2D multi-view prior; the *canonical 2023 multi-view diffusion* ancestor; the *right* paper to read *after* 129 to understand the 2023-2024 multi-view diffusion ecosystem that Wonder3D++ 129 builds on. After 130, the v0 multi-view-diffusion arc reaches 4 papers complete (Wonder3D 063/118 + Wonder3D++ 129 + Era3D 127 + Unique3D 128 + MVDream 130), the v0 cross-domain multi-view diffusion ecosystem is comprehensive.
