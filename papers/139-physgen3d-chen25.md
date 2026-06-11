# Paper 139 — PhysGen3D (now MiniTwin): Crafting a Miniature Interactive World from a Single Image

## TL;DR

**The first single-image-to-interactive-3D-world paper** — a *training-free* framework that orchestrates 6 pretrained vision models (GPT-4o, Grounded-SAM, InstantMesh, Dust3r, LaMA, Bilateral Normal Integration) to reconstruct an amodal 3D scene from a single image, runs a **material point method (MPM) physics simulation** (Taichi-Elements) with user-controlled initial velocity + material, and **physics-based-renders the result with two-pass shadow mapping** (Mitsuba). The killer design lesson is **"training-free digital twin"** — no (image, action, video) triplet collection is needed because the pipeline re-uses pretrained visual models for perception and an analytical MPM simulator for physics. The user-study result beats closed-source Pika 1.5, Kling 1.0, and Runway Gen-3 on physical plausibility + user-intent alignment while matching their rendering quality — the first paper to credibly claim "physics > photorealism" for image-to-video. **The 3-stage digital-twin paradigm (reconstruct → simulate → render) is the 2025 standard for "what-if" generation and the *direct* architectural template for v1's "crown preview from various angles" feature.** This closes the 4-paper physics-aware generative systems arc (137 WonderPlay = 4D optimization, 138 RealWonder = real-time video, **139 PhysGen3D = single-image 3D world**, [Phystwin = reconstruct-then-simulate inverse]).

## Research Question

**Q:** Can a single still image be turned into a *controllable, interactive, physically simulated* 3D miniature world — so the user can ask "what if I push this apple?" and get a physically plausible video back, with explicit control over initial conditions?

**Their answer:** **Yes, *without* task-specific training.** Compose 6 pretrained visual models (GPT-4o for category, Grounded-SAM for segmentation, InstantMesh for object mesh, Dust3r for background depth, LaMA for inpainting, Bilateral Normal Integration for collider surface) → register objects into scene with coarse-to-fine 2D-3D feature matching → run MPM simulation (Taichi-Elements) with user-specified velocity + material → physics-based render (Mitsuba, two-pass shadow) back into original image. **Key insight:** no (image, action, video) data collection is needed because the visual priors + analytical physics are *composable* off-the-shelf.

## Method

### 3-stage pipeline (Figure 2)

**Stage 1: 3D world creation** (Sec 3.1)
- **Segmentation:** GPT-4o identifies foreground object categories → Grounded-SAM detects + segments each instance `o^i ∈ R^{W×H×3}`
- **Mesh generation:** InstantMesh (Zero123++ for multi-view synthesis + multi-view reconstruction) produces object mesh `O`. For multi-object occlusions, **iterative inpainting** extracts each object sequentially (a quiet but important detail from supp)
- **Background handling:** Dust3r predicts depth `z` → unproject to point cloud `P` → Bilateral Normal Integration (BiNI) generates smooth collider surface `S` (the floor/wall for object-scene interaction). LaMA inpainting fills background regions *with shadows removed*
- **Object pose + scale:** Multi-stage coarse-to-fine 2D-3D feature matching (render N views on unit sphere, match to image features) + optimization for 6DoF pose + scale. The 2D-3D registration is the "hard" step because generated mesh is normalized and needs to align with real-world scale
- **Material inference:** GPT-4o infers mass, elasticity (Young's modulus), friction, and deformation characteristics from a single image. The supp has a "get them automatically using GPT-4o" example prompt at /by-luckk/PhysGen3D/blob/main/assets/gpt.md

**Stage 2: Dynamics simulation** (Sec 3.2)
- **Taichi-Elements MPM** (Material Point Method): particle-based hybrid Eulerian-Lagrangian, the *only* simulator that supports rigid + elastic + soft + liquid + granular + multi-material interactions in one engine
- User controls initial velocity vector + material property per object
- Background `S` serves as collider; objects bounce/roll/squash against it
- Stable, no catastrophic failures (the "robustness" comes from MPM's mature timestep control)

**Stage 3: Physics-based rendering** (Sec 3.3)
- **Mitsuba** renderer with **two-pass shadow mapping** (first pass from light POV, second pass from camera POV)
- HDR env map for relighting (data/hdr/teddy.exr in repo)
- Motion blur (shutter-time parameter, defaults to 0.0 = off)
- Integrates dynamic objects back into the *original* image (not a separate synthesized image), preserving the original lighting + background

### Why training-free works
The "no task-specific training" claim is strong because:
- All visual priors (segmentation, 3D, depth, inpainting) are *general-purpose* foundation models trained on massive internet data
- MPM is *analytical* — it doesn't learn physics, it solves the physics equations
- The pipeline is "orchestration" of pretrained experts, not a new model trained on (image, action, video) tuples — which the field can't collect at scale

## Results

**No traditional metrics — user study only.** The paper is honest that the comparison is closed-source Pika 1.5, Kling 1.0, Runway Gen-3, so no reproducible quantitative benchmark. They conduct "a carefully designed and rigorous user study" (their words).

**User study wins:**
- **Physical plausibility:** PhysGen3D wins (objects follow gravity, friction, soft-body dynamics, collision correctly)
- **User-intent alignment:** PhysGen3D wins (because user controls *exact* initial velocity and material, vs I2V models that can only be prompted with text)
- **Rendering quality:** comparable to I2V SOTA (Pika/Kling/Gen-3 have slight edge on raw photorealism from more data)
- **Controllability:** PhysGen3D wins decisively (the "drag-the-arrow" teddy bear is the *killer* demo)

**Demos (from project page):**
- Apple rolls with friction (rigid material, downward velocity)
- 3 stuffed animals collide (rigid bodies, complex contact)
- Toy potato bounces with soft-body dynamics (elastic material, downward force)
- Material editing: same scene rendered with rigid-rigid / elastic-rigid / soft-soft
- Motion editing: same teddy bear with jump-to-front / jump-to-right / jump-back
- Video editing: object exchange between two scenes, object removal

## Connections to H1-H5

**H1 (STRONG SUPPORT, 3rd composability model):** Like paper 138 RealWonder's 3 composability models (additive-loss, learned-bottleneck, physical-prior), PhysGen3D adds a *4th*: **pretrained-expert-orchestration**. The system is structurally 1-stage end-to-end at the *inference* level (one user click → one output) but composed of 6 independently-trained expert modules. For v0: confirms that "composability comes in 4 flavors" — additive-loss (Hwang 061), learnable-bottleneck (DMC 033), physics-prior (RealWonder 138), expert-orchestration (PhysGen3D 139). v0 sub-task 1 (full-arch synthesis) is *already* a 4-flavor composability stack via DiffSplat 126 + Dust3r 111 + Grounded-SAM + GPT-4o — validate the design.

**H2 (NEUTRAL — this is *not* a diffusion paper):** PhysGen3D is *training-free* — the "generative" component is MPM simulation (deterministic given initial conditions), not diffusion. The paper *compares to* diffusion-based I2V (Pika, Kling, Gen-3) and beats them on physics, loses on photorealism. For v0: H2's "latent diffusion refinement" (paper 126 DiffSplat) is a *complement* to PhysGen3D's expert-orchestration, not a substitute. v0 sub-task 1 should adopt BOTH: DiffSplat for unobserved-region completion + PhysGen3D-style expert orchestration for multi-step perception.

**H3 (WEAK BUT CLEAN):** The "rich context" is the 3D world itself, not prep/adjacent/opposing teeth. But the *pattern* of multi-source conditioning (GPT-4o text + Grounded-SAM mask + Dust3r depth + InstantMesh mesh) is the *exact* H3 mechanism for v0 sub-task 1 (intra-oral scan + FDI labels + margin segmentation + scanner calibration). v0 already has this 4-source conditioning via the *rich* H3 stack in paper 127 (Bolt3D cross-attn + FDI one-hot + EFReg scanner prediction). PhysGen3D confirms multi-source is the right design.

**H4 (MIXED):** Uses mesh (foreground objects from InstantMesh) + point cloud (background from Dust3r) + particles (MPM) — the *richest* substrate mix in our reading list. The 3DGS substrate (papers 114-126) is *not* used here because the camera-centric rendering + Mitsuba physical-rendering is the *right* choice for interactive simulation. For v0 sub-task 2 (crown gen) the *mesh* substrate is right (clinical CAD uses STL); for v0 sub-task 1 the *3DGS* substrate is right (generative completion); for v0 sub-task 3 (crown visualization) the *Mitsuba* physical-renderer is right (chairside preview). Three different substrates for three different sub-tasks. PhysGen3D validates this multi-substrate design.

**H5 (STRONGEST SUPPORT, the *ultimate* H5):** Training-free = *zero* task-specific data. This is the *pinnacle* of H5 — no synthetic + real finetune, no patient-specific retraining, no scanner-specific finetune, just *pretrained* visual priors + analytical physics. For v0: validates the v0 plan of "pretrain on Objaverse + 3DTeethSeg22 + ToSynFCD, finetune minimally on 1K clinical scans" — but pushes the design space to "even better: use pretrained multi-modal models (GPT-4o, Grounded-SAM, SAM 3D, VGGT) and analytical methods, train only the *small* perceptual adapters". This is the *killer* v1 design — replace DMC with "SAM 3D + GPT-4o material advisor + Mitsuba renderer" for chairside deployment with no GPU training.

## Surprises / Interesting Things Buried in Section 4

1. **"Training-free" is the *philosophical breakthrough* for 2025-2026 generative systems.** Papers 137-139 form a 4D / real-time-video / 3D-world arc that *all* converge on "compose pretrained modules" rather than "train a new big model on (image, video, action) tuples". For v0: the implication is that the v0 *system* should be a composition of pretrained models + small finetuned adapters, not a monolithic 1-3B-param v0 model trained on dental data. This is the 2026 design pattern for *application* AI (vs *frontier* AI).

2. **Bilateral Normal Integration (BiNI, 2023) is the unsung hero for collider geometry.** The dust3r depth → point cloud → BiNI surface pipeline is a *simple, robust* alternative to NeRF or 3DGS for "the floor that the objects rest on". For v0: when modeling the gum line as the *collider* for crown generation, BiNI-style surface reconstruction from depth is the *right* v0 sub-task 1.5 choice (the v0 paper's contribution: "dental BiNI-style surface reconstruction from intra-oral depth scan, validated against margin-line ground truth").

3. **MPM is the *only* simulator that handles rigid + elastic + soft + granular + fluid in one engine.** PyBullet, MuJoCo, Bullet, Isaac Gym all handle rigid + soft via different libraries. MPM is the *unified* substrate. For v1 dental simulation (tooth wear, food bolus, tongue pressure), MPM is the *right* choice if v1 ever ships an interactive simulation. v0 doesn't need this, but the *reference architecture* for v1's "patient sees the crown behavior under bite force" feature is Taichi-Elements MPM.

4. **The "iterative inpainting" for multi-object occlusions is a quiet but important detail (supp).** For 3+ occluded objects (e.g., three teeth partially hidden by gingiva), the *correct* strategy is to inpaint + reconstruct each one iteratively, not all at once. For v0 sub-task 1: when reconstructing an arch with 5+ occluded teeth, the iterative-inpainting strategy is *essential* (one-shot would fail on the rear molars hidden by the front incisors).

5. **The Dust3r → BiNI → MPM pipeline fails on "transparent / reflective / dusty" surfaces (Sec 5 Limitation).** This is the *exact same failure mode* as v0 will see with crown (reflective ceramic) + saliva (transparent liquid) + plaque (dusty). The paper's *proposed fix* is "use VGGT or SAM 3D for better 3D reconstruction" — a v0-relevant suggestion. v0 should benchmark *which* 3D-reconstruction method handles crown surfaces best (Dust3r vs VGGT vs SAM 3D vs PVD-AF-DiGS-FC vs DiffSplat).

6. **The "vomp" (NVIDIA, 2025) follow-up cites both PhysGen (Liu 2024) and PhysGen3D for VLM-based material inference.** This is the *direction* the field is heading: VLM (GPT-4o, LLaVA) infers material from image → MPM simulates → Mitsuba renders. v0 paper should cite both PhysGen and vomp as the *general* pattern, and position v0 as the *first* clinical dental application.

7. **Two-pass shadow mapping is the gold standard for physically-based 3D rendering.** Mitsuba's differentiable renderer is *the* SOTA for inverse rendering — used in neural radiance fields, NeRF, gaussian splatting, and many 2024-2026 papers. v0 should consider Mitsuba for the v0 sub-task 3 (chairside crown preview) — it's the *only* renderer that gives physically-accurate shadows + relighting + materials without DL.

8. **GPT-4o infers mass + elasticity + friction from a single image is the *killer* demo for dental material selection.** The paper's GPT-4o prompt (in supp) takes a "teddy bear" image and returns "soft body, low Young's modulus, low friction, etc". For v1 dental material advisor: GPT-4o takes a prep scan → returns "zirconia (high hardness) vs PFM (medium) vs lithium disilicate (medium-high) vs gold (low for posterior)" with reasoning. This is a *killer* v1 product feature (no other vendor does AI material selection).

9. **The "drag the arrow" interactive demo is the killer UX feature for v1 chairside.** The user drags an arrow on a 2D rendering of the image, the velocity vector updates in real-time, the simulation replays. For v0 paper positioning: v0 sub-task 3 (crown preview) could adopt the same "user-drag-arrow" interface to let the dentist *manually adjust* the crown position before exporting. This is the *killer* v1 UX differentiator.

10. **The "video editing: object exchange between two scenes" demo is the v0 sub-task 3 inspiration.** Two intra-oral scans of the same patient at different timepoints → exchange one tooth between them → see the natural variation over time. For v1 chairside: "this is what your tooth will look like in 5 years if you don't get a crown" (compare to predicted-future scan).

## Quote-Worthy Sentences

> "PhysGen3D, a novel framework that transforms a single image into an amodal, camera-centric, interactive 3D scene"

> "By combining advanced image-based geometric and semantic understanding with physics-based simulation, PhysGen3D creates an interactive 3D world from a static image"

> "Due to the use of large pretrained models, our pipeline operates effectively without task-specific training"

> "our approach is training-free and generalizable to all objects in the world"

> "explicitly controls the motion and interaction with simulation, allowing us to create more sophisticated effects without the need for extensive training data"

> "a comprehensive understanding of objects' relationships, geometry, appearance, material, and physical properties"

> "However, obtaining this understanding from a single image is highly ill-posed" (the fundamental challenge)

> "We adopt InstantMesh, which uses Zero123++ to synthesize multi-view images from the segmented object image"

> "physical property understanding from language-embedded feature fields" (the related-work framing that v0 paper should mirror)

> "Compared to closed-source SOTA video AIGC models such as Pika, Kling, and Gen-3, our framework provides significantly more flexible control over object motions"

## Code / Data Links

- **Project page:** https://by-luckk.github.io/PhysGen3D
- **Paper (arXiv v1):** https://arxiv.org/abs/2503.20746
- **Paper (arXiv HTML):** https://arxiv.org/html/2503.20746v1
- **CVPR 2025 page (camera-ready renamed "MiniTwin"):** https://cvpr.thecvf.com/virtual/2025/poster/33820
- **Code:** https://github.com/by-luckk/PhysGen3D (early stage, conda env Python 3.10, install via `bash env_install/env_install.sh && bash env_install/download_pretrained.sh`)
- **GPT-4o material prompt:** https://github.com/by-luckk/PhysGen3D/blob/main/assets/gpt.md
- **DOI:** 10.1109/CVPR52734.2025.00579 (CVPR proceedings)
- **Semantic Scholar:** https://www.semanticscholar.org/paper/PhysGen3D%3A-Crafting-a-Miniature-Interactive-World-a-Chen-Jiang/9f868188d3c8cf09a2cc684e1c95e8cfdff1cef1
- **Cited by NVIDIA vomp 2025:** https://research.nvidia.com/labs/sil/projects/vomp/vomp_compressed.pdf
- **Citations:** ~70-100 GS as of 2026-06-11 (15 months post-arXiv, CVPR 2025 paper)

## For Our Project (Dental Crown Gen)

### Direct takeaways

**(a) The 3-stage digital-twin paradigm (reconstruct → simulate → render) is the *direct architectural template* for v1's "what-if crown preview" feature.** v0 paper positions v1 as "first clinical application of the 2025 PhysGen3D-style digital-twin paradigm". v0's contribution is the *clinical specialization* — replace general-purpose GPT-4o with dental-trained GPT-4o (Cao25 paper 026 for FDI labels, Alsheghri24 paper 123 for margin line, Hwang18 paper 061 for clinical-fit priors). This is a *killer* v0 paper positioning.

**(b) H5 (training-free) is the *killer* design lesson for v0 → v1 evolution.** v0 trains on Objaverse + 3DTeethSeg22 + ToSynFCD; v1 is *training-free* for new clinic deployments (just plug in the patient's intra-oral scan and the pretrained modules work). v0 paper should *position* v0 as "v0 requires 1K-10K clinical cases for finetuning, v1 is training-free via PhysGen3D-style expert orchestration".

**(c) GPT-4o for material inference is the *killer* v1 product feature.** v0 paper: "we use GPT-4o to predict prep 'wear' → suggest crown material (zirconia vs PFM vs lithium disilicate vs gold) with confidence score" — *no other vendor* does AI material selection. Cite the GPT-4o prompt from PhysGen3D's supp (`/by-luckk/PhysGen3D/blob/main/assets/gpt.md`) as the v0 paper's *methodology template*.

**(d) Bilateral Normal Integration (BiNI) is the *right* v0 sub-task 1.5 collider-surface method.** The Dust3r → BiNI → MPM pipeline in PhysGen3D is the *cleanest* "intra-oral depth → gum-line collider" pipeline in our reading list. v0 paper contribution: "BiNI-style surface reconstruction from intra-oral depth scan, validated against margin-line ground truth". BiNI is a 2023 paper (Cao et al.), simple, robust, no GPU required.

**(e) Mitsuba two-pass shadow mapping is the *gold standard* 2025-2026 renderer for chairside crown visualization.** v0 sub-task 3 (crown preview) should adopt Mitsuba + HDR env maps for physically-accurate shadows + relighting + materials. This is the *only* renderer that gives sub-pixel-accurate shadows + diffuse/specular IBL + motion blur without DL. v0 paper contribution: "Mitsuba-rendered chairside crown preview with HDR env map and two-pass shadow".

**(f) The "iterative inpainting" for multi-object occlusions is *essential* for v0 sub-task 1.** An intra-oral arch has 28-32 teeth, 5+ occluded by gingiva or adjacent teeth. v0 sub-task 1 should *explicitly* use iterative inpainting: (i) segment all visible teeth, (ii) inpaint occluded regions, (iii) re-segment, (iv) repeat. Cite the supp of PhysGen3D as the methodology reference.

**(g) MPM (Taichi-Elements) is the *right* v1 simulation engine for v1 sub-task 2.5 "bite-force simulation".** v0 doesn't ship simulation, but v1's "patient sees crown behavior under bite force" feature needs MPM. The *right* v0 paper positioning is "v0 ships static crown mesh, v1 will ship interactive MPM simulation". Cite Taichi-Elements + PhysGen3D as the v1 reference architecture.

**(h) "Camera-centric 3D world" is the *right* v0 sub-task 1 design.** PhysGen3D reconstructs the 3D world in the *camera's* coordinate frame, not in a canonical world frame. For v0 intra-oral scanner: the *camera-centric* frame is the natural choice (the scanner moves around the mouth, the *camera's* view is what the dentist sees). Cite this as a v0 design rationale.

**(i) The "drag-the-arrow" interactive demo is the *killer* v1 UX feature.** v0 paper: "v0 is a one-shot generation system; v1 will add interactive 'drag-the-arrow' UX for crown position adjustment" — same paradigm as PhysGen3D's teddy-bear demo. This is a *killer* clinical selling point ("the dentist can manually adjust the crown before exporting").

**(j) Use of GPT-4o + Grounded-SAM + InstantMesh + Dust3r + LaMA is the *right* v0 sub-task 1 perception toolchain.** All 5 are open-source / API-accessible. v0 paper can *list* them as the v0 perception stack. Add Bilateral Normal Integration (BiNI) for the collider surface, Mitsuba for the renderer, Taichi-Elements for the v1 simulator. **The 8-component "PhysGen3D-style" stack is the *complete* v0 → v1 design.**

### v0/v1 implications summary

- **v0 cost:** no change ($0, literature reading)
- **v0 stack update:** v0 sub-task 1 perception toolchain = GPT-4o + Grounded-SAM + InstantMesh + Dust3r + LaMA + BiNI; v0 sub-task 3 renderer = Mitsuba + HDR env map; the *complete* v0 perception + rendering stack is *exactly* the PhysGen3D 8-component stack with dental-specific finetuning
- **v0 paper positioning:** v0 is the *first* clinical dental application of the 2025 PhysGen3D-style digital-twin paradigm (the v0 paper's "related work" section should position v0 in the digital-twin arc alongside WonderPlay 137, RealWonder 138, PhysGen3D 139, vomp 2025 — the "physics-aware generative systems" arc that will dominate 2026-2028 clinical AI)
- **v1 candidates:** (i) GPT-4o material advisor (replace vendor's "what material?" dropdown with AI recommendation), (ii) Mitsuba-rendered chairside preview with HDR env map (replace simple CAD rendering with physically-accurate rendering), (iii) Taichi-Elements MPM bite-force simulation (v1 long-term, with 12-18 months R&D), (iv) "drag-the-arrow" interactive crown-position UX (v1 medium-term, with 6-12 months R&D)
- **Compute:** v0 unchanged, v1 Mitsuba + Taichi-Elements + GPT-4o API = +$200-500/month in production (mostly GPT-4o API costs)

## Next Paper to Read

**vomp (Shen et al. NVIDIA 2025, "Volumetric Mechanical Property understanding")** — the direct follow-up to PhysGen3D that uses VLM (LLaVA) for mechanical property inference from a single image, then runs MPM simulation. The "vomp" name is a play on "VoMP = Volumetric Mechanical Properties". Cites both PhysGen (Liu 2024) and PhysGen3D (Chen 2025) as the predecessors. Will close the physics-aware arc by showing the *mechanical-property-only* version of PhysGen3D, validating the GPT-4o material advisor direction for v1.

(Alternative: **PhysDreamer (Zhang et al. CVPR 2024)** — the *prior* PhysGen-style paper that uses video diffusion as the "learned physics simulator" instead of MPM. The "video diffusion is the physics" alternative to MPM-is-the-physics. v0/v1 should consider both architectures for the v1 simulation: MPM for accuracy, video diffusion for "soft" materials like food bolus that MPM handles poorly.)

**Recommendation: *read 140 = vomp (Shen NVIDIA 2025)*** — the direct PhysGen3D follow-up, validates GPT-4o material advisor for v1, closes the physics-aware arc.
