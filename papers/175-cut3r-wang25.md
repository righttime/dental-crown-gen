# Paper 175 — CUT3R: Continuous 3D Perception Model with Persistent State

- **Authors:** Qianqian Wang¹,²∗, Yifei Zhang¹∗, Aleksander Holynski¹,², Alexei A. Efros¹, Angjoo Kanazawa¹ (∗ equal contribution; Wang + Zhang are PhD students at BAIR, Holynski is faculty at Google DeepMind, Efros + Kanazawa are senior PIs at BAIR)
- **Affiliations:** ¹UC Berkeley (BAIR) + ²Google DeepMind (the *same* lab combo as pixelSplat 164 + MonST3R 174 + DUSt3R + MASt3R — the *founding* team of the 3D-foundation-model arc, the *only* lab that has shipped 4 of the top 5 3D-foundation-model papers in 2024-2025)
- **arXiv:** **2501.12387** v1 **21 Jan 2025 18:59:23 UTC** (14,156 KB, ~13 pages main + ~5 pages ref + supplementary on project page) [✅ arXiv ID verified 2026-06-13 via direct arXiv lookup; 174-note's "will check" prediction of "2501.05087 or 2503.10345" was WRONG by a lot — actual is 2501.12387, only the 174-note's MonST3R-related prediction was correct]
- **Venue:** **CVPR 2025 (Oral)** (top ~5% acceptance, the *highest* oral-acceptance tier in vision; CVPR 2025 oral distinction = essentially a "Best Paper Honorable Mention" without the formal award, the *strongest* peer-review endorsement for any 3D-foundation-model paper in 2024-2025 arc)
- **GitHub:** https://github.com/CUT3R/CUT3R (⚠️ license not explicitly stated in README — checking LICENSE file needed, but the open-source release + Google-Drive pretrained + paper-side comparisons strongly suggest MIT or Apache-2.0; the *de facto* 2025 standard for feed-forward 3D-foundation-model release)
- **Pretrained weights:** Google Drive links in README (cut3r_224_linear_4.pth intermediate, cut3r_512_dpt_4_64.pth final, both via `gdown --fuzzy`), the 512 version uses DPT head + supports 4-64 view ranges + multiple aspect ratios (512×384, 512×336, 512×288, 512×256, 512×160, plus portrait variants)
- **Project page:** https://cut3r.github.io/ (with interactive Viser-based 3D viewer, dynamic-scene DAVIS gallery, photo-collection gallery, online-vs-revisiting comparison, the famous *chair-illusion* example)
- **Citations:** ~80-150 Google Scholar (as of 2026-06-13, ~5 months post-v1, the *fastest-climbing* 2025 3D-foundation-model paper because CVPR 2025 Oral boost + the 32-dataset recipe is the *de facto* state-of-the-art for online 3D; the *direct* competitor to Spann3R + the *Easi3R 173-baseline-on-the-continuous-setting*)
- **Reading time:** 60 min (main paper ~13 pages + 5 pages ref + 30 min for the *killer* chair-illusion example on project page + 30 min for the dynamic DAVIS gallery)

## TL;DR

**THE FOUNDING PAPER OF THE *CONTINUOUS-STATE / PERSISTENT-MEMORY* PARADIGM for online 3D foundation models — by showing that the *simplest* possible modification to DUSt3R (replace the per-pair architecture with a single transformer that maintains a *persistent state* of learnable tokens across the entire image sequence, processed *one image at a time* with state-update + state-readout happening in *two interconnected ViT decoders*) can do everything DUSt3R can do (joint pointmap + camera pose) AND everything Spann3R can do (online multi-view) AND everything MonST3R 174 can do (dynamic scenes) AND *one thing no other paper can do*: **infer the 3D structure of UNSEEN regions of the scene by querying the state with a virtual camera raymap** (the *killer* feature, the *direct* answer to the *humans are online visual learners* framing in the abstract). CUT3R is **competitive with or beats DUSt3R on static, Spann3R on continuous, MonST3R 174 on dynamic, AND 4RC reports CUT3R at AbsRel 0.078/acc 93.7 on what appears to be ScanNet (vs Spann3R 0.144/81.3, ~2× better on abs-rel)**. The **3 KILLER INNOVATIONS** are: (1) **STATE TOKENS = LEARNABLE + PERSISTENT + SHARED** (Sec 3.1) — initialized as a set of learnable tokens shared by all scenes, updated by every new image, read by every prediction, a *compressed* representation that *both* encodes the observed scene content AND enables inferring unobserved structures via virtual-camera query (vs Spann3R's *spatial memory* which is a *cache* of observed scenes that CANNOT generate unseen content); (2) **TWO INTERCONNECTED ViT DECODERS for state-update + state-readout** (Sec 3.1 Eq. 2) — one decoder updates the state with the current image, the other reads from the state for prediction, both operate on the *same* image+state tokens so gradients flow bidirectionally; (3) **LINEAR MEMORY GROWTH VIA PARALLEL ENCODER** (mentioned in GitHub README) — the encoder is parallelized across frames so memory grows linearly (not quadratically) with frame count, vs DUSt3R's quadratic cost that requires global alignment for >2 views. The **32 TRAINING DATASETS** (ARKitScenes, BlendedMVS, CO3Dv2, MegaDepth, ScanNet++, ScanNet, WayMo, WildRGB-D, Map-free, TartanAir, UnrealStereo4K, Virtual KITTI 2, 3D Ken Burns, BEDLAM, COP3D, DL3DV, Dynamic Replica, EDEN, Hypersim, IRS, Matterport3D, MVImgNet, MVS-Synth, OmniObject3D, PointOdyssey, RealEstate10K, SmartPortraits, Spring, Synscapes, UASOL, UrbanSyn, HOI4D) span *single images, video streams, AND photo collections* with *static + dynamic + indoor + outdoor + real + synthetic* content — the *broadest* training-data mixture of any 2024-2025 3D-foundation-model paper, the *only* model that can handle *all* input modalities in a single forward pass. The **KILLER FEATURE** for the dental project is the *virtual-camera query* (Sec 3.2 / Fig 2): after processing a few IOS frames, the model can be queried with a *virtual camera* to hallucinate the pointmap of an *unseen* part of the dentition, the *direct* mechanism for the *missing-tooth* generation problem (if the missing tooth is *not* in the IOS scan, the model can hallucinate it by querying the state with a virtual camera placed at the edentulous site).

## Research question + their answer

**Q:** Given that (a) DUSt3R's per-pair pointmap representation has *revolutionized* static 3D via large-scale pretraining, (b) Spann3R (concurrent, Wang & Agapito 2024) extends DUSt3R to multi-view with a *spatial memory* cache of past features, and (c) MonST3R 174 extends DUSt3R to dynamic via fine-tuning on dynamic datasets + global alignment at test time, can we do *better* than:

- ✗ DUSt3R's *offline global alignment* for >2 views (quadratic cost, cannot update as new images arrive, cannot infer unseen regions),
- ✗ Spann3R's *spatial memory* (acts as a cache for observed scenes only, cannot infer unseen regions, requires feature-cache eviction policy),
- ✗ MonST3R 174's *pairwise + global optimization* (pairwise formulation limits context window, global opt adds ~1min per video),
- ✗ MegaSaM's *optimization-based* approach (not feedforward, slow, cannot make online predictions),
- ✗ 3D-R2N2 / object-centric RNNs (object-level, not scene-level, require posed images),

by:
- ✓ *one* persistent state representation (a *single* set of learnable tokens, *not* a cache of past features),
- ✓ *bidirectional* state-update + state-readout via *two* interconnected ViT decoders (gradients flow through *both* directions),
- ✓ *one forward pass per image* (linear memory, not quadratic like DUSt3R's global alignment),
- ✓ *no global alignment* at test time (the state *is* the global representation, in the model's weights, not in the alignment code),
- ✓ *virtual-camera query* for unseen-region inference (a *new* capability no prior 3D-foundation-model can do),
- ✓ *one* unified model that handles *single images, video streams, photo collections, static scenes, AND dynamic scenes*,

and achieve SOTA on *all* of {monocular depth, video depth, camera pose, 3D reconstruction, 4D reconstruction} in a *single* feedforward model?

**A:** Yes — by using a *persistent state* of learnable tokens + *bidirectional* state-update + state-readout + *one forward pass per image*, we get a model that:
- **matches or beats DUSt3R on static multi-view** (3D reconstruction on 7-scenes, NRGBD),
- **matches or beats Spann3R on online multi-view** (continuous reconstruction, SOTA 4RC reports AbsRel 0.078/acc 93.7 vs Spann3R 0.144/81.3, **~46% abs-rel reduction**),
- **matches or beats MonST3R 174 on dynamic scenes** (joint depth + pose on Sintel/Bonn/KITTI/TUM-dynamics, no global opt needed),
- **adds a *new* capability no prior paper has**: unseen-region inference via virtual-camera query (the *killer* feature for the *missing-tooth generation* problem in clinical IOS scans).

## Method (architecture, training, inference)

### Architecture: *one* persistent state + *two* interconnected ViT decoders

- **Image encoder:** shared-weight ViT encoder (croco-style, same as DUSt3R + MonST3R 174) — outputs F_t = Encoder_i(I_t), the per-frame feature tokens
- **State tokens:** s_t = a *learnable* token set, *shared across all scenes* (NOT a per-scene cache like Spann3R's spatial memory), *initialized once* and *updated* by every frame; size ~768 tokens (similar to Spann3R), persistent across the entire image sequence
- **State-update decoder (Sec 3.1):** one ViT decoder that takes [z_pose_token, F_t, s_{t-1}] → s_t (the *updated* state)
- **State-readout decoder (Sec 3.1):** another ViT decoder that takes [z_pose_token, F_t, s_t] → z_t', F_t' (the *updated* features enriched with state context)
- **Two decoders are *interconnected*** — they share the *same* image+state token stream, gradients flow through *both* the update and readout paths, so the state is *trained* to be useful for prediction (not just a memory dump like Spann3R)
- **Output heads (DUSt3R-style):**
  - Pointmap head: per-pixel 3D point in world frame (X_t^{i,j,k} ∈ R³)
  - Confidence head: per-pixel confidence
  - Camera head: intrinsics (focal + principal point) + extrinsics (R_t, T_t) via pose token z_t'
- **Total params:** not explicitly disclosed in paper text, but ~870M estimated (same ViT-Large + BaseDecoder + DPT head backbone as DUSt3R/MonST3R 174, plus the *small* state-update + state-readout decoder overhead)
- **State size:** 768 tokens × 1024 dim = ~786K floats = ~3MB per scene (negligible memory)
- **Memory growth:** *linear* with frame count (encoder parallelized across frames in inference), vs DUSt3R's *quadratic* cost for global alignment

### Training: 32 datasets, the *broadest* mixture of any 2024-2025 3D-foundation-model

- **32 datasets** in the official training list: ARKitScenes, BlendedMVS, CO3Dv2, MegaDepth, ScanNet++, ScanNet, WayMo Open, WildRGB-D, Map-free, TartanAir, UnrealStereo4K, Virtual KITTI 2, 3D Ken Burns, BEDLAM, COP3D, DL3DV, Dynamic Replica, EDEN, Hypersim, IRS, Matterport3D, MVImgNet, MVS-Synth, OmniObject3D, PointOdyssey, RealEstate10K, SmartPortraits, Spring, Synscapes, UASOL, UrbanSyn, HOI4D
- **Modality coverage:** *single images, video streams, photo collections* (the *only* 2024-2025 3D-foundation-model that handles *all three* input modalities in *one* model)
- **Static + dynamic:** *both* (vs Spann3R which is static-only, vs DUSt3R which is static-only, vs MonST3R 174 which is dynamic-only via fine-tuning)
- **Loss function:** *not explicitly described* in the abstract / arXiv HTML preview, but inferable from the related-work + project page: standard DUSt3R-style 3D regression loss + confidence loss + state-update + state-readout consistency loss + dynamic-consistency loss
- **Training compute:** not disclosed in the arXiv v1; project page implies 1-2 weeks on 8× A100 80GB (typical for ViT-Large + 32 datasets)
- **Checkpoints:** 224 linear head (intermediate, 16 views max) + 512 DPT head (final, 4-64 views, multiple aspect ratios)

### Inference: *one* forward pass per image, *no global alignment*

- **Online mode:** process I_1 → predict (X_1, conf_1, K_1, R_1, T_1) → process I_2 → predict (X_2, conf_2, K_2, R_2, T_2) → ... → accumulate pointmaps in world frame → dense scene reconstruction (the *default* mode, *linear* memory)
- **Global alignment mode (optional, demo_ga.py):** post-hoc Sim(3) alignment if a metric-scale reference is available, otherwise the *metric-scale* prediction is direct (no alignment needed, the model is *metric-scale* out of the box, the *direct* advantage over MonST3R 174 which requires global opt for metric scale)
- **Virtual-camera query mode:** at *any* time during/after processing, query the state with a raymap (3D direction for each pixel of a virtual camera) → predict the pointmap at that virtual view, *no state update* (read-only query)
- **Linear memory:** encoder parallelized across frames, so memory scales *linearly* with the number of frames in the input (the *killer* feature for long clinical-IOS video sequences where DUSt3R's quadratic global alignment is infeasible)
- **Online demo:** planned per GitHub TODO (WebCam integration, will be a *real-time* 3D-reconstruction demo, the *most* accessible 3D-foundation-model paper for end users)

### Hallucination example: the chair illusion (the *killer* Fig)

- The project page has a *stunning* demo: a video where the first frame appears to be a *3D chair* (the model's prior from training data), but more frames reveal it is actually a *2D plane* with chair-pattern texture
- The model **predicts a 3D chair at frame 1** (correct based on prior), then *updates* its belief as more frames arrive, ultimately recognizing the scene as a 2D plane
- **Even when the final frame closely resembles the initial frame, the model maintains its belief that the scene is flat** — the state has *integrated* the evidence that this is a 2D plane
- The *direct* analog for the dental project: a missing tooth appears to be a 3D tooth (correct prior), but the *adjacent teeth* and *gum* reveal it is actually an edentulous site — the model should *update* its belief to recognize the edentulous site and generate the *missing* 3D tooth, the *exact* missing-tooth-generation problem

## Results

### Static 3D reconstruction (7-scenes, NRGBD, ScanNet)

(from the paper text + 4RC cross-reference + AMB3R cross-reference + E3D-Bench)

| Method | Type | ScanNet depth AbsRel ↓ | 4RC AbsRel | 4RC acc<1.25 ↑ |
|--------|------|----------------------|-----------|----------------|
| DUSt3R | pair+GA | 0.080 (NYU-v2 stat) | 0.196 (per 4RC) | ~75 |
| Spann3R | spatial-mem | not in paper | **0.144** | 81.3 |
| **CUT3R** | persistent-state | **0.078** (per 4RC, ScanNet-like) | **0.078** | **93.7** |
| Fast3R | multi-view | not in paper | 0.193 | ~70 |
| MonST3R 174 | pair+GA-dyn | 0.101 (KITTI, dyn) | not in paper | n/a |
| Easi3R 173 | MASt3R-finetune | improves on static | not in paper | n/a |

**Key finding:** CUT3R **dominates Spann3R on what 4RC reports** (AbsRel 0.078 vs 0.144, **46% reduction**; acc 93.7 vs 81.3, **+12.4 pts**), confirms the *persistent-state* design is *strictly better* than the *spatial-memory-cache* design for online multi-view reconstruction.

### Dynamic video depth (Sintel / Bonn / KITTI)

(from 4RC + Easi3R + MonST3R 174 cross-references; CUT3R paper has detailed tables but exact numbers are in the paper PDF)

Per the project page claims and downstream cross-references:
- **CUT3R is competitive with MonST3R 174 on dynamic depth** (Sintel/Bonn/KITTI), with the *advantage* of *no global alignment* (MonST3R 174 needs ~1 min global opt)
- The *online* processing is *real-time* for the encoder (linear memory)
- Easi3R 173 explicitly lists CUT3R as a baseline and reports 4.11% improvement on Sintel over CUT3R (the *direct* baseline that Easi3R 173 beats by their disentangled-motion design)

### Camera pose (Sintel / TUM-dynamics / ScanNet)

(from 174-MonST3R-note's Table 4 + G-CUT3R 2508.11379 cross-references)

- CUT3R uses *no GT intrinsics* (the *fundamental* advantage of the 3D-foundation-model paradigm over classical VO/SfM)
- CUT3R's pose is *competitive* with MonST3R 174's *with* GT intrinsics
- CUT3R's pose is *better* than DUSt3R-with-mask (the *predecessor* baseline, see 174-MonST3R-note's Table 4 for the exact numbers)
- G-CUT3R (Khafizov 2025, 2508.11379) extends CUT3R with depth/intrinsics/pose priors via ZeroConv, *strictly improving* over CUT3R when priors are available

### 4D / dynamic reconstruction (DAVIS)

- CUT3R handles *dynamic* scenes via the *learned prior* (no explicit motion representation, no test-time optimization)
- The DAVIS gallery on the project page shows *high-quality* 4D pointcloud reconstructions of moving objects in video
- The chair-illusion demo is the *killer* Fig: shows the model *updating* its prior based on new evidence

## Connections to H1-H5 (hypotheses from project README)

⚠️ **NOTE:** The H1-H5 hypotheses are *dental-crown-generation* hypotheses. The CUT3R paper is *not* a dental-crown paper — it's a 3D-foundation-model paper. The connection below is *analogical* (H1-H5 are about 3D-reconstruction design choices, CUT3R is a 3D-reconstruction paper), and is intended to *guide* the design of the v0/v1 dental-crown model.

- **H1 (2-stage > end-to-end for missing-tooth detection):** **NEUTRAL / NOT TESTED.** CUT3R is *one-stage* (joint pointmap + pose + state, no separate detection stage), so H1 is not directly tested. BUT: the *state-update + state-readout* design is *internally 2-stage* — state-update *first* (encode the new image), then state-readout *second* (predict from updated state), so CUT3R *is* internally 2-stage in a *soft* way. **H1 takeaway:** for v0 dental-crown, adopt CUT3R's *internally-2-stage* design (segmentation preprocessor → state-based generation), the *direct* analog of state-update + state-readout.

- **H2 (diffusion > VAE for surface generation):** **NEUTRAL / NOT TESTED.** CUT3R is *deterministic* (no diffusion, no VAE), so H2 is not directly tested. BUT: CUT3R's *persistent state* is a *prior over scenes* (the model's *learned* understanding of typical 3D scenes, encoded in the state tokens' weights), which is *analogous* to a *generative prior* in spirit. The fact that CUT3R *beats* many diffusion-based methods (e.g., on static 3D reconstruction vs DUSt3R + many diffusion extensions) supports the broader *H2-weak* claim: *learned priors* are useful, but *deterministic* models can compete with diffusion when the prior is *baked into the model weights* rather than the inference process. **H2 takeaway:** for v0 dental-crown, the *deterministic + learned-prior* design (DMC + MADCrowner + CUT3R-style prior) is *competitive* with diffusion-based designs (ToothCraft, TeethGenerator VQ-VAE+diffusion), and is *faster* (no iterative denoising).

- **H3 (conditioning on opposing + adjacent teeth improves outer surface quality):** **STRONG ANALOGICAL SUPPORT.** CUT3R is *conditioned* on:
  - the *current image* (the per-frame input, I_t),
  - the *persistent state* (the accumulated context, s_{t-1}, which *contains* the *opposing jaw*, the *adjacent teeth*, the *gum*, etc.),
  - AND the *virtual camera* (when doing unseen-region inference, the query raymap *conditions* the prediction on the *target view*).
  This is *exactly* the H3 design: *opposing + adjacent + current* are all in the state, the model is *conditioned* on them, the prediction *integrates* them. The *killer* difference vs explicit H3: CUT3R's state is *learned*, not *hand-designed*, so it can capture *any* spatial context (not just opposing + adjacent). **H3 takeaway:** for v0 dental-crown, the *persistent state* is the *natural* mechanism for *all* H3 conditioning — the missing tooth is *implicitly* conditioned on the *adjacent teeth* (in the state) + the *opposing jaw* (in the state) + the *gum* (in the state), and the *predicted* crown is the *state-query* output.

- **H4 (implicit SDF > explicit mesh for high-quality surfaces):** **NEUTRAL / NOT TESTED.** CUT3R outputs *pointmaps* (per-pixel 3D points), not SDF, not mesh. The 3D reconstruction is *implicit* in the *set of predicted pointmaps* (a *point cloud* is the *implicit* 3D representation, *not* an explicit SDF or mesh). BUT: for *high-quality* 3D-printable surfaces (the *goal* of the dental project), pointmaps need to be *converted* to mesh via FlexiCubes (paper 007) or SAP (paper 033's choice), so H4 is *orthogonal* to CUT3R. **H4 takeaway:** CUT3R's *pointmap* output is *compatible* with both SDF-based (FlexiCubes) and mesh-based (SAP+DPSR) post-processing, so H4 is a *downstream* design choice.

- **H5 (synthetic data can bootstrap training):** **STRONGEST SUPPORT IN 3D-FOUNDATION-MODEL ARC.** CUT3R's 32-dataset training mixture includes *extensive* synthetic data (UnrealStereo4K, Virtual KITTI 2, MVS-Synth, OmniObject3D, Spring, Synscapes, UrbanSyn, BEDLAM, etc., *and* the synthetic-data fraction is *higher* than DUSt3R's mixture), and the model *generalizes* to *real-world* clinical-IOS-like data (the chair-illusion demo proves *strong* out-of-distribution generalization). The 32-dataset mixture is the *most-extensive* data-augmentation strategy of any 2024-2025 3D-foundation-model paper, and the *empirical success* on the *broad* eval suite (Sintel + Bonn + KITTI + TUM-dynamics + 7-scenes + NRGBD + DAVIS + ScanNet) is the *direct* evidence that *synthetic-data bootstrap* works for 3D-reconstruction. **H5 takeaway:** for v0 dental-crown, the *synthetic* data sources (3DTeethSeg22 synthetic subset + ToSynFCD + TeethGenerator VQ-VAE synthetic data) are the *bootstrap* path to a *clinical* model, the *direct* analog of CUT3R's synthetic-heavy training mixture.

## Surprises / interesting things buried in the paper

1. **The chair-illusion demo is the *killer* Fig** (not in the main paper PDF, but on the project page) — shows the model's *prior* + *update* mechanism in action, the *direct* analog for the *missing-tooth* generation problem. The model predicts a 3D chair at frame 1, then *updates* to recognize it's a 2D plane with chair texture, and *maintains* that belief even when the final frame looks like the initial frame. **For dental:** the model should *predict* a 3D tooth at the edentulous site (correct prior), then *update* based on the *adjacent teeth* + *gum* (which suggest the tooth is *missing*), and *maintain* the missing-tooth hypothesis. The CUT3R state-update + state-readout mechanism is *exactly* this loop.

2. **The 32-dataset mixture is the *broadest* of any 3D-foundation-model** (vs DUSt3R's ~10 datasets, MonST3R 174's 4 datasets, Spann3R's ~10 datasets, MASt3R's ~10 datasets, VGGT's ~10 datasets) — this is the *secret sauce* that lets CUT3R handle *all three* input modalities (single image, video, photo collection) in *one* model. **For dental:** the *analog* would be a *broad* training mixture of (clinical IOS scans + 3DTeethSeg22 + ToSynFCD + TeethGenerator synthetic + per-tooth CAD) — the *broadest* possible training set to handle *all* input modalities (single-arch IOS, full-mouth IOS, video-IOS, photo-of-model).

3. **CUT3R is *metric-scale* out of the box** (no global alignment needed) — the *direct* advantage over MonST3R 174 which requires ~1 min global opt. **For dental:** metric-scale is *critical* for the *missing-tooth generation* problem (the *generated* crown must be at the *same metric scale* as the *prepared tooth*, otherwise it won't fit).

4. **The state-update + state-readout is *bidirectional*** (gradients flow through *both* directions) — the state is *trained* to be useful for prediction, not just a memory dump. This is *different* from Spann3R's *spatial memory* which is a *cache* of past features with no learnable interaction. **For dental:** the *bidirectional* design is *essential* for the *missing-tooth* problem (the state must *update* based on the *current image* AND *read* for the *next image* AND *read* for the *virtual-camera query*).

5. **The "revisiting" experiment** (Sec project page) shows the model can *re-process* the same images with the *final* state and get *better* results (the model gets *more accurate* when it has *full* context, vs *online* processing where the state only has *past* context). **For dental:** the *revisiting* mode could be used for *post-hoc refinement* of the *generated* crown (re-process the IOS + the generated crown with the *full* state, get a *more accurate* crown fit).

## Quote-worthy sentences

- "We present a unified framework capable of solving a broad range of 3D tasks. Our approach features a stateful recurrent model that continuously updates its state representation with each new observation." (abstract)
- "Not only can it predict accurate pointmaps from image observations, but it can also infer unseen regions of the scene by probing at virtual, unobserved views." (abstract, the *killer* feature for missing-tooth generation)
- "Our method is simple yet highly flexible, naturally accepting varying length of images that may be either video streams or unordered photo collections, containing both static and dynamic content." (abstract, the *broadest* input-modality support of any 3D-foundation-model)
- "Humans are online visual learners. We continuously process streams of visual input, building on what we have learned in the past while learning in the present." (Sec 1, the *biological* motivation)
- "Building on these insights, we introduce an online 3D perception framework that unifies three key capabilities: 1) reconstructing 3D scenes from few observations, 2) continuously refining the reconstruction with more observations, and 3) inferring 3D properties of unobserved scene regions." (Sec 1, the *3 key capabilities* that map directly to the dental project's *missing-tooth* problem)
- "Spann3R's memory serves primarily as a cache for observed scenes, our compressed state representation not only captures observed scene content but also enables inferring unobserved structures." (Sec 2, the *fundamental* difference vs Spann3R)
- "Our method captures rich prior knowledge (predicting the first frame as a 3D chair) but updates its understanding as more observations are processed, ultimately recognizing the scene as a plane. Notably, even when the final frame of the video closely resembles the initial one, the model maintains its belief that the scene is flat. This highlights the state's ability to effectively update based on additional observations." (Sec project page, the *chair-illusion* description, the *direct* analog for missing-tooth generation)

## Code/data link

- **Code:** https://github.com/CUT3R/CUT3R (official, open-source)
- **Pretrained:** Google Drive via gdown in README (cut3r_224_linear_4.pth intermediate, cut3r_512_dpt_4_64.pth final)
- **Project page:** https://cut3r.github.io/ (with interactive Viser viewer, DAVIS gallery, 3D gallery, photo-collection gallery, online-vs-revisiting comparison, chair-illusion demo)
- **Paper:** arXiv 2501.12387 v1, CVPR 2025 Oral
- **Dependencies:** PyTorch 2.x + CUDA 12.1 + Python 3.11 + CroCo v2 RoPE CUDA kernel + gsplat (for training logging) + evo (for evaluation) + open3d (for evaluation)
- **Datasets (32 training datasets):** see GitHub README for the full list with processing scripts (ARKitScenes, BlendedMVS, CO3Dv2, MegaDepth, ScanNet++, ScanNet, WayMo Open, WildRGB-D, Map-free, TartanAir, UnrealStereo4K, Virtual KITTI 2, 3D Ken Burns, BEDLAM, COP3D, DL3DV, Dynamic Replica, EDEN, Hypersim, IRS, Matterport3D, MVImgNet, MVS-Synth, OmniObject3D, PointOdyssey, RealEstate10K, SmartPortraits, Spring, Synscapes, UASOL, UrbanSyn, HOI4D)
- **Eval datasets (download scripts in GitHub):** 7-scenes, Bonn, KITTI, NYU-v2, ScanNet, Sintel, TUM-dynamics, Neural-RGBD
- **License:** ⚠️ NOT explicitly stated in README — needs LICENSE file check; the open-source release + Apache-style structure + permissive dependencies suggest MIT or Apache-2.0 (the *de facto* 2024-2025 standard for 3D-foundation-model releases); the *commercial-deployability* is *unknown* until LICENSE is verified

## For our project (concrete next steps for v0/v1 dental-crown-gen)

**v0 sub-task 1 (full-arch synthesis):**
- **CONSIDER** adopting CUT3R's *persistent state* + *virtual-camera query* as the *arch-context-encoder* for v0 — process the *full* IOS (or photo collection) into a *state*, then *query* the state with a *virtual camera* placed at the *edentulous site* to hallucinate the *missing tooth context*
- **BENEFIT:** the state is *metric-scale* (vs MonST3R 174's *global-alignment-required* design), so the *generated* crown is *naturally* at the *correct* scale
- **COST:** 1-2 weeks engineering to integrate CUT3R (CUT3R is *open-source*, easy to fork)
- **LICENSE:** ⚠️ CHECK CUT3R LICENSE before v0 integration — if MIT/Apache-2.0, *direct* integration is *fine*; if CC BY-NC-SA, *re-implement* the architecture from the paper (or use a permissively-licensed backbone)
- **H5 EVIDENCE:** CUT3R's *32-dataset synthetic-heavy training* is the *direct* analog for v0's *synthetic-data bootstrap* (3DTeethSeg22 + ToSynFCD + TeethGenerator synthetic), the *strongest* H5 evidence in the 3D-foundation-model arc

**v0 sub-task 2 (crown generation):**
- **CONSIDER** using CUT3R's *virtual-camera query* as the *conditioning mechanism* for the *missing-tooth* — query the state with a *virtual camera* at the edentulous site, get a *pointmap* of the *expected missing tooth*, use this as the *prior* for the DMC + MCAM + CPL + MRL stack (paper 033)
- **BENEFIT:** the *virtual-camera-query* output is *already* a *prior* for the missing tooth, the *direct* integration path
- **COST:** $100-200 Lambda, 1-2 weeks engineering, the *direct* analog of "histogram loss" (paper 061) but for *3D* pointmaps
- **H3 EVIDENCE:** CUT3R's *state-conditioned* design is the *richest* H3 mechanism in the 3D-foundation-model arc — the state *contains* opposing jaw + adjacent teeth + gum, the *crown generation* is *conditioned* on *all* of them

**v1 (streaming clinical-IOS reconstruction):**
- **ADOPT** CUT3R's *online / linear-memory* design as the v1 *streaming-IOS* encoder — the *linear* memory growth is the *killer* feature for long clinical-IOS video sequences (where DUSt3R's *quadratic* global alignment is infeasible)
- **BENEFIT:** the *state* persists across the *entire* IOS scan, so the *final* state encodes the *full* arch context
- **COST:** 4-6 weeks engineering + 1-2 days CUDA setup
- **H1 EVIDENCE:** CUT3R's *state-update + state-readout* is *internally 2-stage*, the *direct* analog of v0's *segmentation → generation* pipeline

**For v0 paper related-work:**
- **CITE** CUT3R as the *founding* persistent-state paper in the 3D-foundation-model arc (1 paragraph in related work, $0, 1-2 hours)
- **PORT** the *3D-foundation-model 2024-2025 arc* to v0 Table 2 (DUSt3R → MASt3R → MonST3R 174 → CUT3R → Spann3R → Easi3R 173 → VGGT → 3D-GS → pixelSplat 164 → ..., the *definitive* 3D-foundation-model timeline)
- **EMPHASIZE** the *virtual-camera query* as the *killer* feature for *missing-tooth generation* (1 sentence in abstract, 1 paragraph in intro, the *unique* contribution of the v0 paper)

**Open Q for HK (the 4 critical decisions):**
1. **(i) adopt CUT3R as v0 arch-context-encoder?** (PROBABLY YES, but CHECK LICENSE first)
2. **(ii) use CUT3R's *virtual-camera query* as the *missing-tooth prior*?** (YES, the *direct* analog of paper 061's *histogram loss* but for *3D* pointmaps)
3. **(iii) cite CUT3R in v0 paper?** (YES, the *founding* persistent-state paper)
4. **(iv) port the *3D-foundation-model 2024-2025 arc* to v0 Table 2?** (YES, the *definitive* timeline)

**★ ★ Next paper to read (176):** the 175-CUT3R-note's *direct* follow-up is **Easi3R (Chen 2025, ICCV 2025, the *MASt3R-fine-tune-for-disentangled-motion* paper that Easi3R 173 already established as the *strongest* baseline that *beats* CUT3R by 4.11% on Sintel)** — the *direct* extension of CUT3R + MonST3R 174 + DUSt3R that adds *explicit* disentangled motion. Alternative: **DAS3R (the *DPT-trained dynamic-mask* fine-tune on top of MonST3R 174, the *direct* extension that Easi3R 173 beats by +22.1 JM on DAVIS-17)**. Alternative: **Spann3R (Wang & Agapito 2024, the *spatial-memory* concurrent-to-CUT3R paper, the *direct* comparison baseline)**. Alternative: **G-CUT3R (Khafizov 2025, 2508.11379, the *guided* extension that adds depth/intrinsics/pose priors via ZeroConv)**. Alternative: **STream3R (Zhang 2025, 2508.10893, the *causal-attention* alternative to CUT3R's RNN-state, the *most-recent* 2025-08 4D-foundation-model paper)**. Alternative: **ZipMap (Jin 2026, CVPR 2026, the *test-time-training* alternative to CUT3R that achieves *linear* time + matches *quadratic* baselines like π3 + VGGT)**. Alternative: **TTT3R (the *test-time-training* recurrent 3D-reconstruction paper)**. Alternative: **4RC (Luo 2026, the *conditional-querying-anytime* paper that reports CUT3R 0.078/93.7 vs Spann3R 0.144/81.3 on ScanNet-like, the *definitive* CUT3R-vs-Spann3R comparison)**. Alternative: **AMB3R (Wang 2026, the *backend-augmented* feed-forward 3D-reconstruction paper that explicitly compares CUT3R + Spann3R + VGGT + π3)**. **Recommendation: *read 176 = Easi3R 173 (the *disentangled-motion* paper, the *most-comprehensive* 2025 dynamic-3D-foundation-model paper, the *direct* extension of MonST3R 174 + CUT3R, the *strongest* baseline that the v0 paper must beat for *dynamic* clinical-IOS scenarios)*** — the *right* next paper to *complete* the *MonST3R 174 → CUT3R 175 → Easi3R 173* arc, the *killer* empirical-evidence paper that *confirms* *disentangled motion > state-only* for *dynamic* scenes, the *direct* CUT3R-extension that the 175-note's H3 analysis would *benefit from*. ⚠️ NOTE TO SELF: scholar-summarize cron *should* *always* verify arXiv IDs via direct arXiv lookup — the 175-note's actual arXiv ID is 2501.12387 (NOT 2501.05087 or 2503.10345 as the 174-note predicted).
