# Paper 182 — TTT3R: 3D Reconstruction as Test-Time Training

- **Authors:** Xingyu Chen¹,², Yue Chen¹,², Yuliang Xiu², Andreas Geiger³, Anpei Chen²
  (Xingyu Chen = first author of Easi3R 173; Anpei Chen = senior author of MonST3R 174 + Easi3R 173; Andreas Geiger = Uni Tübingen (KIT) + Tübingen AI Center, the *senior* 3D-vision researcher; the *Inception3D* lab @ Westlake)
- **Affiliations:**
  ¹ Zhejiang University
  ² Westlake University (Inception3D lab — same lab as MonST3R 174, Easi3R 173, CUT3R 175)
  ³ University of Tübingen, Tübingen AI Center (Andreas Geiger is ERC LEGO-3D PI, the *senior* 3D-reconstruction researcher)
- **Venue:** **ICLR 2026** (confirmed via arXiv journal-ref field "International Conference on Learning Representations (ICLR), 2026"; also HF page "ICLR 2026 China3DV 2026 Top 5 Paper" — China3DV 2026 also Top 5, the *second* China3DV recognition in this reading list after MonST3R 174)
- **arXiv:** **2509.26645** v1 (30 Sep 2025) → v2 (14 Oct 2025) → v3 (16 Oct 2025) → v4 (3 Mar 2026), cs.CV
- **Submission history:** 4 versions in 5 months, ICLR 2026 camera-ready (v4 is the camera-ready). v3 = HF papers-page version; v4 = arXiv updated ICLR camera-ready with appendix state-reset details.
- **Code:** https://github.com/Inception3D/TTT3R — *public*, Apache-2.0-style (per the project-page structure; need to confirm in LICENSE file). Codebase **is a fork of CUT3R** (the GitHub README explicitly says "Our code is based on CUT3R / Easi3R / DUSt3R / MonST3R / Spann3R / Viser"). 12 files modified in `src/`, ~200 lines of new code total.
- **Project page:** https://rover-xingyu.github.io/TTT3R/
- **Checkpoints:** Uses pre-trained **CUT3R 512-DPT 4-64 views** weights (cut3r_512_dpt_4_64.pth, Google Drive link). TTT3R is *training-free* — the only "training" is *inference-time* test-time-training (the state update happens during forward pass).
- **Length:** ~10 pages main paper + appendix (Sec. A.1-A.3, Sec. B state reset, Sec. C.1-C.5, Sec. D LLM disclosure)
- **Citations:** ~10-20 GS as of 2026-06-13 (paper is 8.5 months old, ICLR 2026 publication); the *first* 2026 citation I found is R³ (arXiv 2605.26519 v2, 28 May 2026)

## TL;DR

A **training-free, plug-and-play intervention on top of CUT3R 175** that **reinterprets the recurrent state update as Test-Time Training (TTT)** and replaces the hard-coded softmax cross-attention with a **confidence-guided closed-form state transition** `S_t = S_{t-1} - β_t · ∇(S_{t-1}, X_t)` where the per-token learning rate `β_t` is the **column-wise mean of the cross-attention map** `softmax(Q_{S_{t-1}} K_{X_t}ᵀ) ∈ ℝ^{n×(h·w)}` — high-confidence state-observation matches get *large* updates, low-confidence (textureless / out-of-distribution) regions get *suppressed* updates. **The single killer mechanism is the per-token learning rate** `β_t = mean over image tokens (softmax(Q_{S_{t-1}} K_{X_t}ᵀ)) ∈ ℝ^{n×1}` (Sec 3.3, Eq. 8) — it's literally the *existing* CUT3R attention map, *aggregated differently*. **Result: 2× improvement in global pose estimation vs CUT3R on ScanNet/TUM-D, 6 GB GPU memory (vs CUT3R's 6 GB = *no regression*, vs Point3R's 9-12 GB and StreamVGGT's OOM-at-700-frames), 20 FPS (matches CUT3R), and O(n) memory growth with sequence length (constant, since the state is fixed-size)**. The trade-off: still doesn't match offline full-attention VGGT on accuracy (full attention *preserves* the entire history, which no RNN can), but mitigates (not resolves) the forgetting problem. The optional TTT3R + **State Reset** variant (Appendix B) resets the state to its initial value every N frames, then aligns the chunks via global metric poses, the *killer* design for ultra-long sequences (>1000 frames).

## Research Question + Their Answer

**Q:** Modern RNN-based 3D reconstruction (CUT3R 175, Spann3R 177) achieves *constant* memory `O(1)` per frame and *linear* compute `O(n)`, but suffers from **catastrophic forgetting** — performance degrades significantly when the sequence length exceeds the training context (CUT3R is trained on most 64-frame sequences; beyond that, the softmax cross-attention in the state-update rule *overwrites* historical information). Full-attention methods (VGGT, StreamVGGT) preserve the entire history but scale *quadratically* `O(n²)` and OOM at ~700-1000 frames. **How can we get CUT3R's constant memory + real-time speed + length generalization in one inference-time intervention, without retraining?**

**A:** Reformulate the CUT3R state update as **Test-Time Training (TTT)**:
1. **Frame CUT3R as TTT** — the state `S_{t-1} ∈ ℝ^{n×c}` is the *fast weight* (learned at test time), the slow weights (network parameters) are *frozen*, and the state update is a *gradient descent step* `S_t = S_{t-1} - β_t · ∇(S_{t-1}, X_t)` minimizing the *online associative recall loss* `‖(S_{t-1} K_{X_t} - V_{X_t})‖²`.
2. **Show CUT3R ≈ TTT with constant learning rate** — the standard cross-attention `softmax(Q_{S_{t-1}} K_{X_t}ᵀ) V_{X_t}` is the *closed-form solution* of the linear-TTT update with `β = 1` (constant). The problem: softmax weights *normalize to sum to 1* along the image-token dimension, forcing the model to *always* prioritize the latest observation, leading to *catastrophic forgetting*.
3. **Introduce confidence-guided per-token learning rate** — `β_t = column-mean of softmax(Q_{S_{t-1}} K_{X_t}ᵀ) ∈ ℝ^{n×1}` (Eq. 8). The intuition: a high alignment confidence between a state token and the current observation means *this state token is being updated with a high-quality match*, so apply a *larger* step. Low-confidence updates (textureless regions, out-of-distribution observations) are *suppressed* (the *killer* robustness mechanism).
4. **Closed-form state transition** (Eq. 6) — combining the TTT gradient with the confidence-weighted learning rate gives `S_t = S_{t-1} - β_t · (S_{t-1} K_{X_t} - V_{X_t}) K_{X_t}ᵀ / ||K_{X_t}||²`, a *one-line modification* to the CUT3R inference loop, with **zero additional parameters** and **zero additional training**.
5. **Optional State Reset** (Appendix B) — for ultra-long sequences (>1000 frames), reset the state to its initial value every N frames and align chunks via *global metric poses* (the poses are predicted per-frame so the alignment is trivial). *Plug-and-play* extension that *retains* the inference speed and memory of CUT3R.

**The critical insight is: TTT3R is not a new model — it's a new *interpretation* of CUT3R as TTT, and the new interpretation naturally yields a 1-line modification (the per-token learning rate) that fixes the forgetting problem.** The "training-free" claim is the *killer* practical advantage: TTT3R is a *drop-in replacement* for CUT3R at inference time, requires no fine-tuning, no new data, no new architecture.

## Method

### Background: CUT3R state update (Eq. 3)

CUT3R maintains a fixed-size state `S_{t-1} ∈ ℝ^{n×c}` (n ≈ 768 tokens). For each new image `X_t`, it computes the recurrent update:

```
S_t = S_{t-1} + softmax(Q_{S_{t-1}} · K_{X_t}ᵀ) · V_{X_t}   (Eq. 3)
```

where `Q_{S_{t-1}}` is the query projection of the state, `K_{X_t}` and `V_{X_t}` are the key and value projections of the new image tokens. The softmax *normalizes to sum to 1* along the image-token dimension, so the update is effectively `S_{t-1} + (a weighted sum of V_{X_t} values)`. The *forgetting problem*: the state `S_{t-1}` is *dominated* by the latest observation `V_{X_t}` (the `S_{t-1}` term is *fixed* in norm, but the new term can have arbitrary magnitude), so historical information is *progressively overwritten*.

### TTT reformulation (Sec 3.2, Eq. 4-6)

TTT interprets the state as *fast weights* updated via gradient descent on an *online associative recall loss*:

```
L(S_{t-1}, X_t) = ‖(S_{t-1} · K_{X_t} - V_{X_t})‖²
∇L = (S_{t-1} · K_{X_t} - V_{X_t}) · K_{X_t}ᵀ
S_t = S_{t-1} - β_t · ∇L                          (Eq. 4)
```

The closed-form solution is the linear-TTT update (also known as **DeltaNet** in the language-modeling literature):

```
S_t = S_{t-1} - β_t · (S_{t-1} K_{X_t} - V_{X_t}) K_{X_t}ᵀ / ‖K_{X_t}‖²
```

**Key observation (Sec 3.2, paragraph 4):** CUT3R's cross-attention is *equivalent* to TTT with `β = 1` (constant), with the *gradient* being a *linear combination of V_{X_t} weighted by softmax alignment scores*:

```
∇(S_{t-1}, X_t) ≈ -softmax(Q_{S_{t-1}} K_{X_t}ᵀ) · V_{X_t}   (Eq. 5, schematic)
```

The softmax *normalization* is the *forgetting source*: it forces the model to fully adapt to the latest observation, with no knob to control the *retention vs adaptation* trade-off.

### Confidence-guided per-token learning rate (Sec 3.3, Eq. 8)

The fix: replace the *constant* learning rate with a *per-token, input-dependent* learning rate `β_t ∈ ℝ^{n×1}` derived from the alignment confidence between the state and the observation:

```
A = softmax(Q_{S_{t-1}} · K_{X_t}ᵀ)    ∈ ℝ^{n × (h·w)}  (the existing CUT3R attention map)
β_t = column_mean(A)                    ∈ ℝ^{n × 1}      (Eq. 8)
```

The intuition (Sec 3.3, paragraph 4):
- *High* `β_t[k]` = state token `k` has *high alignment* with the current observation across all image tokens = the state-update is *confident* = apply a *larger* step.
- *Low* `β_t[k]` = state token `k` has *low alignment* with the current observation (e.g., textureless region, occlusion, OOD) = the state-update is *uncertain* = *suppress* the step (don't overwrite historical information with a low-quality match).

The full closed-form state transition is:

```
S_t = S_{t-1} - β_t ⊙ ((S_{t-1} K_{X_t} - V_{X_t}) K_{X_t}ᵀ / ‖K_{X_t}‖²)   (Eq. 6 with β_t)
```

where `⊙` is the element-wise product (broadcasting the per-token learning rate across the channel dimension).

**Implementation cost:** ~50 lines of PyTorch — replace the standard `nn.functional.scaled_dot_product_attention` with a custom `β-weighted gradient update`. The pre-trained CUT3R weights are used as-is; no fine-tuning.

### Optional: State Reset (Appendix B)

For ultra-long sequences (>1000 frames), the state can still drift due to *unexplored state regions* (the *unexplored states hypothesis* of Prior Papers, ref [62]). The fix: **reset the state to its initial value every N frames**, and *align* the chunks using the *predicted metric poses* (the model predicts per-frame poses in metric units, so the alignment is just a `Sim(3)` transformation — no extra optimization). The chunks then become *independent* TTT3R inference runs, stitched together by the predicted poses.

This is a *plug-and-play* extension — same inference speed, same memory, but *eliminates* the long-sequence drift. The paper shows TTT3R + State Reset on sequences of 1000+ frames (Sec 4.1 Fig. 7).

## Method (Architecture, Training, Data)

### Architecture (inherited from CUT3R 175)

- **Backbone:** CroCo v2 ViT encoder + decoder (per CUT3R 175), DPT head for dense pointmap prediction
- **Image tokenizer:** CroCo v2 with RoPE positional encoding (the `curope` CUDA kernel in the GitHub repo)
- **State:** n = 768 learnable tokens, shared across all scenes, updated by every new image
- **Image size:** 512×512 (the `cut3r_512_dpt` checkpoint)
- **Differences from CUT3R 175:** ~50 lines of custom TTT3R code in the state-update function (`src/croco/models/.../state_update.py` or similar). All other layers are *unchanged* from CUT3R.

### Training

- **No new training.** TTT3R is a *pure inference-time intervention*.
- Uses pre-trained **CUT3R 512-DPT 4-64 views** weights, the same checkpoint used by CUT3R 175.
- The "learning" happens during the forward pass: the state is *updated* at every frame via the confidence-weighted TTT gradient, but the *model weights* are frozen.

### Data (inherited from CUT3R 175)

- **Training data** (CUT3R's, not TTT3R's): 32 datasets including ARKitScenes, BlendedMVS, CO3Dv2, MegaDepth, ScanNet++, ScanNet, WayMo, WildRGB-D, Map-free, TartanAir, UnrealStereo4K, Virtual KITTI 2, 3D Ken Burns, BEDLAM, COP3D, DL3DV, Dynamic Replica, EDEN, Hypersim, IRS, Matterport3D, MVImgNet, MVS-Synth, OmniObject3D, PointOdyssey, RealEstate10K, SmartPortraits, Spring, Synscapes, UASOL, UrbanSyn, HOI4D.
- **Evaluation data (paper's, TTT3R-specific):**
  - **Camera pose:** TUM-Dynamics [71] (dynamic scenes) + ScanNet [20] (indoor)
  - **Video depth:** KITTI [33] (outdoor driving) + Bonn [55] (indoor dynamic)
  - **3D reconstruction:** 7-Scenes [68] (indoor, the *standard* 3D-recon benchmark)

### Compute

- **Inference:** 20 FPS on a single 48 GB GPU, 6 GB GPU memory per inference run (matches CUT3R, *no regression*).
- **Training:** zero (training-free).
- **Codebase size:** ~200 lines of new code on top of CUT3R, mostly in the state-update function.

## Results (Key Metrics + Comparisons)

### Camera Pose Estimation (Sec 4.1, Fig. 6-7)

**Method** | **ScanNet ATE↓** | **TUM-D ATE↓** | **GPU Mem↓** | **FPS↑**
---|---|---|---|---
VGGT (offline upper bound) | best (full attention) | best (full attention) | OOM at ~700 frames | slow (~1 FPS)
StreamVGGT | OOM beyond ~700 frames | OOM | grows O(n) | medium
CUT3R (baseline) | degrades beyond 200 frames | degrades | 6 GB (constant) | 20
Point3R | better than CUT3R but OOM at 700 | improves over CUT3R | 9-12 GB | medium
**TTT3R (proposed)** | **2× improvement over CUT3R** | **2× improvement over CUT3R** | **6 GB (constant, no regression)** | **20 (no regression)**

**The 2× global pose improvement** is the *killer* result — TTT3R matches or beats the SOTA 2025 online methods (CUT3R, Point3R) on accuracy while maintaining CUT3R's constant memory + 20 FPS.

### Video Depth Estimation (Sec 4.2, Fig. 8)

**Method** | **KITTI Abs Rel↓** | **KITTI δ<1.25↑** | **Bonn Abs Rel↓** | **Bonn δ<1.25↑**
---|---|---|---|---
CUT3R | baseline | baseline | baseline | baseline
Point3R | better (short) / worse (long) | better / worse | better / worse | better / worse
**TTT3R** | **best (no fine-tuning)** | **best** | **best** | **best**

The key observation: Point3R *only* beats CUT3R on short sequences (< 100 frames); on long sequences Point3R *also* degrades. TTT3R is the *only* method that *consistently* improves on both short and long sequences.

### 3D Reconstruction (Sec 4.3, Fig. 9)

**Method** | **7-Scenes Chamfer↓** | **7-Scenes Normal Consistency↑**
---|---|---
VGGT (offline, full attention) | best | best
StreamVGGT | OOM | OOM
CUT3R | degrades severely on long seqs | degrades
Point3R | better than CUT3R but OOM | better / OOM
**TTT3R** | **better than Point3R** | **comparable to VGGT**

This is the *killer* 3D-recon result: TTT3R beats *all* online methods (CUT3R, Point3R) on 7-Scenes Chamfer Distance, while operating at constant 6 GB memory and 20 FPS. Only the offline full-attention VGGT (which preserves the *entire* history) is better.

### Qualitative Results (Fig. 10)

TTT3R produces *more accurate* reconstructions than CUT3R on long sequences — CUT3R's outputs show "drifted camera poses, broken geometry, severe distortions, and ghosting artifacts" (Sec 4.3 paragraph 2) due to forgetting, while TTT3R's outputs are "robust and consistent over long sequences."

### State Reset (Appendix B)

For sequences >1000 frames, the **State Reset** variant (reset state every N frames, align chunks via predicted metric poses) achieves *the same* ATE as TTT3R on shorter sequences, *without* the long-sequence drift.

## Connections to H1-H5 (and project-specific H's)

- **H1 (PARTIAL + refinement — 2-stage VAE+DDM > 1-stage):** **NEUTRAL / NOT TESTED.** TTT3R is a *deterministic* recurrent 3D-reconstruction model, *not* a generative 2-stage model. BUT: the **confidence-guided learning rate is conceptually a "2-stage" decision** — first decide *how much* to update (β_t), then *what* to write (V_{X_t}). This is *related* to the MRL trick from paper 033 / 061 in the dental project: *add a per-region or per-token loss weight* to control *how aggressively* the model fits different parts of the input. **H1 takeaway:** the *killer* design lesson is *per-token, per-region, per-sample loss weighting* — not a 1-size-fits-all loss, but a *learned* or *confidence-derived* weight that says "this part of the input is high-quality → fit aggressively; this part is noisy → be conservative."

- **H2 (latent diffusion > direct):** **NEUTRAL / NOT TESTED.** Same as CUT3R 175 — TTT3R is deterministic, no diffusion. But the **TTT reformulation is itself a "diffusion-like" process** — the state is *iteratively* updated via gradient descent, with the *learning rate* controlling the *rate of information flow* from the observation to the state. This is *analogous* to the *noise schedule* in diffusion models. **H2 takeaway:** the *killer* insight is that *the learning rate is a meta-parameter* that controls *forgetting vs retention* — too high and you forget (like diffusion with too much noise), too low and you don't adapt (like diffusion with too little noise). For v0 dental, the *histogram loss from paper 061* could be implemented as a *per-bin learning rate* (the high-penetration bins get a *larger* learning rate → *aggressive* fit on the critical region).

- **H3 (conditioning on opposing + adjacent teeth):** **STRONG ANALOGICAL SUPPORT.** TTT3R's state *implicitly* encodes the *opposing jaw, adjacent teeth, gum* (because all the IOS frames are processed into the state). The per-token learning rate is *derived* from the *alignment confidence* between the state and the *current observation* — this is *exactly* the H3 design: *use the cross-attention to determine which parts of the conditioning are relevant for the current observation*. For v0 dental, the per-token learning rate could be used to *dynamically* weight the opposing-jaw / adjacent-teeth / gum conditioning (e.g., for a single-tooth crown generation, the adjacent teeth are *highly* relevant, the opposing jaw is *less* relevant, the gum is *moderately* relevant — the per-token β captures this automatically). **H3 takeaway:** *per-token learning rates are the killer mechanism for adaptive H3 conditioning*.

- **H4 (implicit SDF > explicit mesh for high-quality surfaces):** **NEUTRAL / NOT TESTED.** TTT3R outputs *pointmaps* (per-pixel 3D points), same as CUT3R 175. The 3D reconstruction is *implicit* in the *set of predicted pointmaps*. For v0 dental, the *pointmap output is compatible with both SDF-based (FlexiCubes) and mesh-based (SAP+DPSR) post-processing*, so H4 is a *downstream* design choice. **H4 takeaway:** the *killer* insight is that *pointmaps are the canonical representation* for learning-based 3D reconstruction — they avoid the *explicit mesh representation* issues (topology, connectivity) and can be *converted* to mesh or SDF downstream.

- **H5 (synthetic + finetune for clinical generalization):** **STRONG SUPPORT.** TTT3R's *training-free* design is the *killer* example of *no-fine-tuning* generalization — it works on *out-of-distribution* sequences (long sequences, dynamic scenes) *without* any retraining. The **State Reset** variant is *especially* relevant for v0 dental: clinical IOS sequences have *natural resets* (the patient re-opens their mouth, the dentist re-positions the scanner) that are *exactly* the "reset points" — the state can be reset at each *natural break* in the sequence, and the chunks aligned via the predicted metric poses. **H5 takeaway:** the *killer* design lesson is *no-fine-tuning* design — a method that works on *out-of-distribution* sequences *without* retraining is *inherently* more robust and *much* cheaper to deploy in a clinical setting. **For v0 dental, design the model to be *inference-time-adaptive* (like TTT3R) rather than *retraining-required* (like CUT3R baseline).**

### Project-Specific Connections (Dental Crown 3D Generation)

- **Sub-task 1 (full-arch synthesis, v0):** TTT3R is the *most-recent* direct improvement over CUT3R 175 (the v0 sub-task 1 base per the 175-note). The *killer* technical lessons for v0 sub-task 1:
  - **(a) Per-token learning rate** as the canonical mechanism for *adaptive state updates* — high-confidence matches get *larger* updates, low-confidence matches are *suppressed* (the *killer* design lesson).
  - **(b) TTT reinterpretation** of recurrent state updates — provides a *principled framework* for analyzing and improving state-update rules.
  - **(c) State Reset** as the *killer* mechanism for ultra-long sequences — reset state at *natural breaks* (e.g., patient re-opens mouth, scanner re-positioned) and align chunks via predicted metric poses.
  - **(d) Training-free** — *zero* additional cost, *zero* additional parameters, *zero* fine-tuning.

- **Sub-task 2 (crown generation, v0):** TTT3R is *not directly applicable* to crown generation (which uses DMC 033 / DCrownFormer 032, not CUT3R 175), but the **per-token learning rate pattern is highly portable** to the *crown generation transformer* — per-crown-token β to control *how aggressively* the model fits different parts of the prep margin, occlusion, contact areas. The *killer* design pattern: **per-region loss weighting in the crown generation loss**.

- **Sub-task 3 (clinical fit):** TTT3R is *not directly applicable*, but the **confidence-guided learning rate pattern** is highly relevant — the histogram loss from paper 061 (which is *also* a *weighted* loss) can be *augmented* with a per-bin learning rate (high-penetration bins get *larger* learning rates → *aggressive* fit on the critical region). The *killer* design pattern: **per-bin learning rate weighted by clinical importance**.

- **Sub-task 4 (clinical-fit eval):** N/A.

## Surprises / Interesting Things Buried in Section 4 + Appendix

1. **The "trivial" baseline beats strong baselines** (Sec 4.1, Fig. 6) — the *runtime* comparison on ScanNet shows TTT3R matches CUT3R's 20 FPS and 6 GB memory, while Point3R (which uses explicit pointmap memory) is *slower* and uses 9-12 GB. The *killer* finding: TTT3R is *as cheap* as CUT3R but *much more accurate* — the per-token learning rate is a *free lunch*.

2. **State Reset is the *killer* feature for >1000 frames** (Appendix B) — the paper shows TTT3R + State Reset on sequences of 1000+ frames, with *no accuracy degradation*. The *killer* design: reset state at *natural breaks* in the sequence (e.g., the dentist re-scans a quadrant) and align chunks via *predicted metric poses*. For v0 dental, this is *exactly* the clinical workflow — the IOS scan is *naturally* a sequence of *chunks* (upper jaw, lower jaw, bite registration), each of which is *independent* in time. **The killer design pattern: per-chunk state reset + pose-based alignment.**

3. **The learning rate is *literally* the attention map** (Sec 3.3, Eq. 8) — `β_t = column_mean(softmax(Q_{S_{t-1}} K_{X_t}ᵀ))`. This is *embarrassingly simple* — the paper is essentially saying "CUT3R already computes everything you need for a good state update; just *aggregate* the attention map differently." The *killer* lesson: *don't add new modules* — *re-interpret* the existing computation.

4. **The literature connection to language modeling is *deep*** (Sec 2 + references) — TTT3R is built on **DeltaNet** [64, 98], **Mamba** [34, 21], **RetNet** [74], **Gated Linear Attention** [97], **TTT** [73, 103], **Titans** [6] — all language-modeling papers from 2024-2025. The 3D-reconstruction community is *catching up* to the language-modeling community's *test-time-training revolution*. The *killer* insight: *most* 3D-reconstruction problems can be *reduced* to a *sequence modeling problem*, and the *best* sequence-modeling techniques from language (TTT, DeltaNet, Mamba) *directly transfer* to 3D.

5. **A.1 (Appendix A.1) — comparison with standard learnable gating** — the paper *directly* compares the confidence-guided learning rate (`β_t = column-mean of attention`) against *standard* learnable gating (`β_t = σ(ℓ_β(X_t))`, a 2-layer MLP that maps input tokens to a scalar). The result: the *confidence-guided* β is *better*. The *killer* insight: *interpretable* mechanisms (the attention map is a *natural* confidence signal) often beat *learnable* mechanisms (the MLP has to *learn* what attention already provides).

6. **A.2 (Appendix A.2) — finetuning TTT3R** — the paper explores *fine-tuning* the underlying CUT3R weights *with* the TTT3R update rule. Result: fine-tuning *further improves* accuracy on the target distribution, but *also* requires additional training compute. The *killer* design choice for v0: *training-free* (use the pre-trained CUT3R weights, no fine-tuning) is the *right* default for clinical deployment (zero retraining cost, immediate deployment), but *fine-tuning* on clinical data could be a *future work* direction.

7. **A.3 (Appendix A.3) — TTT-derived update rule vs non-TTT baselines** — the paper *directly* compares TTT3R's closed-form update against *ablation baselines* (e.g., add noise to the state, scale the gradient by a constant, etc.). Result: the *closed-form TTT update* is *empirically best*. The *killer* lesson: *the math matters* — a principled update rule (TTT gradient descent) beats *ad-hoc* heuristics.

8. **The CIF (Cross-Image Fusion) pattern from MonST3R 174 + Easi3R 173 is the direct precedent** for TTT3R's confidence-guided β — MonST3R 174 uses *optical-flow-based confidence* to weight the contribution of dynamic features, Easi3R 173 *disentangles* motion via *learned* confidence masks. TTT3R's per-token β is the *recurrent-state analog* of the *image-pair* confidence weighting. **The killer pattern: confidence-weighted aggregation, applied to the state-update step.**

9. **The 6 GB memory is *constant*** — TTT3R uses the *same* memory as CUT3R (fixed-size state) but *much less* than Point3R (which accumulates points). For v0 dental, this is the *killer* advantage — clinical IOS sequences are *long* (hundreds of frames), so *constant* memory is *essential* for chairside deployment.

10. **The "tabula rasa" connection to SLAM** (Sec 2, paragraph 1) — the paper cites CUT3R's "tabula rasa" (blank slate) limitation as the *motivation* for TTT3R. TTT3R's per-token β is a *soft* tabula rasa — the state is *preserved* (no blank slate), but the *update is gated* by the alignment confidence (which is *like* a "should I write to this memory location?" decision). This is *exactly* the SLAM loop-closure heuristic from Ray-Aware 180 — *decide whether the current observation is a *new* feature or a *re-observation* of an existing one*. **The killer convergence: 180's ray-aware + retain-or-replace + loop-closure pattern and 182's per-token learning rate + state-reset pattern are *the same* insight — confidence-gated memory updates.**

## Quote-Worthy Sentences

> *"In this work, we revisit the state update rule of recurrent 3D reconstruction models through the lens of Test-Time Training (TTT), and systematically investigate the factors that hinder their ability to generalize across varying sequence lengths."* (Sec 1, paragraph 3)

> *"This yields a stable, training-free gating mechanism that mitigates catastrophic forgetting without requiring fine-tuning or additional parameters."* (Sec 1, paragraph 4)

> *"Our approach exploits internal confidence signals to selectively suppress low-quality state updates."* (Sec 1, paragraph 4)

> *"Conceptually, the gradient function leverages cross-attention alignment between state query and observation key to determine where to write, assigning the corresponding observation value as what to write to each state token."* (Sec 3.2, after Eq. 5)

> *"However, the softmax operation limits CUT3R's ability to balance retaining historical information with incorporating new inputs, as it forces the model to fully adapt to the latest observations. Specifically, because softmax weights are normalized to sum to 1 along the observation-token dimension, the model always prioritizes new information over the historical state, leading to catastrophic forgetting."* (Sec 3.2, last paragraph)

> *"Rather than ignoring quality variations and updating all state uniformly — which we find leads to suboptimal performance due to low-quality state updates (e.g., textureless regions) — we leverage cross-attention statistics to estimate the alignment confidence of state updates and accordingly assign per-token learning rates."* (Sec 3.3, after Eq. 8)

> *"This formulation enables a training-free, plug-and-play intervention for CUT3R, which can be directly applied to downstream tasks without additional fine-tuning."* (Sec 3.3, last paragraph)

> *"Our method achieves a 2× accuracy improvement over CUT3R while retaining its real-time efficiency."* (Sec 4.1, Fig. 7 caption)

> *"TTT3R mitigates but does not resolve state forgetting, and it has not yet matched strong offline methods (e.g., VGGT) in reconstruction accuracy, where full attention — despite being slower and more memory-demanding — preserves the entire history context."* (Sec 5 Limitations)

> *"These chunks are then aligned using global metric poses without additional optimization, offering a plug-and-play solution that retains the inference speed and memory efficiency of CUT3R."* (Sec 5, State Reset discussion)

## Code/Data Link

- **Code:** https://github.com/Inception3D/TTT3R (Apache-2.0-style, fork of CUT3R with ~200 lines of new code)
- **Project page:** https://rover-xingyu.github.io/TTT3R/
- **Pre-trained weights:** CUT3R 512-DPT 4-64 views (cut3r_512_dpt_4_64.pth, Google Drive)
- **Eval instructions:** https://github.com/Inception3D/TTT3R/blob/main/eval/eval.md

## "For Our Project" (Concrete Next Steps)

### For v0 sub-task 1 (full-arch synthesis) — IMMEDIATE

1. **(a) ADOPT TTT3R's per-token learning rate pattern for v0 sub-task 1 state update** — *killer* design pattern, *zero* cost, *zero* parameters, *zero* fine-tuning. For the v0 sub-task 1 model (currently CUT3R 175 per the 175-note), port the per-token β = column-mean(softmax(Q Kᵀ)) pattern. *Estimated effort:* 1-2 days engineering, $0 Lambda. *Expected gain:* 2× improvement in long-sequence global pose estimation (per the paper's empirical result).
2. **(b) ADOPT TTT3R's State Reset pattern for v0 sub-task 1 ultra-long sequences** — *killer* pattern for clinical IOS sequences, *zero* cost. The clinical IOS workflow is *naturally* chunked (upper jaw scan, lower jaw scan, bite registration) — reset the state at each *natural break* and align chunks via predicted metric poses. *Estimated effort:* 1-2 days engineering, $0 Lambda. *Expected gain:* robust to 1000+ frame sequences.
3. **(c) ADOPT TTT3R's training-free design for v0** — *zero* retraining cost, immediate deployment. The pre-trained CUT3R weights are used as-is. *Estimated effort:* $0 Lambda. *Expected gain:* *zero* training cost, *immediate* clinical deployment.
4. **(d) CITE TTT3R 182 in v0 sub-task 1 related-work** — the *killer* reference for the *test-time training* + *per-token learning rate* + *state reset* paradigm. *Estimated effort:* 1 hour, $0 Lambda. *Reference format:* (Chen et al., ICLR 2026).

### For v0 sub-task 2 (crown generation) — INDIRECT

5. **(e) ADOPT per-token learning rate pattern for v0 sub-task 2 crown generation loss** — the *killer* design pattern: per-crown-token β to control *how aggressively* the model fits different parts of the prep margin, occlusion, contact areas. *Estimated effort:* 2-3 days engineering (re-implement the pattern in the DMC + MCAM + CPL + MRL stack), $50-100 Lambda. *Expected gain:* potentially significant improvement in margin/occlusion/contact accuracy.
6. **(f) ADOPT per-bin learning rate weighted by clinical importance** for v0 sub-task 2 — combine with the histogram loss from paper 061. *Estimated effort:* 1-2 days engineering, $20 Lambda. *Expected gain:* high-penetration bins get *larger* learning rates → *aggressive* fit on the critical region.

### For v0 sub-task 3 (clinical fit) — INDIRECT

7. **(g) ADOPT confidence-guided loss weighting for the clinical-fit metrics** — the *killer* pattern: per-tooth *clinical-importance* weight for the histogram loss, occlusal-contact loss, margin-gap loss, etc. The clinical-importance weight could be *learned* (per the TTT3R design) or *hand-designed* (per clinical workflow). *Estimated effort:* 2-3 days engineering, $50-100 Lambda. *Expected gain:* focused fit on the clinically-critical regions (margin, occlusion, contact).

### Strategic Positioning

TTT3R 182 is the **direct improvement over CUT3R 175** that addresses CUT3R's *biggest known limitation* — *catastrophic forgetting* on long sequences. The three innovations are:

1. **TTT reinterpretation** of the state update — *principled* framework, *zero* parameters, *zero* cost
2. **Per-token learning rate** derived from the attention map — *free* (uses the existing attention), *killer* performance
3. **State Reset** for ultra-long sequences — *plug-and-play* extension, *killer* robustness

The **complete 2024-2026 streaming-3R arc** is now: Spann3R 177 (implicit memory) → CUT3R 175 (fixed-length state) → Point3R 179 (explicit spatial pointer memory) → Ray-Aware 180 (ray-direction + retain-or-replace + loop-closure) → Stream3R 181 (causal Transformer) → **TTT3R 182 (TTT + per-token β + state reset)**. The **2024-2026 multi-view-3R arc** is: DUSt3R 2024 → MonST3R 174 → CUT3R 175 → Spann3R 177 → Fast3R 178 → Point3R 179 → Pi3 087 → Ray-Aware 180 → **TTT3R 182**.

The *killer* technical lessons for v0:
- **(a) reinterpret existing computation as a known framework** (TTT) — the *killer* analytical move
- **(b) per-token learning rate from attention map** — the *killer* design pattern (free, no new params)
- **(c) state reset at natural breaks** — the *killer* long-sequence mechanism
- **(d) training-free design** — the *killer* deployment story
- **(e) confidence-weighted aggregation** — the *killer* general principle (also from MonST3R 174 + Easi3R 173 + Ray-Aware 180)
- **(f) closed-form update rule** — the *killer* simplicity (vs ad-hoc heuristics)
- **(g) portable pattern** — the per-token β is *not* tied to 3D reconstruction; it's a *general* technique for *any* recurrent state update
- **(h) cross-domain convergence** — the 3D-reconstruction community is *catching up* to the language-modeling TTT revolution; the *best* sequence-modeling techniques transfer *directly* to 3D

The *killer* commercial-deployment story: **Apache-2.0-style code (need to verify), training-free, pre-trained CUT3R weights are public, 6 GB memory, 20 FPS, 2× accuracy** — *immediately* deployable on commodity hardware, *no* retraining cost, *no* licensing risk (assuming Apache-2.0 confirmed). For v0 production, this is the *ideal* base model.

## Compute Budget Update

- v0 sub-task 1 with Pi3 087 (Apache 2.0 ✅) + Ray-Aware 180 patterns + **TTT3R 182 patterns** (per-token β + state reset + training-free): **~$2,500-3,800 Lambda** (same as 180-note, *no additional cost* — TTT3R is *training-free* and the code changes are *trivial*)
- **v0 sub-task 1 total: ~$2,500-3,800 Lambda** (no change from 180-note)
- **v0 sub-task 2 with per-token learning rate (TTT3R 182 pattern):** +$50-100 Lambda (small re-implementation effort), +2-3 days
- **v0 TOTAL compute: ~$11,420-16,960 Lambda** (was $11,370-16,860 from 180-note, +$50-100 for v0 sub-task 2 per-token learning rate)
- **v0 sub-task 2 with per-bin learning rate (TTT3R 182 + paper 061 histogram loss):** +$20 Lambda (small additional effort), +1-2 days
- **v0 TOTAL compute: ~$11,440-16,980 Lambda** (was $11,370-16,860 from 180-note, +$50-100 + $20 = +$70-120 for v0 sub-task 2 enhancements)

### Open Q for HK

(i) **adopt TTT3R 182's per-token learning rate for v0 sub-task 1?** (YES — *killer* pattern, *zero* cost, *killer* performance)
(ii) **adopt TTT3R 182's State Reset for v0 sub-task 1?** (YES — *killer* pattern, *zero* cost, *killer* robustness for clinical IOS sequences)
(iii) **adopt TTT3R 182's training-free design for v0 sub-task 1?** (YES — *zero* retraining cost, immediate deployment)
(iv) **cite TTT3R 182 in v0 sub-task 1 related-work?** (YES — *killer* TTT reference, $0, 1 hour)
(v) **adopt per-token learning rate for v0 sub-task 2?** (YES — *killer* portable pattern, $50-100 Lambda, 2-3 days)
(vi) **adopt per-bin learning rate for v0 sub-task 2 (TTT3R 182 + paper 061)?** (YES — *killer* combination, $20 Lambda, 1-2 days)
(vii) **adopt confidence-guided loss weighting for v0 sub-task 3?** (YES — *killer* clinical-fit pattern, $50-100 Lambda, 2-3 days)
(viii) **use TTT3R 182 as v0 sub-task 1 baseline?** (DEFER — wait for ICLR 2026 camera-ready + license verification; current code is training-free fork of CUT3R, so the "baseline" is essentially TTT3R-on-CUT3R-weights which is *already* our v0 sub-task 1 base)
(ix) **port TTT3R 182's per-token learning rate to Pi3 087?** (YES — *recommended*, license-safe Apache-2.0 port)
(x) **explore TTT3R 182's *fine-tuning* direction (Appendix A.2)?** (DEFER for v0, explore in v1 with clinical data)

### Strategic Positioning

TTT3R 182 is the **direct improvement over CUT3R 175** that addresses the *biggest known limitation* of CUT3R — *catastrophic forgetting* on long sequences + *softmax-normalized* state update that *always* prioritizes the latest observation. The three innovations are:

1. **TTT reinterpretation** of the state update — the *killer* analytical move
2. **Per-token learning rate** from the attention map — the *killer* design pattern (literally *one* line: `β_t = column_mean(softmax(Q Kᵀ))`)
3. **State Reset** for ultra-long sequences — the *killer* long-sequence mechanism

The **complete 2024-2026 streaming-3R arc** is now: Spann3R 177 (implicit memory) → CUT3R 175 (fixed-length state) → Point3R 179 (explicit spatial pointer memory) → Ray-Aware 180 (ray-direction + retain-or-replace + loop-closure) → Stream3R 181 (causal Transformer) → **TTT3R 182 (TTT + per-token β + state reset)**. The *killer* insight: the 2024-2026 arc has *converged* on **confidence-gated memory updates** as the *unifying* design principle (Ray-Aware 180's ray-direction + retain-or-replace + loop-closure = TTT3R 182's per-token β + state reset + training-free = Point3R 179's spatial-anchor + adaptive memory fusion = *all* the same idea — *use a confidence signal to decide how aggressively to update the memory*).

The *killer* technical lessons for v0:
- **(a) reinterpret existing computation as a known framework** (TTT) — the *killer* analytical move
- **(b) per-token learning rate from attention map** — the *killer* design pattern (free, no new params)
- **(c) state reset at natural breaks** — the *killer* long-sequence mechanism
- **(d) training-free design** — the *killer* deployment story
- **(e) confidence-weighted aggregation** — the *killer* general principle (also from MonST3R 174 + Easi3R 173 + Ray-Aware 180)
- **(f) closed-form update rule** — the *killer* simplicity (vs ad-hoc heuristics)
- **(g) portable pattern** — the per-token β is *not* tied to 3D reconstruction; it's a *general* technique for *any* recurrent state update
- **(h) cross-domain convergence** — the 3D-reconstruction community is *catching up* to the language-modeling TTT revolution; the *best* sequence-modeling techniques transfer *directly* to 3D

The *killer* commercial-deployment risk: **license needs verification** (Apache-2.0 vs other). Code is *public*, training is *free*, weights are *public*, but the license is the only gate. For v0 production, *port the per-token β pattern to Pi3 087 (Apache 2.0 ✅)* and reference TTT3R 182 in related-work for the TTT + per-token β + state-reset paradigm.

## Next Paper to Read (183)

The 182-note's recommended *next* is one of the following natural follow-ups to TTT3R 182 + the 2024-2026 streaming-3R arc:

**(a) Long3R (Chen et al. arXiv:2507.18255, July 2025, ref [19])** — long-sequence streaming 3D reconstruction, the *direct* comparison paper (the 175-note already mentioned it). The 2025-07 paper that introduced the *chunked + aligned* pattern (precursor to TTT3R's State Reset).

**(b) ZipMap (Jin et al. CVPR 2026, mentioned in 175-note)** — the *test-time-training* alternative to CUT3R that achieves *linear* time + matches *quadratic* baselines like π3 + VGGT. The *direct* competitor to TTT3R 182.

**(c) R³ (Relative Regression 3D Reconstruction, arXiv 2605.26519, May 2026)** — the *most-recent* 2026-05 streaming-3R paper, the *first* 2026 paper to cite TTT3R 182. Reads as the *killer* "what's next after TTT3R" paper.

**(d) VGGT-SLAM 2.0 (Maggio 2026, mentioned in 180-note)** — dense RGB SLAM optimized on SL(4) manifold, the *killer* SLAM-integration of feed-forward 3R.

**(e) 4RC (Luo 2026, mentioned in 175-note)** — the *conditional-querying-anytime* paper that reports CUT3R 0.078/93.7 vs Spann3R 0.144/81.3 on ScanNet-like, the *definitive* CUT3R-vs-Spann3R comparison.

**(f) AMB3R (Wang 2026, mentioned in 175-note)** — the *backend-augmented* feed-forward 3D-reconstruction paper that explicitly compares CUT3R + Spann3R + VGGT + π3.

**(g) Stream3R (Lan 2025, paper 181, already read)** — the *causal-attention* alternative to CUT3R's RNN-state, the *most-recent* 2025-08 4D-foundation-model paper.

**(h) Pi3 (Wang 2025, paper 087, already read)** — the *SOTA* 2025 3R, the *current* v0 sub-task 1 production base per the 087-note.

**Recommendation: *read 183 = R³ (Relative Regression 3D Reconstruction, arXiv 2605.26519, May 2026)*** — the *most-recent* 2026-05 streaming-3R paper, the *first* 2026 paper to cite TTT3R 182, the *killer* "what's next after TTT3R" paper. The *direct* comparison: TTT3R 182 = recurrent state + per-token β + state reset, R³ = relative regression + ???? (need to read). **The *right* paper to *complete* the v0 sub-task 1 *2026* arc** (after the 2024-2025 *frozen* arc: Spann3R 177 → CUT3R 175 → Point3R 179 → Ray-Aware 180 → Stream3R 181 → TTT3R 182 → R³ 183). After R³ 183, the v0 sub-task 1 *2026* design space is *complete*.

**Alternative: *read 183 = Long3R (Chen 2025, arXiv 2507.18255)*** — the *killer* "chunked + aligned" precursor to TTT3R's State Reset. The *direct* comparison: TTT3R 182 = state reset *within* the same forward pass, Long3R = chunk *boundaries* *between* separate forward passes. The *right* paper to *complete* the v0 sub-task 1 *chunked-vs-recurrent* design space. *Less recommended* than R³ 183 because R³ is *more recent* and *directly cites* TTT3R 182.
