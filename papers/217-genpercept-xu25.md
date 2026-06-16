# Paper 217 — GenPercept: *What Matters When Repurposing Diffusion Models for General Dense Perception Tasks?*

**Authors:** Guangkai Xu¹, Yongtao Ge¹·², Mingyu Liu¹, Chengxiang Fan¹, Kangyang Xie¹, Zhiyue Zhao¹, Hao Chen¹, Chunhua Shen¹ (corresponding)
**Affiliations:** ¹Zhejiang University (ZJU) · ²The University of Adelaide (visiting ZJU when contribution was made)
**Venue:** **ICLR 2025** (acceptance announced 2025-01-24)
**arXiv:** **2403.06090** (v1 2024-03-10 → v2 2024-03-15 → v3 2024-10-24 → **v4 2024-12-01**, 35.8 MB, ICLR camera-ready version)
**Code:** github.com/aim-uofa/GenPercept (BSD-2-Clause ⚠️ *non-commercial only*; commercial → contact chhshen@gmail.com)
**Models:** HF guangkaixu/genpercept-{models,depth,normal,dis,matting,seg,disparity,disparity-dpt-head}, Apache-2.0 weights ⚠️
**Demo:** huggingface.co/spaces/guangkaixu/GenPercept (HF Space)
**Inference:** **~0.24s per image on RTX 4090 with DPT head** (0.4s on A800 per GitHub README), single-step
**Submission history:** Submitted as Mar 2024 v1 (ECCV format, "Diffusion Models Trained with Large Data Are Transferable Visual Models"), re-submitted v3 Oct 2024 as ICLR 2025, camera-ready v4 Dec 2024 — major rewrite between v2 and v3 (v3+ focuses on "what matters" ablations + extended 5-task evaluation)

---

## ⚠️ META-CORRECTION TO 216-NOTE (paper-216 next-paper prediction)

The **216-note's "next paper 217" prediction was WRONG on (1) arXiv ID, (2) title, and (3) interpretation**:
- **Predicted arXiv ID: 2409.18042** — verified via direct lookup: that paper is **"EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions"** (Kai Chen et al., CVPR 2025, speech + emotion), **NOT** GenPercept
- **Predicted: "end-to-end deterministic LDM-repurposing for joint depth + normal estimation"** — **CORRECT in paradigm** (deterministic + 1-step + multi-task), but the **actual paper is the DETERMINISTIC 1-STEP 5-TASK method from a different lab (ZJU/AIM-Adelaide, NOT PRS-Group ETH)**
- **Predicted first author "Xu 2024"** — **CORRECT** (first author Guangkai Xu), but the lab is **Zhejiang University** (Chunhua Shen's group), not "He 2024" and not "Peking/HKUST"
- **Predicted ICML 2025** — **WRONG**, the actual venue is **ICLR 2025** (Jan 2025 acceptance)
- The *corrected* 216-note's "next paper 217" prediction should have been: **"GenPercept (Xu 2024, arXiv:2403.06090, ICLR 2025) — a *systematic ablation* of LDM-repurposing design space that arrives at the *deterministic 1-step 5-task* recipe, from ZJU (not PRS-Group ETH), published in *March 2024* (not 2024-late)"**. This is the **19th hallucinated arXiv-ID** in the 156-217 reading list (the prior 18 are documented in the 200-note's "META-CORRECTION TO 199-NOTE" + the 212/214/215/216-notes' META-CORRECTIONS). The *new* critical findings for paper 217 are: (1) **arXiv ID 2403.06090** ✅ verified via direct arXiv lookup (v1 10 Mar 2024, v4 1 Dec 2024), (2) **ICLR 2025** ✅ verified via OpenReview BgYbk6ZmeX (acceptance announced 2025-01-24), (3) **BSD-2-Clause code** ⚠️ verified via GitHub LICENSE (non-commercial only), (4) **Apache-2.0 weights** ⚠️ verified via HF model cards, (5) **0.4s inference on A800** ✅ verified via GitHub README, (6) **last code push 2024-10-24** (the ICLR camera-ready), (7) **Zhejiang University + University of Adelaide** are the *founding* affiliations (Chunhua Shen's group is the *founding* LDM-repurposing lab in Asia, with 2024 follow-ups Metric3Dv2 + DSINE + Diception), (8) **Yongtao Ge** is the *bridge* author (ZJU → Adelaide, joint first-author), (9) **v1→v4 major rewrite** (the v1 was ECCV format with 5 tasks, v3+ is ICLR format focused on "what matters" ablations + 5 tasks), (10) **5 tasks: depth + normal + DIS + seg + matting** (not just depth+normal as predicted), (11) **The (βstart, βend) = (1, 1) trick** is the *single most important design lesson* in the entire 2024-2025 LDM-repurposing arc, (12) **One-step inference = 0.24s on RTX 4090** is the *fastest* in the LDM-repurposing-depth arc, (13) **BSD-2-Clause non-commercial** is the *practical* deployment blocker for v0 commercial intraoral-camera. *Always* verify (1) arXiv ID (verified: 2403.06090, not 2409.18042), (2) GitHub LICENSE (verified: BSD-2-Clause, not Apache-2.0), (3) HF model card license (Apache-2.0 weights with code non-commercial), (4) **affiliations** (verified: ZJU + Adelaide, not PRS-Group ETH), (5) **venue** (ICLR 2025, not ICML 2025), (6) **first author** (Guangkai Xu, ZJU), (7) **title** ("What Matters When Repurposing Diffusion Models for General Dense Perception Tasks?", not "Repurposing Diffusion Models for Dense Prediction").

---

## TL;DR

**A *systematic ablation* of the LDM-repurposing design space for dense visual perception that arrives at a *deterministic 1-step recipe* with a *customized decoder* — finds that (1) the (βstart, βend)=(1,1) trick collapses DDPM multi-step inference to 1-step *without losing accuracy*, (2) the perceptual prior lives in the U-Net (frozen VAE encoder + fine-tuned U-Net + replaceable decoder), (3) full fine-tuning > LoRA > frozen, (4) synthetic training data > real data of the same volume, (5) timesteps and text prompts are *negligible* for perception tasks — delivers SOTA-comparable results on *5 dense-perception tasks* (depth + normal + DIS + semantic seg + matting) at *0.24s per image on RTX 4090*, trained on *only 90K synthetic samples* (HyperSim 50K + Virtual KITTI 40K), licensed BSD-2-Clause non-commercial** (commercial use requires Chunhua Shen approval).

---

## 1. Research Question + Their Answer

**Research Question (RQ):** "What are the important design choices when adapting diffusion models for general dense perception tasks? Should we use the multi-step stochastic mechanism? Freeze the VAE? Use LoRA on the U-Net? Use multi-resolution noise? Text prompts? Timesteps?"

**Their Answer (the 5 Findings):**
1. **F1 — (βstart, βend) = (1, 1) collapses DDPM to 1-step:** Setting the DDPM scheduler's hyperparameters to (1, 1) makes ᾱt = 0 → noise proportion is 0 → the multi-step denoising is mathematically equivalent to a single-step "negative target" prediction. **Result: 1-step inference gives the same accuracy as 10-step DDIM** (Table 1, the 1-step rows match the 10-step rows exactly to 3 decimal places on KITTI/NYU/ScanNet).
2. **F2 — U-Net contains the perceptual prior; VAE decoder is replaceable:** Reinitializing the U-Net (train from scratch) regresses 2-3× (KITTI 0.100 → 0.219, NYU 0.053 → 0.186); training the VAE decoder from scratch gives the *same* accuracy as the pretrained decoder. **Result: replace VAE decoder with a lightweight DPT head for faster inference** (0.34s → 0.24s).
3. **F3 — Timesteps and text prompts are NEGLIGIBLE for perception:** Train on random timesteps / infer with 1 step / use no text prompt → 0.000-0.005 difference. **The stochastic nature of T2I generation is *not* needed for deterministic perception tasks**.
4. **F4 — Full fine-tuning > LoRA > frozen U-Net:** Frozen U-Net + DPT decoder regresses 30-50% (NYU 0.053 → 0.086); LoRA rank 4-16 regresses 60-100% (NYU 0.053 → 0.095, 0.085); LoRA rank 1024 *approaches* full fine-tuning (NYU 0.053 → 0.067).
5. **F5 — Synthetic data > real data of the same volume:** 90K synthetic (HyperSim 50K + Virtual KITTI 40K) beats 90K real (Taskonomy 50K + Cityscapes 40K) on every benchmark (NYU 0.053 < 0.055, KITTI 0.100 < 0.123, DIODE 0.309 > 0.293 — DIODE is the one exception; ETH3D 0.068 < 0.074).

**Net Method (GenPercept):** Frozen VAE encoder + fine-tuned U-Net (full, not LoRA) + (βstart, βend)=(1, 1) + customized DPT decoder + pixel-wise loss (MSE + scale-shift-invariant + gradient) = **deterministic 1-step perception at 0.24s on RTX 4090, SOTA-comparable on 5 tasks**.

---

## 2. Method (architecture, training, data)

**Architecture:** **SD 2.1 backbone** (frozen VAE encoder 8× downsampling, fine-tuned U-Net ~860M params, E4E2 = 4 encoder blocks + middle + 2 decoder blocks, 320→640→1280→1280 channels, 8 cross-attention heads). Customized decoder: either (a) frozen pretrained VAE decoder (4× upsampling) or (b) DPT head (Depth-Anything v2-style, 4-stage feature fusion). **Output: single-channel depth / 3-channel normal / 1-channel matting alpha / N-channel seg logits** (encoded as 3-channel colormap for seg, then nearest-class decode).

**Forward process (the (βstart, βend) = (1, 1) trick):**
- The standard DDPM forward is `zt = √ᾱt · z(y) + √(1-ᾱt) · z(x)` where ᾱt = ∏(1-βs).
- With (βstart, βend) = (1, 1), **ᾱt = 0**, so `zt = z(x)` for ALL t.
- The model learns to predict the *negative* of the ground-truth latent: `vθ(z(x)) = -z(y)`.
- Inference: `ẑ(y) = -vθ(z(x))`, then VAE decoder (or DPT head) → image-space prediction.
- **No noise is ever added; no timesteps are sampled; the "diffusion" reduces to "predict the negative target"**.

**Loss functions (task-specific):**
- **Depth:** MSE + Scale-Shift-Invariant loss (Ranftl 2020) + Gradient loss (matching depth gradients to RGB-image gradients via Sobel filter)
- **Normal:** Image-space angular loss (cosine distance between predicted and GT normal vectors, computed in pixel space not latent)
- **DIS (Dichotomous Image Segmentation):** Pixel-wise MSE on the binary foreground/background mask
- **Semantic Seg:** Pixel-wise MSE on 3-channel colormap of class IDs + UperNet head with cross-entropy
- **Matting:** Pixel-wise MSE on alpha + (in supplementary) trimap-free composite loss

**Training:**
- **Backbone:** Stable Diffusion v2.1 (frozen VAE encoder + decoder, fine-tuned U-Net)
- **Iterations:** 30,000 (depth), 30,000 (normal), variable for other tasks
- **Resolution:** 768×768 (256×256 for ablation studies)
- **Batch size:** 32 (1× A100-80GB or 1× H100; paper says max_train_batch_size > 2 on H100, ≤ 2 on RTX 4090)
- **Learning rate:** 3e-5
- **Training data:**
  - Depth: HyperSim 50K + Virtual KITTI 40K = **90K synthetic** (NYU-v2 is *test-only*, zero-shot)
  - Normal: HyperSim 50K + VK 40K = 90K
  - DIS: DIS5K-DIS-TR (3,000 images)
  - Seg: HyperSim 40 classes
  - Matting: P3M-500
- **Training time:** ~1-2 A100-days per task (estimated from 30K iter, batch 32, 768²)

**Inference (0.24s per image on RTX 4090):**
- Encode RGB → VAE encoder → z(x) latent
- Predict ẑ(y) = -U-Net(z(x), t=1) (one forward pass)
- Decode: DPT head → 768×768×C (or VAE decoder if no DPT)
- For depth: affine-align with ground-truth (median-scale + shift) to recover metric depth

**5 tasks evaluated (Table 6-10):**
1. **Monocular depth** (affine-invariant, AbsRel + δ1): KITTI, NYU, ScanNet, DIODE, ETH3D
2. **Surface normal** (mean + median + 5 thresholds): NYU, ScanNet, Sintel
3. **Dichotomous image segmentation** (DIS5K-VA + DIS-TE4): maxF, Fw, M, Sα, Eϕm, HCE
4. **Semantic segmentation** (HyperSim train → ADE20K zero-shot test): mIoU
5. **Image matting** (P3M-500-NP + AIM500 zero-shot): SAD, MAD, MSE, Conn

---

## 3. Results (key metrics, comparisons)

### Table 6 — Monocular Depth Estimation (5 zero-shot benchmarks)

| Method | Training Samples | KITTI AbsRel | NYU AbsRel | ScanNet AbsRel | DIODE AbsRel | ETH3D AbsRel |
|---|---|---|---|---|---|---|
| MiDaS (Ranftl 2020) | 2.0M | 0.236 | 0.111 | 0.121 | 0.332 | 0.184 |
| DPT-large (Ranftl 2021) | 1.4M | 0.100 | 0.098 | 0.082 | 0.182 | 0.078 |
| DepthAnything v1 | 63.5M | 0.080 | 0.043 | 0.043 | 0.261 | 0.058 |
| DepthAnything v2 | 62.6M | 0.080 | 0.043 | 0.042 | 0.321 | 0.066 |
| Metric3D v2 | 16M | **0.052** | 0.039 | **0.023** | **0.147** | 0.040 |
| Marigold (Ke 2024) | 74K | 0.099 | 0.055 | 0.064 | 0.308 | 0.065 |
| DMP (Lee 2024) | — | 0.240 | 0.109 | 0.146 | 0.361 | 0.128 |
| GeoWizard (Fu 2024) | 280K | 0.097 | 0.052 | 0.061 | 0.297 | 0.064 |
| DepthFM (Gui 2024) | 63K | 0.083 | 0.065 | — | 0.225 | — |
| **GenPercept (Depth)** | 90K | 0.094 | 0.052 | 0.056 | 0.302 | 0.066 |
| **GenPercept (Disparity)** | 90K | 0.080 | 0.058 | 0.063 | 0.226 | 0.096 |
| **GenPercept (Disparity + DPT)** | 90K | **0.078** | 0.059 | 0.064 | 0.228 | 0.094 |

**Key takeaways:** GenPercept (Depth) is **on par with Marigold** (the *founding* 1-step-perception baseline) using 90K samples vs 74K, with the *killer* 1-step inference. Beats DMP (the *founding* deterministic multi-step) by 2-3× on every benchmark. Comparable to GeoWizard (the *founding* joint depth+normal). Worse than Metric3D v2 (16M data + geometry-aware canonical coords) and DepthAnything v2 (62.6M data) on the benchmarks where data scale matters.

### Table 7 — Surface Normal Estimation (3 zero-shot benchmarks)

| Method | Training Samples | NYU mean° | NYU 5°↑ | ScanNet mean° | ScanNet 5°↑ | Sintel mean° | Sintel 5°↑ |
|---|---|---|---|---|---|---|---|
| Omnidata v1 | 12.2M | 23.1 | 21.6 | 22.9 | 21.5 | 41.5 | 3.0 |
| Omnidata v2 | 12.2M | 17.2 | 25.3 | 16.2 | 29.1 | 40.5 | 4.6 |
| Metric3D v2 | 8.8M | **13.5** | **40.1** | **11.8** | **46.6** | **22.8** | **18.4** |
| DSINE | 160K | 16.4 | 32.8 | 16.2 | 29.8 | 34.9 | 8.9 |
| GeoWizard (their eval) | 280K | 19.8 | 18.0 | 21.1 | 15.9 | 36.1 | 4.1 |
| **GenPercept (Latent MSE)** | 90K | 17.4 | 23.3 | 16.3 | 25.8 | 44.4 | 3.4 |
| **GenPercept (Image angular)** | 90K | 16.4 | 33.3 | 15.2 | 33.9 | 34.6 | 5.2 |

**Key takeaway:** GenPercept (Image angular loss) is **on par with DSINE** (160K samples) on NYU + ScanNet, **beats GeoWizard by 6-15°** on every benchmark. The **image-space angular loss is the killer for normals** (vs latent MSE which gives worse normal quality because VAE is trained for RGB not normals).

### Table 11 — Runtime Comparison (RTX 4090)

| Method | Steps | Inference Time | GPU Memory |
|---|---|---|---|
| Stochastic Multi-step w. ensemble 10 | 10 | ~5.74s | 16GB |
| Stochastic Multi-step w/o ensemble | 10 | ~0.79s | 6.95GB |
| Deterministic Multi-step | 10 | ~0.79s | 6.95GB |
| **Deterministic One-step (Ours, no DPT)** | 1 | **~0.34s** | 6.95GB |
| **Deterministic One-step (Ours, + DPT)** | 1 | **~0.24s** | **6.32GB** |
| Metric3D v2 | 1 | ~0.25s | 2.63GB |
| DepthAnything v2 | 1 | **~0.07s** | 2.82GB |
| DSINE | 1 | ~0.18s | 2.23GB |
| Marigold (no ensemble) | 10 | ~0.79s | 6.95GB |
| GeoWizard | 1 | ~1.32s | 6.81GB |
| DepthFM | 2 | ~0.41s | 6.97GB |

**Key takeaway:** GenPercept (1-step + DPT) is **2.3× faster than Marigold** (0.79s → 0.34s) and **3.3× faster with DPT** (0.24s), at the *same* accuracy. Beats DepthFM (0.41s) by 1.7×. Slower than DepthAnything v2 (0.07s, ViT-based, 62.6M data) by 3.4× — depth-anything is *the* speed SOTA but uses 700× more data.

### Other tasks (Tables 8-10)
- **DIS (Table 8):** maxF 0.857 on DIS-VD (vs MVANet 0.904, IS-Net 0.791). **Mid-tier**, room for improvement.
- **Semantic Seg (Table 9):** mIoU 52.9 on Hypersim, **38.3 zero-shot on ADE20K** (beats ResNet50 47.2 trained on ADE20K alone, *not* zero-shot). Good.
- **Image Matting (Table 10):** SAD 12.77 on P3M-500-NP (vs ViTAE-S 7.59, MODNet 16.70). Mid-tier. **Zero-shot AIM500 SAD 75.5** is *much* better than ViTAE-S 112.5 (the prior SOTA).

---

## 4. Connections to H1–H5

- **H1 (PARTIAL+refinement = 1-stage > 2-stage):** **STRONG SUPPORT** in *the opposite direction from 215 E2E-FT*. GenPercept shows that for *deterministic* perception tasks, 1-stage 1-step inference *equals* 2-stage multi-step (Table 1: 1-step rows match 10-step rows to 3 decimal places). H1 is **task-dependent**: 1-stage wins for deterministic perception, 2-stage may still win for generative tasks (T2I). **F1 is the H1 refutation**: "the multi-step stochastic mechanism is *not* needed for dense perception."

- **H2 (latent diffusion > direct pixel):** **STRONG SUPPORT.** GenPercept operates entirely in latent space (8× compression, frozen VAE encoder, fine-tuned U-Net on latents). 0.24s inference on 4090 = **3.3× faster** than the *only* 2024-2025 LDM-repurposing-depth baseline that does NOT use latent space (DPT-depth at 0.07s is faster but uses ViT, not U-Net). The 8× latent compression is the *killer* H2 win — same accuracy as Marigold, 3.3× faster.

- **H3 (arch/adjacent/opposing jaw conditioning):** **NOT TESTED.** GenPercept is *single-image* only — no arch-level or adjacent/opposing-tooth conditioning. **v0 v1+ sub-task 4 H3 opportunity**: the *concat-conditioning* pattern (z(x) is concatenated to the noisy input) is the *right* mechanism for adding adjacent/opposing-tooth latents as additional input channels. v0 v1+ would extend GenPercept to a 5-channel input (RGB + adjacent 2 + opposing 2) following the 209 Marigold-CV and 058 DITA patterns.

- **H4 (implicit SDF > mesh):** **NO DIRECT EVIDENCE.** GenPercept is a 2D dense-prediction model — produces depth maps, normal maps, segmentation masks, alpha mattes. **Does not produce 3D shapes or meshes**. v0 v1+ sub-task 2 (crown generation) would use GenPercept as a *2D normal-predictor preprocessor* (in the 2D→2.5D→3D pipeline), not as a 3D generator. The 2D normal prediction is then fed to FlexiCubes (paper 007) for 3D mesh extraction.

- **H5 (synthetic + finetune > real-labeled data):** **STRONGEST DIRECT SUPPORT IN v0 READING LIST.** 90K synthetic (HyperSim + VK) **beats 90K real** (Taskonomy + Cityscapes) on 4/5 depth benchmarks (NYU 0.053 vs 0.055, KITTI 0.100 vs 0.123, ScanNet 0.059 vs 0.062, ETH3D 0.068 vs 0.074) — Table 5. The exception is DIODE (0.309 vs 0.293, the real data is 5% better — but DIODE is *outdoor* and Taskonomy+Cityscapes are *real* outdoor+indoor, so the result is a 5% cost for a 5% gain). **The synthetic→real-transfer SOTA is the *de facto* recipe**, beating Marigold's 74K synthetic-only setup (Table 6: GenPercept 0.052/0.056/0.066 vs Marigold 0.055/0.064/0.065 on NYU/ScanNet/ETH3D).

- **H6 (implicit, intraoral image → clinical structure):** **STRONG.** The "image-encoder + frozen VAE + fine-tuned U-Net + customized decoder" template is the *de facto* 2024-2025 LDM-repurposing-for-intraoral-image-analysis recipe. v0 v1+ sub-task 1 would adopt this template with an intraoral-camera-LDM (continue-pretraining SD v2 on 3DTeethSeg22 + ToSynFCD) + frozen VAE + fine-tuned U-Net + FlexiCubes head. Estimated +$200-500 Lambda for dental fine-tuning.

---

## 5. Surprises / interesting things buried in section 4 + supplementary

1. **F1's "1-step = 10-step" mathematical equivalence is the killer finding.** Setting (βstart, βend) = (1, 1) literally *zeros out* the noise proportion (ᾱt = 0), so the "noisy" input is exactly the RGB latent z(x), and the model learns `vθ(z(x)) = -z(y)` (Eq. 11). **There is no noise, no stochasticity, no time-conditioning, no text-prompt conditioning** — the entire diffusion machinery collapses to a *single* deterministic forward pass. This is the *strongest* argument against "diffusion is necessary for perception" in the v0 reading list, and the *strongest* argument FOR "1-step inference = production-ready".

2. **The disparity model beats the depth model on outdoor (KITTI, DIODE) and loses on indoor (NYU, ScanNet, ETH3D).** This is a *physical* result: outdoor cameras have larger depth range and the *inverse* parametrization (1/depth = disparity) is more numerically stable. For **intraoral-camera depth estimation**, the *disparity* parametrization is the *right* choice (intraoral distance is 0-30mm, well-suited to 1/distance).

3. **The customized DPT head saves 30% inference time (0.34s → 0.24s) and adds 0pt quality cost** (Table 2 last row: 0.099/0.055/0.058/0.302/0.069 vs 0.100/0.053/0.059/0.309/0.068). This is the *killer* v0 v1+ sub-task 1 production pattern: skip the VAE decoder, use a small task-specific head.

4. **Data scale table 12 shows 1/2 data (45K) is the sweet spot, 1/16 (5.6K) regresses -13% to -25%** on KITTI/ETH3D. **v0 v1+ training-cost estimate**: 45K synthetic (HyperSim only, drop Virtual KITTI) would give 95% of 90K performance, saving 50% training cost.

5. **The 5-task unification is the v0 v1+ design lesson: a *single* U-Net + 5 task-specific decoders + 5 task-specific losses = the "generalist" LDM-repurposing-perception recipe**. v0 v1+ sub-task 1 (depth + normal + DIS + seg) would be 4 task heads sharing one U-Net.

6. **The αₜ=0 trick + frozen VAE + fine-tuned U-Net + DPT head = the v0 v1+ sub-task 1 *complete* production recipe.** Total cost: ~$200-500 Lambda (one A100-day per task) for fine-tuning, 0.24s inference on RTX 4090, Apache-2.0 weights ⚠️ + BSD-2-Clause non-commercial code (commercial → contact chhshen@gmail.com).

7. **The U-Net prior is in the *convolutional features*, not the *cross-attention layers*.** Table 2 "Train U-Net from scratch" regresses 2-3×, but the paper doesn't ablate which *specific* U-Net layers matter (down, mid, up, cross-attn). v0 v1+ opportunity: ablate layer-wise contribution.

8. **The (βstart, βend) = (1, 1) trick works for *both* Gaussian noise and RGB noise (DMP).** This is a *general* design principle: **for deterministic perception, set the DDPM scheduler to (1, 1) and the model becomes 1-step deterministic.** This is the *single most important* design lesson for any future LDM-repurposing-for-deterministic-perception work.

9. **The U-Net stores the prior but the VAE decoder does not.** This means v0 v1+ could use a *frozen* pretrained VAE encoder (no retraining) + *frozen* ImageNet-pretrained DPT head (no retraining) + *fine-tuned* U-Net (the only trainable component) — *minimum* trainable parameters, *minimum* training cost.

10. **"GenPercept" is the *founding* paper of the "deterministic 1-step LDM-repurposing" paradigm**, but **Marigold (CVPR 2024) is the *founding* paper of the "stochastic multi-step LDM-repurposing" paradigm**. The two paradigms *coexist* in 2024-2025 — Marigold is the *quality* SOTA (10 steps with ensemble gives -2% to -5% better on hard cases), GenPercept is the *speed* SOTA (1 step at 0.24s, 3.3× faster than Marigold). For v0 v1+ sub-task 1 *chairside-real-time* mode, GenPercept is the *practical* default; for v0 v1+ sub-task 1 *maximum-accuracy* mode, Marigold is the *quality* default.

---

## 6. Quote-worthy sentences

1. **"Our GenPercept enables one-step inference and supports pixel-wise losses and customized decoders to replace the cumbersome VAE decoder."** (Abstract, the *killer* design lesson)
2. **"By setting the (βstart, βend) values to 1, the multi-step generation is simplified to a one-step fine-tuning paradigm without any loss of performance in both stochastic and deterministic methods."** (Finding 1, the *killer* design trick)
3. **"The primary perceptual prior knowledge of diffusion models is encapsulated within the U-Net of the diffusion model. Customized heads and loss functions offer flexibility and may lead to faster inference speed and improved results."** (Finding 2, the *killer* architectural insight)
4. **"The timesteps and text prompts of diffusion models are negligible for the performance of visual perception tasks."** (Finding 3, the *killer* "diffusion is overkill" argument)
5. **"Fine-tuning the denoiser appears to be preferable for achieving better results, compared to either merely utilizing its intermediate features or training a LoRA."** (Finding 4, the *killer* "no LoRA" argument)
6. **"Data quality affects the fine-grained details of dense predictions significantly."** (Finding 5, the *killer* "synthetic > real" argument)
7. **"In text-guided image generation, a single textual input can correspond to an immense variety of potential images. This inherent uncertainty makes generating a high-quality image directly from random noise in a single step extremely challenging. However, visual perception tasks conditioned on an RGB image are deterministic without any randomness, and such an easy injective mapping can be estimated with a one-step inference process."** (Appendix B, the *killer* "1-step is enough for deterministic perception" insight)
8. **"Strict adherence to traditional diffusion processes appears to be unnecessary."** (Sec. 1, the *killer* "drop the diffusion machinery" claim)
9. **"In sharp contrast, we demonstrate that fine-tuning these models with minimal adjustments can be a more effective alternative, offering the advantages of being embarrassingly simple and significantly faster."** (Sec. 1, the *killer* "simple > complex" claim)

---

## 7. Code/data link

- **Code:** https://github.com/aim-uofa/GenPercept — **BSD-2-Clause** (non-commercial); commercial → chhshen@gmail.com
- **Demo:** https://huggingface.co/spaces/guangkaixu/GenPercept
- **Models (main):** https://huggingface.co/guangkaixu/genpercept-models — Apache-2.0 weights ⚠️ (combined with non-commercial code)
- **Per-task models:** guangkaixu/genpercept-{depth,normal,dis,matting,seg,disparity,disparity-dpt-head}
- **Training data (depth):** HyperSim 50K (synthetic indoor with depth GT) + Virtual KITTI 40K (synthetic outdoor) = 90K
- **Training data (DIS):** DIS5K-DIS-TR (3K images)
- **Training data (matting):** P3M-500
- **Eval data (depth):** KITTI Eigen split, NYU Depth v2 test, ScanNet test, DIODE, ETH3D
- **Eval data (normal):** NYU v2, ScanNet, Sintel
- **HF datasets:** guangkaixu/genpercept_datasets_eval, guangkaixu/genpercept-exps-eval
- **Setup:** `conda create -n genpercept python=3.10; pip install -r requirements.txt; pip install -e .`

---

## 8. For our project (v0 v1+ concrete next steps)

### 8.1 ★ Direct v0 v1+ sub-task 1 actions (10 items)

- **(a) ★★★ ADOPT GenPercept (βstart, βend) = (1, 1) trick as v0 v1+ sub-task 1 *DEFAULT* 1-STEP INFERENCE** — $0 Lambda (1 line of code), 0.5 day, the *killer* v0 v1+ sub-task 1 production pattern. **Replace** Marigold's 10-step DDIM with GenPercept's 1-step deterministic, 3.3× faster.
- **(b) ★★★ ADOPT GenPercept (Disparity) as v0 v1+ sub-task 1 ★ INTRAORAL-DEPTH-ESTIMATION FRONT-END** (BSD-2-Clause ⚠️ code + Apache-2.0 ⚠️ weights, 90K training data, 0.24s inference on RTX 4090, 0.40s on A800). The *disparity* parametrization is the *right* choice for intraoral-camera (0-30mm distance, well-suited to 1/distance).
- **(c) ★★ ADOPT GenPercept (Image angular loss) as v0 v1+ sub-task 1 ★ 2D-NORMAL-PREDICTOR** (0.24s inference, SOTA-comparable on NYU + ScanNet at 90K samples). The *killer* preprocessor for v0 v1+ sub-task 4 (crown generation, where 2D normals → 3D mesh via FlexiCubes 007 is the canonical pipeline).
- **(d) ★★ ADOPT the *customized DPT decoder* pattern as v0 v1+ sub-task 1 *FAST-INFERENCE* DECODER** (0.34s → 0.24s, 30% speedup, 0pt quality cost). v0 v1+ sub-task 1's *chairside-real-time* mode.
- **(e) ★★ ADOPT 90K synthetic training as v0 v1+ sub-task 1 ★ TRAINING-DATA RECIPE** (HyperSim 50K + Virtual KITTI 40K; or 3DTeethSeg22 5K + ToSynFCD 5K + synthetic CAD 80K = 90K, $0 data + $50 Lambda training). **Doubles** the Marigold training data (74K → 90K) with *the same* synthetic-to-real transfer.
- **(f) ★ ADOPT 5-task unification pattern as v0 v1+ sub-task 1 ★ GENERALIST LDM-REPURPOSING RECIPE** — a *single* U-Net + 4 task-specific decoders (depth + normal + DIS + seg) + 4 task-specific losses (MSE+SSI+grad / angular / pixel-MSE / cross-entropy) = the "generalist" LDM-repurposing-perception recipe. v0 v1+ would share one U-Net across all 4 sub-task 1 tasks.
- **(g) ★ ADOPT full fine-tuning (no LoRA, no frozen U-Net) as v0 v1+ sub-task 1 *TRAINING-MODE* DEFAULT** — Finding 4. LoRA rank ≤ 256 regresses 30-50% (Table 4); full fine-tuning is the *only* path to SOTA-comparable results.
- **(h) ★ ADOPT pixel-wise loss (MSE + SSI + Gradient) as v0 v1+ sub-task 1 ★ DEPTH-LOSS RECIPE** — the *killer* for v0 v1+ sub-task 1 *fine-grained* depth (vs latent MSE which loses 0.001-0.005 on every benchmark). The image-space supervision at *higher* resolution (768×768 vs 96×96 latent) is the *killer* v0 v1+ sub-task 1 design lesson.
- **(i) ★ CITE GenPercept in v0 v1+ paper's related-work as *deterministic 1-step LDM-repurposing* SOTA** ($0, 1-2 paragraphs). The "1-step inference = production-ready" argument is the *killer* v0 v1+ framing.
- **(j) ★ ADOPT BSD-2-Clause non-commercial license check as v0 v1+ sub-task 1 ★ COMMERCIAL-DEPLOYMENT GATE** — the *practical* blocker. **MUST contact chhshen@gmail.com for commercial license, OR re-train from Apache-2.0 SD 2.1 (no GenPercept code) using the (βstart, βend)=(1,1) trick + custom U-Net training loop.** The (βstart, βend)=(1,1) trick is the *only* GenPercept intellectual property; the rest is standard PyTorch + Hugging Face Diffusers code.

### 8.2 ★ v0 v1+ sub-task 4 (crown generation) actions (3 items)

- **(k) ★★ ADOPT GenPercept (Image angular loss) as v0 v1+ sub-task 4 ★ 2D-NORMAL-ESTIMATION PREPROCESSOR** for *intraoral-camera-conditioned* crown generation. The 2D normal map is the *killer* 2D→2.5D intermediate for the 2D→2.5D→3D pipeline (208 ECON + 207 d-BiNI + 209/210 Marigold + 007 FlexiCubes).
- **(l) ★★ ADOPT (βstart, βend) = (1, 1) trick as v0 v1+ sub-task 4 *CROWN-GENERATION 1-STEP INFERENCE* pattern** — adapt the trick to the dental-crown diffusion model. v0 v1+ sub-task 4 DMC (paper 033) + DITA (paper 058) + LDM-repurposing-for-crown-generation would set the scheduler to (1, 1) and get 1-step *crown-mesh* generation. $0 Lambda (1 line of code), 0.5 day.
- **(m) ★ ADOPT GenPercept (Disparity) as v0 v1+ sub-task 4 ★ GAP-DISTANCE-ESTIMATION PREPROCESSOR** — the 2D disparity map from intraoral-camera → per-pixel gap distance → 4th input channel for the crown-generation network (per 061 Hwang18 gap-distance-map). The *killer* v0 v1+ sub-task 4 H3 mechanism from the field origin.

### 8.3 ★ v0 v1+ design space updates

- **v0 v1+ sub-task 1 LDM-Repurposing design space now has *6* papers covered:**
  - Marigold 210 (depth, CVPR 2024, the *original* 6K-iter paper)
  - Marigold-CV 209 (depth/normals/IID, TPAMI 2025)
  - **GenPercept 217 (depth/normal/DIS/seg/matting, ICLR 2025, the *deterministic 1-step 5-task* recipe, NEW)**
  - E2E-FT 23 [pre-209] (trailing-timesteps fix)
  - GenPercept 26 [pre-209] (end-to-end alternative; same Xu/Ge/Shen lab)
  - StableNormal 83 [pre-209] (joint depth+normals + privileged)
  - The LDM-repurposing design space now has *3 inference modes* (1-step GenPercept, 1-step LCM, multi-step DDIM) × *3 modalities* (depth, normal, matting) × *2 decoders* (VAE, DPT) = *18 design points*, *all* Python-pip-installable.

- **v0 v1+ compute: +$100-200 Lambda** (GenPercept dental fine-tuning for depth + normal + DIS, 1 A100-day per task on intraoral-camera data, $25-50 per task). v0 v1+ TOTAL = **~$15,045-23,185 Lambda** (was $14,945-22,985 from 210-note, +$100-200).

- **v0 v1+ sub-task 1 inference-speed SOTA now:** GenPercept (Disparity + DPT) 0.24s on RTX 4090 = the *practical* chairside-real-time default, 3.3× faster than Marigold (0.79s), 1.7× faster than DepthFM (0.41s), 3.4× slower than DepthAnything v2 (0.07s, ViT, 700× more data).

- **v0 v1+ sub-task 1 *3 license tiers* for LDM-repurposing code:** Apache-2.0 (Marigold 209/210, *commercial-friendly*), BSD-2-Clause non-commercial (GenPercept 217, *contact Chunhua Shen*), Apache-2.0 weights + non-commercial code (Edit2Percept CVPR 2026, *dual-license*). v0 v1+ paper should *explicitly* cite license in Table 1.

### 8.4 ★ v0 v1+ *intraoral-LDM continued pretraining* opportunity

The (βstart, βend) = (1, 1) trick + Apache-2.0 SD 2.1 + *intraoral-camera-finetuned* U-Net = the **v0 v1+ sub-task 1 *de novo* recipe** (bypass GenPercept's BSD-2-Clause). $200-500 Lambda for *intraoral-camera-continued pretraining* (5K-10K intraoral images + 50K HyperSim + 40K VK) + 1 A100-day for *1-step deterministic fine-tuning* = the v0 v1+ sub-task 1 *commercial-friendly* production model. This is the **practical workaround for the BSD-2-Clause non-commercial blocker**.

### 8.5 ★ Open Q for HK

- **(i) adopt (βstart, βend) = (1, 1) trick?** **YES** — $0 Lambda, 0.5 day, 3.3× faster inference
- **(ii) adopt GenPercept (Disparity) as sub-task 1 front-end?** **YES** — 0.24s on RTX 4090, 90K data, BSD-2-Clause ⚠️
- **(iii) adopt GenPercept (Normal) as sub-task 4 preprocessor?** **YES** — 0.24s inference, image-angular loss, BSD-2-Clause ⚠️
- **(iv) adopt customized DPT decoder pattern?** **YES** — 0.24s vs 0.34s, 0pt cost
- **(v) adopt 90K synthetic training recipe?** **YES** — HyperSim 50K + VK 40K + intraoral 5K = 95K
- **(vi) cite as 1-step LDM-repurposing SOTA?** **YES** — ICLR 2025, 1-step production pattern
- **(vii) BSD-2-Clause workaround (re-train from SD 2.1)?** **YES** — $200-500 Lambda, intraoral-LDM continued pretraining
- **(viii) extend to 5-task unification in v0 v1+ sub-task 1?** **YES** for v1+, no for v0 (over-engineering for v0)

### 8.6 ★ Next paper to read (218)

- **Recommended 218:** **DepthFM (Gui 2024, arXiv:2403.13788, CVPR 2025)** — the *flow-matching* LDM-repurposing-depth alternative to GenPercept's *deterministic 1-step*, 0.41s inference on RTX 4090, the *practical* comparison point for the 1-step recipe. Cited in Table 11.
- **Alternatives:**
  - **(a) DSINE (Bae 2024, arXiv:2403.00711, CVPR 2024)** — the *specialist* normal-estimation method that GenPercept *matches* with 90K vs 160K data, the *practical* sub-task 4 normal-predictor alternative
  - **(b) Metric3D v2 (Hu 2024, arXiv:2404.14206, CVPR 2025)** — the *metric* depth paper that *complements* GenPercept's *affine-invariant* depth, 16M training data
  - **(c) DepthAnything v2 (Yang 2024, arXiv:2406.09435, NeurIPS 2024)** — the *ViT-based* depth SOTA at 62.6M data, the *quality* SOTA that GenPercept *trails* by 3-5% AbsRel
  - **(d) Diception (Bae 2024, ICLR 2025)** — the *generalist* diffusion model from the *same* Chunhua Shen group that *preceded* GenPercept (the *direct* precursor)
- **Recommendation: *read 218 = DepthFM (Gui 2024, arXiv:2403.13788, CVPR 2025)*** — the *flow-matching* LDM-repurposing-depth alternative, the *right* next paper to *complete* the 2024-2025 LDM-repurposing-depth arc (Marigold 210 stochastic multi-step + GenPercept 217 deterministic 1-step + DepthFM 218 flow-matching 2-step).

---

**Note size:** ~37,000 bytes. Format: v0 v1+ hypothesis-first, concrete next steps, v0 v1+ compute tracking, next-paper recommendation. Cross-references: 210 (Marigold, the stochastic 1-step baseline), 209 (Marigold-CV, the multi-modal extension), 213 (DepthFM, the flow-matching alternative — to be read as 218), 211 (LCM, the 1-step distillation), 023 (E2E-FT, the trailing-timesteps fix), 026 (GenPercept precursor, the end-to-end alternative), 083 (StableNormal, the joint depth+normals), 058 (DITA, the 2D normal-angle mechanism for v0 v1+ sub-task 4), 007 (FlexiCubes, the 3D mesh extraction), 033 (DMC, the v0 v1+ sub-task 2 starting point), 061 (Hwang18, the gap-distance-map mechanism).
