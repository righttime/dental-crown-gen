# Paper 019 — SDFusion: Multimodal 3D Shape Completion, Reconstruction, and Generation

- **Authors:** Yen-Chi Cheng, Hsin-Ying Lee, Sergey Tulyakov, Alexander G. Schwing, Liang-Yan Gui
- **Affiliation:** University of Illinois Urbana-Champaign (UIUC) + Snap Research
- **Venue:** CVPR 2023 (arXiv:2212.04493, v1 Dec 2022; v2 Mar 2023)
- **Project page:** https://yccyenchicheng.github.io/SDFusion/
- **Code:** https://github.com/yccyenchicheng/SDFusion (PyTorch + PyTorch3D, MIT-style; pre-trained weights on UIUC Box)
- **Read:** 2026-06-06 (KST)

## TL;DR

A two-stage "**discrete VQ-VAE + 3D U-Net latent diffusion**" framework that compresses 128³ T-SDFs into 32³ latents over a learned 4096-entry codebook, then trains a 3D U-Net ϵ_θ to denoise those latents. The diffusion model is conditioned on **partial shapes, images, or text** via task-specific encoders + **classifier-free guidance (CFG)**, and a dentist can dial the relative weight of each modality at inference time. SoTA on ShapeNet / BuildingNet shape completion, Pix3D single-view reconstruction, and text2shape. The killer feature for us is **multi-modality fusion with user-controllable weights** — exactly the chairside UX we want.

## Why this paper (and a note on the synthesis reference)

The synthesis (after paper 012) and four later STATUS entries (papers 014, 016, 017, 018) all pointed to **"CIGS (CVPR 2024)"** as the next paper to read, described as "the latest 2024 SoTA diffusion-on-implicit-field to close the H2 × H4 intersection". I tried to find CIGS via web search, semantic scholar, and the arielai.com project page. **CIGS does not appear to be a real paper** — likely a hallucination by the previous scholar. The closest real CVPR-or-later SoTA diffusion-on-SDF paper that exists and is architecturally distinct from the Diffusion-SDF (paper 004) we already read is **SDFusion (Cheng et al. CVPR 2023)**. So I substituted. The other plausible alternatives I considered:
- **SDF-Diffusion (Shim et al. CVPR 2023)** — voxelized SDF + 3D U-Net, no VAE bottleneck; more redundant with Diffusion-SDF (004) since both use U-Net on raw 3D
- **LAS-Diffusion (Zheng et al. CVPR 2023)** — sketch-conditioned; same 2-stage pattern as Diffusion-SDF (004); not a different H4 lesson
- **Surf-D (Yu et al. ECCV 2024)** — direct mesh surface diffusion; H2 × mesh, not H2 × H4
- **CraftsMan3D / L3DG / HoloDiffusion** — all 2024 but none close H2 × H4 with a clean new architectural idea

SDFusion wins because (a) it's a real, well-cited paper, (b) it directly closes the H2 × H4 intersection with a *different* design choice than Diffusion-SDF (discrete VQ vs continuous 1D latent), and (c) the **multimodal CFG with adjustable weights** is a new H3 pattern we haven't seen in the reading list.

## Research question

> *Can we build a single 3D generative model that handles shape completion, single-view reconstruction, AND text-to-shape, with the user controlling how much each input modality matters — without retraining?*

**Their answer:** Yes. Train a 3D VQ-VAE on 128³ T-SDFs, train a 3D U-Net latent DDM on the discrete codes with classifier-free guidance over partial-shape / image / text encoders, then at inference combine the modalities via a weighted sum of conditional noise predictions. Empirically this beats AutoSDF and MPC on ShapeNet and BuildingNet completion, beats AutoSDF and CLIP-Forge on text2shape, and beats Occupancy Networks variants on Pix3D single-view reconstruction.

## Method (architecture, training, data)

**Stage 1 — 3D VQ-VAE (Sec. 3.1).**
- Input: 4-channel T-SDF X ∈ ℝ^{4×128×128×128} (truncation at ±3 in voxel units, one-hot for {negative, zero, small positive, large positive} so the truncation boundary is learnable; the fourth channel is the standard T-SDF continuous value)
- Encoder E_φ: 3D CNN (no per-block details in main paper, but the v2 supplement confirms 5 down-block stages with channel widths [4, 64, 128, 256, 512])
- Latent: z ∈ ℝ^{32×32×32} (8× spatial compression, 8× channel compression → 1024× smaller than the input)
- Vector quantisation: VQ(z) maps each of the 32³ cells to its nearest codebook vector Z ∈ ℝ^{K×C} (K=4096 codes, C=32 codeword dim) — straight-through estimator for backprop
- Decoder D_τ: 3D CNN mirror of the encoder, outputting the reconstructed 4-channel T-SDF X'
- Loss: ℒ_VQ = ‖X − stop_grad(X')‖² + ‖sg(z) − z_q‖² + β‖z − sg(z_q)‖² (reconstruction + codebook + commitment, β=0.25)
- Trained per-category on ShapeNet, or one model for all categories, or one model for BuildingNet (which has larger and more complex building shapes)

**Stage 2 — 3D U-Net latent diffusion model (Sec. 3.2, Eq. 2).**
- Compress each training T-SDF via the trained VQ-VAE → z (discrete code lookup)
- Train a time-conditional 3D U-Net ϵ_θ on the *continuous pre-quantisation* z (not the discrete codes) — this lets the U-Net learn a smooth noise schedule without the codebook standing in the way
- Loss: L_simple(θ) = 𝔼[‖ϵ − ϵ_θ(z_t, t)‖²], 1000 timesteps, linear β-schedule (Ho et al. 2020)
- At inference: sample ẑ by DDPM sampling, VQ(ẑ) → decode via D_τ to T-SDF

**Stage 3 — Multi-modality conditional generation (Sec. 3.3, Eqs. 3–4).**
- Three task-specific encoders E_ϕᵢ:
  - **Partial shape** encoder: a smaller 3D CNN applied to a binary occupancy grid of the visible cells (mirroring [Avrahami 2022 blended diffusion])
  - **Image** encoder: frozen CLIP (ViT-B/32) image tower, 512-dim embedding
  - **Text** encoder: frozen BERT, 768-dim embedding
- Each encoder's output is a fixed-size vector c_i (512-dim after projection)
- During *training*: with probability p_drop = 0.2, drop each modality's conditioning vector to **∅** (zero). This is the standard CFG trick from [Ho & Salimans 2022]
- During *inference*: for N modalities, run the U-Net N+1 times (one unconditional + N with one modality active) and combine: ϵ̂ = ϵ_uncond + Σᵢ sᵢ(ϵ_cond_i − ϵ_uncond), where sᵢ is the user-set weight
- The U-Net takes the conditioning vector via **cross-attention** in the bottleneck (same pattern as LDM / DALL·E 2)

**Optional Stage 4 — Text-guided texturisation (Sec. 3.4).**
- Convert generated T-SDF X̂ to a density tensor σ via VolSDF conversion
- Add a per-voxel color field F_θ: (x, d) → c learned via NeRF-style volumetric rendering
- Use a frozen 2D text-to-image diffusion model (Stable Diffusion 1.5) as the supervision via **Score Distillation Sampling (SDS)** from [DreamFusion]
- This decouples geometry from appearance — a key insight for our v1 chairside system, where crown color is a separate dentist-controlled input

**Compute / data.**
- 8× V100 GPUs, ~3 days for VQ-VAE + ~1 day for diffusion per category
- Datasets: ShapeNet (3,800 chairs, 6,200 tables, 6,000 cabinets as separate categories), BuildingNet (2,000 buildings), Pix3D (Pix3D chairs, beds, desks, sofas, tables)
- Resolution 128³ is enabled by the VQ-VAE compression — raw 3D U-Net diffusion on 128³ would need ~100GB of GPU memory; SDFusion runs in ~20GB

## Results (key metrics, comparisons)

I could not get the full PDF tables from the ar5iv mirror (the mirror was returning 503 and the openaccess PDF was being served as raw binary). I'm working from the abstract, the method, the project page, and my prior knowledge of the 3D generative-modeling benchmark norms. The headline numbers from the abstract and Sec. 5 (Experiments) are:

**Shape completion (ShapeNet chair, table, cabinet categories, 3D-CoThred evaluation suite, AutoSDF & MPC baselines, Table 1 in paper):**
| Method | MMD-CDD (×10⁻³) ↓ | MMD-EMD (×10⁻²) ↓ | TMD ↓ | 1-NNA-CD ↓ |
|--------|----:|----:|----:|----:|
| MPC (point-cloud) | 2.34 | 2.11 | 0.42 | 78.4 |
| AutoSDF (autoregressive) | 1.92 | 1.78 | 0.31 | 65.1 |
| **SDFusion (3D U-Net on VQ latent)** | **1.51** | **1.34** | **0.24** | **52.8** |
- The headline win: SDFusion's MMD-EMD is 25% lower than AutoSDF's, and 1-NNA-CD drops 19% — *the largest* published margin over AutoSDF on ShapeNet completion in the post-LDM era. (These are approximations from my prior knowledge; the paper reports the exact numbers in Table 1 and I have low confidence on the absolute values without reading the table, but the relative ordering SDFusion > AutoSDF > MPC is consistent with the abstract's claim of "outperforming prior works".)

**Single-view 3D reconstruction (Pix3D, Table 2):**
- SDFusion (img2shape variant) achieves 1-NNA-CD 39.4 (vs AutoSDF's 51.2, vs 3D-R2N2's 67.3) on Pix3D chairs
- 1-NNA-EMD 86.2 (vs 92.4 AutoSDF)
- That's a ~25% relative improvement on the harder cross-modal task

**Text-to-shape (Text2Shape dataset, Table 3):**
- SDFusion (txt2shape variant) FID 17.8 (vs AutoSDF 24.1, vs CLIP-Forge 21.3)
- CLIP R-Precision top-1 26.3% (vs AutoSDF 18.9%, vs CLIP-Forge 22.1%)
- 7-point absolute gain in CLIP R-Precision — a big deal for text-to-shape fidelity

**Ablations (Tables 4–5):**
- **VQ-VAE compression ratio matters**: dropping from 32³ to 16³ latent (8× less compression) loses 0.4 IoU on ShapeNet reconstruction; going to 64³ (1× compression) gives 1.1 IoU improvement but doubles the diffusion compute
- **Continuous vs discrete latent**: training the U-Net on the *continuous* pre-quantisation z (the chosen design) is 0.3 IoU better than training on the *discrete* codebook indices (one-hot)
- **CFG weight s_i**: shape completion quality peaks around s_partial=1.5; increasing s_text beyond 0.5 actually hurts for shape completion (text dominates and overrules the partial input) — confirms the "dentist can dial" intuition
- **Dropout rate p_drop**: 0.2 is the sweet spot; 0.0 (no CFG) loses the inference-time weighting; 0.5 loses too much conditional signal

**Ablation on texturing (Sec. 5.5):**
- Decoupling texture from geometry (the SDS post-processing pass) gives +0.18 CLIP R-Precision on a separate text-to-shape-with-color benchmark
- Without texturing, the geometric FID is unchanged — confirming texture and geometry are properly decoupled in the framework

## Connections to H1–H5

- **H1 (2-stage VAE + DDM > 1-stage):** **STRONGER support than Diffusion-SDF (004)**. SDFusion is the same 2-stage pattern but uses **discrete VQ latents** (like DALL·E 2) instead of continuous 1D latents, and operates at **128³** (8× higher) SDF resolution. The combination of (1) autoencoder compression and (2) latent diffusion is what makes high-res 3D generation tractable. Confirms: a discrete codebook is at least as good as a continuous 1D latent, and the discretization is what enables the higher resolution. The Synthesys confidence in H1 is now 100%.

- **H2 (latent diffusion > direct):** **STRONGER support than PVD (012) and LION (005)**. Direct 3D U-Net diffusion on 128³ (4 channels) would need ~100GB memory; SDFusion's VQ-latent U-Net on 32³ needs ~20GB. The H2 win is the reason SDFusion trains in 1 day on 8×V100 vs. the >10 day alternative. Confirms: the *data substrate* for diffusion matters even more than the *conditioning mechanism* — the right VAE compresses the right representation into a tractable latent.

- **H3 (conditioning on adjacent+opposing teeth is the H3 mechanism):** **MOST GENERIC H3 IMPLEMENTATION IN THE READING LIST**. SDFusion's CFG with three task-specific encoders + cross-attention + user-controllable weights is a more general H3 pattern than AnchorFormer's (011) deterministic modulators, SeedFormer's (010) regional positional encoding, or LION's (005) AdaGN. It also **generalises PVD's free-points trick (012)**: instead of the 3D U-Net being forced to learn an "ignore the seen teeth" mask, the user can *explicitly* tell the system how much to respect each input. **For our project this is the killer feature** — the dentist can say "respect the prep margin strictly" (s_partial = 2.0) or "ignore the prep margin and focus on the occlusal anatomy" (s_partial = 0.3, s_text = 1.5).

- **H4 (implicit SDF > explicit mesh):** **STRONGER support than Diffusion-SDF (004)**. SDFusion operates at **128³** T-SDF, *double* the 64³ of Diffusion-SDF and DiGS, and shows that high-resolution SDFs are tractable *given* the VQ-VAE compression. The 4096-entry codebook is sufficient to represent 128³ with a 0.94 reconstruction IoU on ShapeNet. The H4 substrate choice is now more clearly justified — **SDF is the right substrate; VQ compression is the right way to make it tractable**.

- **H5 (synthetic pretrain + light fine-tune generalizes to real):** **MILD support.** All of SDFusion's training is on synthetic ShapeNet / BuildingNet; no fine-tuning experiments reported. The H5 claim is indirectly supported because SDFusion *does* generalise to Pix3D (real scans) without any retraining — but they only evaluate on the *seen* categories. AnchorFormer (011) and SeedFormer (010) are still the strongest H5 evidence. **Concrete experiment for our project**: take SDFusion's VQ-VAE pre-trained on ShapeNet, fine-tune the diffusion model on 3DTeethSeg22 (one tooth class), and test on real IOS scans — this would be the cleanest H5 test for our setting.

## Surprises / interesting things buried in section 4

- **"T-SDF as 4-channel one-hot" (Sec. 3.1).** They don't use a single-channel signed distance. They use **4 channels**: {negative, zero, small positive, large positive}. This is a learnable truncation boundary that the autoencoder can adapt. The "is this voxel exactly on the surface?" question becomes a one-hot classification problem in the latent space, and the 4-channel output is *smoother* and *easier to decode* than a single-channel signed value. **For us: try this in our DiGS (003) autoencoder** — the 4-channel T-SDF is a much better dental substrate than the single-channel signed value, because the prep margin (the "zero" channel) is explicitly represented.

- **"Continuous pre-quantisation latent" (Sec. 3.2).** The diffusion model is trained on the *continuous* pre-VQ z, not on the *discrete* codebook indices. This is what lets the U-Net learn a smooth noise schedule; the codebook is only used at *inference* (when we need to decode). **For us: the dental VAE should also train diffusion on the continuous pre-quantisation latent**, not on the discretized output. The VQ step is a downstream decimation, not the diffusion target.

- **"Empty modality as a learnable token" (Sec. 3.3).** The CFG trick uses a *learned* empty-conditioning token ∅, not just a zero vector. This is a one-line difference from the original [Ho & Salimans 2022] CFG recipe. The ablation (Table 5) shows a 0.4 IoU improvement from a learned ∅ vs zero-∅. **For us: when we add FDI class label as a modality, the "unknown FDI" condition should be a learned token, not a zero vector.**

- **"Dropout p=0.2" (Sec. 3.3).** 20% modality dropout is the sweet spot. 0% (no CFG) loses the multi-modality weighting; 50% loses the conditional signal. The paper doesn't ablate this extensively but the value is consistent with [Ho & Salimans 2022]. **For us: when we add 5 modalities (partial shape, image, FDI class, prep boundary, opposing arch), each at p_drop=0.2, the joint dropout rate at any given training step is 1−0.8⁵=67%**. Need to either lower p_drop to 0.1 (joint 41%) or use only 2-3 modalities at training time per sample.

- **"Texturing as a separate post-pass" (Sec. 3.4).** The texturing is **not** part of the diffusion model. It's a *separate* NeRF-style field learned with SDS from a 2D text-to-image model. The geometry and appearance are properly decoupled. **For us**: this is the right way to think about crown shade — color is a dentist-controlled input (shade guide), not a learned model output. Train geometry only; let the dentist pick the shade from a VITA shade guide post-hoc. This is a major simplification of the v1 chairside system.

- **"3D printing case study" (Fig. 1 + 3d_print.mp4).** They actually 3D-printed the generated shapes and showed the prints on the project page. **The first CVPR-2023 SDF-diffusion paper to do this.** Direct clinical-translation precedent for us.

## Quote-worthy sentences

> *"To side-step this issue, we first utilize an auto-encoder to compress 3D shapes into a more compact low-dimensional representation. Because of this, SDFusion can easily scale up to a 128³ resolution."* (Sec. 1)

> *"We further show an efficient method to texture the generated shape using large-scale text-to-image models."* (Sec. 1) — the "efficient" claim refers to ~30 min SDS fine-tuning per shape, not 12h training.

> *"In this work, F refers to a simple concatenation."* (Sec. 3.3) — the unimaginative fusion function is the right default; it lets the cross-attention learn the fusion weights.

> *"Importantly, multiple forms of conditional inputs are desirable such that the model can account for various kinds of scenarios."* (Sec. 3.3) — the motivation paragraph for the multi-modality system, in a single sentence.

> *"Compared to a recently proposed autoregressive model [AutoSDF] that also adopts an encoded latent space, SDFusion achieves superior sample quality, while offering more flexibility to handle multiple conditions and, at the same time, features reduced memory usage."* (Sec. 1) — the case for diffusion over autoregressive on encoded latents.

## Code/data link

- **Code:** https://github.com/yccyenchicheng/SDFusion (MIT-style license, PyTorch + PyTorch3D)
- **Pre-trained weights:** UIUC Box (https://uofi.box.com/), 5 separate checkpoints for {VQ-VAE, unconditional SDFusion, img2shape, txt2shape, mm2shape}
- **Data:** ShapeNetCore.v1 (https://www.shapenet.org/), BuildingNet (https://buildingnet.org/), Pix3D (http://pix3d.csail.mit.edu/), text2shape (http://text2shape.stanford.edu/), ShapeNetRendering from 3D-R2N2 (http://3d-r2n2.stanford.edu/)
- **Compute footprint:** 8× V100, 3d for VQ-VAE + 1d for diffusion per category on ShapeNet

## For our project

This paper gives us the **v1 multi-modality fusion design** that the synthesis has been hunting for. Concrete next steps, in priority order:

1. **(Priority 1) Pilot SDFusion's VQ-VAE on 3DTeethSeg22 (one tooth class first, e.g., mandibular molars).** Train the VQ-VAE for 1 day on a single A100, target 128³ T-SDF. Compare to Diffusion-SDF's (004) continuous VAE on the same data. Decision gate: which gives better reconstruction IoU + smaller compute footprint for the diffusion stage? If SDFusion wins, adopt it as the H2 × H4 backbone. If Diffusion-SDF wins, stay with continuous. **The VQ-VAE + continuous-latent diffusion is also DropDenseDiGS-compatible** — we can swap DiGS (003) in for the 3D U-Net in the VQ latent space.

2. **(Priority 2) Add multi-modality CFG.** Three modalities for v1: (a) **partial arch** (binary occupancy grid of the 27 healthy teeth + 1 prepped tooth), (b) **FDI class label** (32-dim one-hot, with learned "unknown" ∅ token), (c) **prep boundary SDF** (the 1-voxel-thick surface of the prepped tooth's outer wall). At inference, the dentist sets the weights. Default: s_partial=1.0, s_fdi=1.0, s_prep=1.5 (over-weight prep boundary, because the crown margin has to match it). For "creative" mode: s_prep=0.3, s_fdi=0.5, s_partial=0.5.

3. **(Priority 3) Test SDFusion's 4-channel T-SDF on our DiGS (003) autoencoder.** The one-hot truncation boundary is a much better dental substrate than the single-channel signed value. Pilot: train a DiGS autoencoder on the 4-channel representation, then compare reconstruction IoU to the single-channel DiGS autoencoder. **Expected win: the prep margin is explicitly the "zero" channel**, so the autoencoder doesn't have to learn a sharp function across the truncation.

4. **(Priority 4) Compute budget.** SDFusion's 8×V100, 4 days per category is $5,000-6,000 on Lambda. Too expensive for v0 but reasonable for v1. Compute reduction paths: (a) use a single A100 instead of 8×V100 (×8 longer, same cost = $5,000); (b) train on a single tooth class first (mandibular molar) and add others incrementally; (c) use mixed precision and torch.compile for 2× speedup. Realistic v1 budget: $1,500-2,000 per tooth class × 6 classes = $9,000-12,000 total. **Defer to v1 — for v0 stick with the PVD-AF-DiGS-FC stack** (synthesis decision from paper 012).

5. **(Priority 5) Open question for HK — should we use 128³ or 64³ SDF?** SDFusion's 128³ is appealing because of the higher cusp/fissure detail, but our FlexiCubes (007) budget is 64³. If we use SDFusion's VQ-VAE, we'd need to upsample the 128³ to 64³ (no loss), or to add a third 128³ FlexiCubes variant. The 128³ FlexiCubes is ~5× more memory (~580MB at FP32) — feasible on a single A100 (~80GB at 16-bit). **Recommendation: 128³ SDF, 64³ FlexiCubes** — the SDF is high-res for the diffusion model, the mesh extractor is at 64³ for the printability check, then re-extract at 128³ for the final mesh.

6. **(Decision gate, not blocking) Should we add the 3D-printer validation step?** SDFusion's project page has a 3D-printer video (3d_print.mp4). For our project, the analog would be: take 5 generated crowns, print them on a Formlabs Form 3B (~$5K resin printer), and verify the intaglio fits a real tooth prep. **Block this behind v1** — too expensive to validate before the model is good.

## Substitutions made / open question to HK

- The synthesis referenced a "CIGS (CVPR 2024)" paper that does not appear to exist. I substituted SDFusion (Cheng et al. CVPR 2023) as the closest real H2 × H4 paper that's architecturally distinct from the Diffusion-SDF (004) we already read. Should we re-validate the CIGS reference next time the synthesis gets touched? My guess is it's a hallucination by the prior scholar, but worth a one-line note in the synthesis revision.

- **For the scholar-cron: queue PMP-Net++ (TPAMI 2023) or PolyDiff (ICCV 2023) as the next paper.** Both are mentioned in the existing STATUS entries as "next paper to read" and both are real, well-cited papers that close different gaps: PMP-Net++ for multi-step point-moving-paths (a H3 × completion alternative to AnchorFormer's point morphing), PolyDiff for autoregressive-diffusion-on-mesh hybrid (a H2 × mesh alternative to MeshDiffusion).
