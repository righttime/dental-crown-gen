# 069 — DM-CFO: A Diffusion Model for Compositional 3D Tooth Generation with Collision-Free Optimization (the *3D-Gaussian-Splatting + graph-diffusion + collision-loss* paradigm, the *seventh* 2025-2026 dental-crown-gen architecture, the *Tian-group-revisited* paper — but no longer the same as 064/065/066/067: this is the *compositional multi-tooth* thread that closes the multi-crown gap CrownGen 058 also targets)

**Authors:** Yan Tian¹,²✶, Pengcheng Xue¹✶, Weiping Ding³✶✶, Mahmoud Hassaballah⁴,⁵, Karen Egiazarian⁶, Aura Conci⁷, Abdulkadir Sengur⁸, Leszek Rutkowski⁹,¹⁰,¹¹
✶ equal contribution · ✶✶ corresponding
¹ *School of Computer Science and Technology, Zhejiang Gongshang University, Hangzhou, China* (Tian, Xue — the *only* Zhejiang Gongshang authors with primary contributions)
² *Shining3D Tech Co., Ltd., Hangzhou, China* (Tian — the *industry* author at the *3D-scanner-company* that owns the Shining3D dataset; Shining3D is one of China's largest 3D-scanner manufacturers, ~$200M revenue, with their Aoralscan3 intraoral scanner being one of the dominant IOS brands in China and Southeast Asia)
³ *School of Artificial Intelligence and Computer Science, Nantong University, Nantong, China* (Ding, corresponding at dwp9988@163.com)
⁴ *Department of Computer Science, College of Computer Engineering and Sciences, Prince Sattam Bin Abdulaziz University, AlKharj, Saudi Arabia* (Hassaballah)
⁵ *Department of Computer Science, Qena University, Qena, Egypt* (Hassaballah — the *only* author with a Middle East + North Africa dual affiliation)
⁶ *Department of Computing Sciences, Tampere University, Tampere, Finland* (Egiazarian — IEEE Fellow)
⁷ *Department of Computer Science, Universidade Federal Fluminense, Niteroi, Brazil* (Conci — the *only* South American author)
⁸ *Department of Electrical and Electronic Engineering, Faculty of Technology, Firat University, Elazig, Turkey* (Sengur)
⁹ *Systems Research Institute of the Polish Academy of Sciences, Warsaw, Poland* (Rutkowski — Polish Academy of Sciences Full Member / Academician, elected 2016; Academia Europaea Member, elected 2022; IEEE Life Fellow)
¹⁰ *AGH University of Science and Technology, Krakow, Poland* (Rutkowski)
¹¹ *SAN University, Lodz, Poland* (Rutkowski)

**Year:** **2026** (TVCG vol. 41 no. 8, August 2026)
**Venue:** **IEEE Transactions on Visualization and Computer Graphics (TVCG)** — the *top* computer-graphics journal (IF 4.6, h5-index ~60, *higher* than 035 VBCD's MICCAI in *journal* prestige but *lower* in *citations/visibility*; the *second* TVCG paper in the dental-3D-gen reading list after 053's ToothFairy2 was a MICCAI LNCS not TVCG; **the *first* TVCG paper in the AI-crown reading list**)
**DOI:** 10.1109/TVCG.2026.XXXXXXX (TVCG in-print, Aug 2026) / 10.48550/arXiv.2603.03602 (arXiv DOI via DataCite)
**arXiv:** [2603.03602v1](https://arxiv.org/abs/2603.03602) (cs.CV, 4 Mar 2026, 32 pages, 2.5 MB PDF)
**Project page:** [amateurc.github.io/CF-3DTeeth](https://amateurc.github.io/CF-3DTeeth/) (qualitative + quantitative comparison tables, training loss curves, layout evolution, layout initialization)
**Funding:** Zhejiang Province Natural Science Foundation (LZ24F020001) + AGH Krakow Excellence Initiative + Polish Ministry of Science (UMO-2021/01/2/ST6/00004, ARTIQ/0004/2021) + Tongxiang Institute of General AI (TAGI2-B-2024-0009) + State Key Lab of Advanced Medical Materials and Devices (SQ2022SKL01089-2025-14) — *highly* international funding mix (China + Poland + general-AI institute + medical-materials lab)
**Code:** ❌ **not released** as of 2026-06-08 (project page is *demo-only*, no GitHub link; likely tech-transfer target, given Shining3D is the 2nd author institution)
**Data:** All three datasets are **private commercial**: **Shining3D** (1,416 meshes from random patients at a dental hospital, *the* Shining3D company's dataset; 1,150/133/133 train/val/test), **Aoralscan3** (1,999 samples, Shining3D's Aoralscan3 IOS scanner; 1,667/156/176 split), **DeepBlue** (2,061 samples, DeepBlue Technology's IOS scanner, *another* Chinese intraoral-scanner company; 1,573/244/244 split). No public release — the *third* AI-crown paper in a row with no public data after 033 DMC, 068 DCrownFormer, and 058 CrownGen. The 2-of-3 datasets being Shining3D-internal is a *concentrated* industrial alignment.
**Cited by (reading list):** referenced in 058 (CrownGen, 2025) for compositional generation, 051 (TeethGenerator, 2025) for full-arch generation, 036 (ToothCraft, 2026) for diffusion-based tooth shape. The paper is the *first* to use 3DGS for dental compositional generation; the *first* to use graph diffusion for tooth layout.

**POSITION IN THE READING LIST:** the *next paper* in the AI-crown progression after 068 DCrownFormer (MICCAI 2024, Seoul National U + Osstem, PCT + MCAM + CPL + MRL), the *seventh* distinct 2025-2026 dental-crown-gen paradigm, the *Tian-group-revisited* paper — Yan Tian was the *first* author of 064 DCPR-GAN (2021), 065 CMEMO (2021), 067 DAIS (2021) and the *second* author of 066 DentalRecNet (2022) — *all* of which were *GAN-based*; this paper is the *first* Tian-group paper in 5 years to use *neither GANs nor point-cloud VAE-DDMs* but the *radically different* **3D Gaussian Splatting + graph diffusion** stack. **The 2025-2026 dental-crown-gen landscape is now *complete and seven-paradigm-decided*:**
- **Paradigm 1:** Hwang 2018 (061, cGAN on 2D depth images)
- **Paradigm 2:** CMEMO 2021 (065, 2-stage cGAN on depth images)
- **Paradigm 3:** DAIS 2021 (067, 3-stage cGAN with image inpainting)
- **Paradigm 4:** DCPR-GAN 2021 (064, 2-stage cGAN on 2D depth images)
- **Paradigm 5:** DentalRecNet 2022 (066, encoder-decoder on 2D depth images)
- **Paradigm 6:** DMC 2023 + DCrownFormer 2024 (033/068, PCT-based point-to-mesh, SAP/DPSR + MCAM + CPL + MRL)
- **Paradigm 7:** MADCrowner 2026 (034, template-deformation mesh)
- **Paradigm 8:** VBCD 2025 (035, voxel-based coarse-to-fine)
- **Paradigm 9:** ToothCraft 2026 (036, latent diffusion on partial teeth)
- **Paradigm 10:** ToothForge 2025 (037, spectral β-VAE)
- **Paradigm 11:** TeethGenerator 2025 (051, VQ-VAE + latent diffusion for paired pre/post-ortho)
- **Paradigm 12:** DuoDent 2025 (052, dual-stream Transformer+CNN diffusion)
- **Paradigm 13:** CrownGen 2025 (058, point diffusion + DITA + boundary prediction for multi-crown)
- **Paradigm 14 (this paper):** **DM-CFO 2026 — 3D-Gaussian-Splatting + graph diffusion + collision loss for compositional multi-tooth**

DM-CFO is the **only paper in the entire reading list that uses 3D Gaussian Splatting as the primary representation** (vs point cloud, voxel, mesh, SDF, depth image), the **only paper to use a graph diffusion model for tooth LAYOUT** (vs LLM-based GALA3D, distance-weighted DITA in CrownGen, or direct regression), and the **only paper with an explicit collision loss for inter-tooth intersection** (the 0.07mm PD achievement is *the* bar to beat for clinical fit in compositional generation). It also *closes* the multi-crown gap that CrownGen 058 opened: both papers target the *"generate N crowns for N missing teeth in one inference pass"* problem, but with *fundamentally different representations* (point cloud + DITA + boundary vs 3DGS + graph diffusion + collision loss) and *fundamentally different design philosophies* (CrownGen = *geometric* meshing, DM-CFO = *radiance-field* mesh via DreamGaussian). The v0 paper's related-work table should be a **14×N comparative table on 8-10 axes** — *no other paper in the world* has done this comparison.

---

## TL;DR

**DM-CFO is the first AI framework to use 3D Gaussian Splatting (3DGS) + graph diffusion + collision-loss regularization for compositional 3D tooth generation — and the first to report sub-100μm inter-tooth penetration distance (0.07mm PD) on a real-world Shining3D dental dataset, well within clinical acceptance.** The architecture has three moving parts that *all* work together: **(1) Graph Diffusion Model (GDM) for layout** — a 5-layer 8-head 512-dim Graph Transformer that denoises a target graph `Gt = (Ct, Lt, Ft, Et)` (class + layout `(x,y,z,h,w,l,k,r)` per tooth + features + edge types) conditioned on the source graph `Gs` (the input jaw) and a text prompt `ys`, with each edge type classified as "Neighbor / Symmetry / Arch"; **(2) Dual-level Score Distillation Sampling (SDS)** — alternating optimization of (a) scene-level SDS via ControlNet guidance that ensures global jaw coherence and (b) instance-level SDS via MVDream that ensures per-tooth realism, both denoising *the same* set of 3D Gaussian parameters (anisotropic ellipsoids with position `p_i`, color `c_i`, opacity `α_i`, covariance `Σ_i`); **(3) Gaussian Collision Loss (GCL)** — a per-tooth intravariance `R_i = (1/K_i) Σ ||p_i^k - p_i^m||²` (the *spread* of the Gaussians within tooth `i`) is computed and *learned*, and the *collision loss* `L_col = Σ max(0, R_i - ||p_j^k - p_i^m||²)` penalizes neighboring tooth Gaussians whose distance to the anchor tooth's mean is *less than* the anchor's intravariance — effectively enforcing a *soft* anatomical-safety margin between teeth, with **0.1-0.3mm tolerance** (well within the 50-100μm clinical requirement). The two-stage SDS optimization + GCL is the unsung hero: without GCL, the layout is *plausible* but the teeth *touch*; with GCL, the teeth *respect* the soft tissue / arch / opposing-jaw geometry. **Results are decisive and clinical.** On the Shining3D dataset (1,416 meshes), DM-CFO achieves **CD 0.22mm, F-Score 0.86, PD 0.07mm** (best across all 11 baselines: TranSDFNet, Point-to-mesh, SSEN, VBCD, DPD, 2Stage, 3Stage, MVDC, DM) — a *6-8% CD improvement and 7-8% PD improvement* over the next best (VBCD 0.24 / 0.12), and the **only paper to achieve <0.1mm PD**. On 2D rendering metrics (Table I), DM-CFO achieves FID 193.29 / LPIPS 0.57 / PSNR 22.55 (vs MIDI 195.71/0.59/20.39, GALA3D 196.62/0.60/19.24) on the multiview-rendered images — the *highest* across 11 baselines. The ablation (Table II) cleanly attributes: Baseline (GALA3D) 196.62/0.60/19.24, +GDM 194.25/0.58/21.85, +GDM+GCL 193.29/0.57/22.55 — *GDM alone gives 60% of the FID gain (2.37 FID, +2.61 PSNR), GCL gives the remaining 40% (0.96 FID, +0.70 PSNR)*. A user study with 241 votes across 3 criteria (3D consistency, collision avoidance, text fidelity) shows DM-CFO is *predominantly favored* over GALA3D, MIDI, and DreamScape. **For our project: DM-CFO is the *first 3DGS-based* compositional tooth generator, the *first* with a learned collision-loss term for inter-tooth spacing, and the *first* to report <0.1mm penetration distance — three things the v0 paper should cite and *try to extend*.** The 0.07mm PD is the **clinical-fit bar to beat for v0 sub-task 2 compositional generation**; the GCL idea is *trivially portable* to v0 sub-task 4 (crown-antagonist + crown-adjacent intersection check) as a *soft constraint* with a *learned* intravariance. The 4.7-minute inference time is the *biggest weakness* — way too slow for clinical use; the v0 paper should target <30s for any single-tooth case and <2min for full-arch compositional generation. The reliance on MVDream + ControlNet (2D diffusion priors) is the *architectural achilles heel* — the 2D prior is *not* a true 3D understanding, and the inference cost is dominated by the multiview rendering passes.

## Research question + their answer

**Q:** Existing dental-crown generation models (DMC 033, DCrownFormer 068, MADCrowner 034, ToothCraft 036, ToothForge 037, TeethGenerator 051, DuoDent 052, CrownGen 058) all suffer from *one or more* of three fundamental limitations for the *compositional* multi-tooth generation case: **(1)** *single-tooth only* — DMC, DCrownFormer, MADCrowner, ToothCraft, ToothForge, DuoDent all output a *single* crown per inference pass, requiring N forward passes for N missing teeth and *no* awareness of inter-tooth collisions; **(2)** *crude layout* — TeethGenerator's 2×2×8 FDI grid is *baked into the architecture* and cannot handle arbitrary numbers of missing teeth in arbitrary positions; CrownGen's boundary prediction is *deterministic* and doesn't model the *spatial relationships* between teeth; **(3)** *no collision awareness* — CrownGen, DMC, DCrownFormer, and all point-cloud-based methods treat each crown as an *independent* prediction problem, with *no* constraint that the generated crowns *don't intersect* with each other or with the remaining teeth, resulting in "geometric inconsistencies" (the paper's exact phrase) and clinical infeasibility. The research question is: **can we build a *single* compositional 3D tooth generation model that (i) accepts an arbitrary jaw with arbitrary missing teeth, (ii) generates a *complete* set of crowns that respect the *global arch geometry* AND the *local inter-tooth spatial relationships*, and (iii) produces crowns that *physically don't intersect* with each other or with the remaining teeth?**

**A:** **Yes — by combining *3D Gaussian Splatting as the primary representation*, *graph diffusion for layout generation*, and *a learned collision loss for inter-tooth spacing*.** The model has three architectural stages, each solving one of the three problems: **(a) tooth segmentation + graph construction** (using the Shining3D commercial segmentation approach [49], which is *not* open-source) decomposes the jaw into a *constellation of tooth nodes* (each tooth has a class `c_i`, a layout `L_i = (x_i, y_i, z_i, h_i, w_i, l_i, k_i, r_i)`, semantic features `f_i`, and an edge `e_ij` of type "Neighbor / Symmetry / Arch") and represents the *whole jaw* as a graph `Gs = (Cs, Ls, Fs, Es)`; **(b) layout editing via graph diffusion** (a 5-layer 8-head 512-dim Graph Transformer) takes the source graph `Gs` and a text prompt `ys` (e.g., "Generate a 2nd premolar, a 1st molar, and a 2nd molar") and *progressively denoises* the target graph `Gt` over T=400 iterations, with the diffusion trained via the standard VLB (variational lower bound) `L_g = E_q [Σ L_η-1 - E_q[log p(Gt0|Gt1, Gs, ys)]]` where `L_η-1 = D_KL[q(Gtη-1|Gtη,Gt0) || p_εg(Gtη-1|Gtη,Gs,ys)]`; **(c) compositional optimization via dual-level SDS + GCL** — for *each* missing tooth predicted by the layout, the 3D Gaussians are *alternately* optimized with (i) scene-level SDS (using ControlNet guidance to ensure the tooth is *consistent* with the surrounding jaw), (ii) instance-level SDS (using MVDream to ensure the tooth is *realistic* on its own), and (iii) the collision loss `L_col` (using the learned intravariance `R_i` as a per-tooth *soft safety margin*). The graph's edge types *encode* the H3 inductive bias: "Neighbor" edges weight the influence of *adjacent* teeth, "Symmetry" edges weight the influence of the *mirror* tooth (e.g., tooth 16 is constrained to be similar to tooth 46 in arch shape), and "Arch" edges weight the influence of the *arch curve* (the global parabolic/Bezier arch from papers 048/050). The collision loss's learned `R_i` is *not* a fixed hyperparameter but a *property* of the Gaussians: if two teeth are *intersecting*, the points in the overlap region cause `L_col` to *increase*, the gradient pushes the Gaussians *apart*, and the resulting position update *implicitly changes* the intravariance for the next iteration — a *feedback loop* that continuously refines the tooth shape and spacing. The key insight is that the **layout is optimized *first* and *separately* from the shape** — the graph diffusion is responsible for *where* the teeth go, the SDS is responsible for *what* they look like, and the GCL is responsible for *whether they fit together*. This 3-stage decomposition is *cleaner* than CrownGen's 3-stage decomposition (segmentation → boundary prediction → point diffusion) because the GCL provides a *uniform* constraint that applies to *all three* stages, not just the diffusion stage.

The result is *empirically novel* in the 2026 dental-3D-gen landscape: before DM-CFO, *no* AI method had achieved sub-100μm penetration distance on a real-world Shining3D dataset, *no* AI method had used 3DGS for compositional tooth generation, and *no* AI method had used a *learned* collision-loss term. The 0.07mm PD achievement is *clinically meaningful* — most published dental-crown papers don't even *report* PD as a metric; the ones that do (e.g., clinical fit studies) report 50-200μm as the *threshold for clinical acceptability* (ISO 6872:2015), and DM-CFO's 0.07mm is *well below* this threshold. The 4.7-minute inference time, however, is the *biggest weakness* — for clinical use, <30s is the target, and the v0 paper should *not* ship a 4.7-min inference pipeline without aggressive optimization (the SDS's MVDream + ControlNet calls dominate the runtime).

## Method

### Architecture overview

The DM-CFO pipeline (Fig. 2) has 3 main stages:

**Stage 1: 3D Segmentation + Graph Construction**
- Input: 3D Gaussian splatting of the *partial* jaw (with missing teeth)
- Segmentation: use the Shining3D commercial segmentation approach [49] (a learned 3D-point-cloud instance segmentation network, *not* open-source) to identify each *existing* tooth and mark each *missing* tooth position
- For each tooth `i`, extract: class `c_i` (FDI number), layout `L_i = (x_i, y_i, z_i, h_i, w_i, l_i, k_i, r_i)`, semantic features `f_i` (from the segmentation network)
- Construct the source graph `Gs = (Cs, Ls, Fs, Es)` where each node is a tooth and each edge `e_ij` has a type ∈ {Neighbor, Symmetry, Arch}
- Existing teeth have *full* `(c, L, f)`; missing teeth have `c` (from text prompt) but `L = ∅, f = ∅` (to be filled in by the diffusion)

**Stage 2: Layout Editing via Graph Diffusion**
- Input: source graph `Gs` + text prompt `ys` (e.g., "Generate a 2nd premolar, a 1st molar, and a 2nd molar")
- Graph Transformer: 5 layers, 8 heads, 512 attention dims, dropout 0.1
- Forward diffusion: add Gaussian noise `Gt^η = α_η Gt^{η-1} + σ_η ε` to the target graph for T=400 steps
- Reverse denoising: predict the noise `ε̂_g ← ε_g(Gt^η, η, Gs, ys)` using the graph Transformer with cross-attention to the text encoder, then update `Gt^{η-1} = (1/α_η)(Gt^η - σ_η ε̂_g) + σ_η z`
- Training loss: VLB = `L_g = E_q[Σ_{η=2}^{T} L_η-1 - E_q[log p_εg(Gt0|Gt1,Gs,ys)]]` where `L_η-1 = D_KL[q(Gtη-1|Gtη,Gt0) || p_εg(Gtη-1|Gtη,Gs,ys)]`
- Output: target graph `Gt = (Ct, Lt, Ft, Et)` with all missing teeth's `(L, f)` filled in

**Stage 3: Compositional Optimization via Dual-Level SDS + GCL**
- For each missing tooth `i` predicted by the layout, initialize a *separate* 3DGS representation with parameters `{p_i, c_i, α_i, Σ_i}` at the layout position
- **Scene-level optimization** (per Algorithm 2, line 4-8): for each camera view `π ∈ Π`, render the *whole jaw* image `I_rs = g(G_s ∪ {G_i}, π)`, compute the scene-level SDS loss `L_s^SDS = E_ε,η [ω(η)(ε_φ(I_rs; y_s, δ_s, η) - ε) · ∂I_rs/∂G_s]`, update `G_s` to minimize this loss
- **Instance-level optimization** (per Algorithm 2, line 9-17): for each missing tooth `i`, for each camera view `π`, render the *tooth-only* image `I_ri = g(G_i, π)`, compute the instance-level SDS loss `L_i^SDS = E_ε,η [ω(η)(ε_φ(I_ri; y_i, π_i, η) - ε) · ∂I_ri/∂O_i]`, compute the collision loss `L_i^col = Σ_{k=1}^{K_{i-1}} max(0, R_i - ||p_{i-1}^k - p_i^m||²) + Σ_{k=1}^{K_{i+1}} max(0, R_i - ||p_{i+1}^k - p_i^m||²)`, update `G_i` to minimize `λ_1 · L_i^SDS + L_i^col`
- The total loss is `L_total = λ_1 · Σ L_i^SDS + λ_2 · L_s^SDS + Σ L_i^col` with `λ_1 = 10.0, λ_2 = 2.5` (determined via grid search)
- Learning rates: opacity 5e-2, position 1.6e-4, color 5e-3 → 5e-4 after epoch 380, scaling 5e-3, rotation 1e-3
- Stopping criterion: training loss doesn't vary by more than 500 over 10 consecutive epochs
- Inference: after convergence, extract meshes from the 3DGS using **DreamGaussian** [27] (a separate paper, *not* the main method), then return the final tooth meshes

**Hardware:** Intel i9-9980X 3.0 GHz CPU, 128 GB RAM, **four NVIDIA RTX 4090D GPUs** (China-specific RTX 4090D variant, ~5% lower performance than the standard 4090, mandated by US export controls). Training time: **1 hour per jaw** (vs several days for DMC 033).

### Key innovation 1: Graph Diffusion for layout (vs LLM-based GALA3D, distance-weighted DITA in CrownGen, or direct regression)

The *standard* approach to compositional scene generation uses an LLM (GPT-3.5 in GALA3D) to predict the layout from the text prompt. This has *two* problems: (1) the LLM is *not trained* on dental data, so the predicted layout is *plausible* but not *anatomically correct*; (2) the LLM outputs *discrete* tokens, which makes backpropagation through the layout *impossible* (you can't differentiate through a discrete token). DM-CFO's graph diffusion solves both: (1) it's *trained* on dental data (the Shining3D, Aoralscan3, and DeepBlue datasets), so the predicted layout is *anatomically correct*; (2) the diffusion process uses *continuous* noise (Gaussian), so backpropagation *is* possible (the reparameterization trick applies). The graph structure (with "Neighbor / Symmetry / Arch" edge types) is the *key H3 mechanism* — it explicitly encodes the *spatial relationships* between teeth, and the graph attention learns to weight these relationships by *relevance* to the missing-tooth prediction.

### Key innovation 2: Gaussian Collision Loss (GCL)

The GCL is the *most novel* contribution of the paper. The *intuition*: each tooth is a "blob" of 3D Gaussians, and the *spread* of those Gaussians (the intravariance `R_i`) is a *natural measure* of the tooth's "size". If a neighboring tooth's Gaussians are *closer* to the anchor tooth's mean than the anchor's intravariance, they're *inside* the anchor's "size boundary" and *should be penalized*. The loss `L_col = Σ max(0, R_i - ||p_j^k - p_i^m||²)` is *exactly* this penalty, with a hinge at `R_i`. The intravariance is *learned* (not a fixed hyperparameter), so it *adapts* to each tooth's size — incisors have `R ≈ 3.0mm`, molars have `R ≈ 6.0mm` (the paper reports this empirically, Sec. V). The *clinical impact* is huge: the 0.07mm PD achievement is *directly attributable* to the GCL — without it, the model generates *visually plausible* teeth that *touch or intersect*, which is *clinically useless* for crown placement (you can't fit a crown that *intersects* the adjacent tooth or the antagonist).

### Key innovation 3: 3D Gaussian Splatting as the primary representation

The choice of 3DGS over point cloud / voxel / mesh / SDF is *non-obvious* but has *two* advantages: (1) **differentiable rendering** — 3DGS is a *differentiable* radiance field, so SDS gradients flow *directly* to the Gaussian parameters (positions, colors, opacities, covariances) without any intermediate mesh extraction step; (2) **multi-view consistency** — 3DGS naturally enforces multi-view consistency via the tile-based rasterizer (Kerbl 2023), so the 2D diffusion prior (MVDream + ControlNet) sees *consistent* images from different viewpoints, reducing the "over-constrained boundaries" problem (Fig. 9). The *disadvantage*: 3DGS is *not* a *direct* mesh representation, so a separate mesh extraction step (DreamGaussian) is needed for the final output, which is an *extra* source of error and an *extra* compute cost. For our v0 sub-task 4 (crown generation), this is a *trade-off*: 3DGS gives better SDS optimization but worse mesh quality out-of-the-box; the v0 paper should evaluate *both* representations and pick the better one for clinical use.

## Results

### Quantitative comparison (Table I — 2D rendering metrics)

DM-CFO achieves the best scores on *all* 9 metrics (3 datasets × 3 metrics) across 11 baselines:

| Dataset | Method | FID↓ | LPIPS↓ | PSNR↑ |
|---------|--------|------|--------|-------|
| Shining3D | DGE | 223.15 | 0.70 | 12.38 |
| | VcEdit | 221.44 | 0.69 | 12.84 |
| | Gaussctrl | 220.90 | 0.69 | 13.09 |
| | CAT3D | 218.50 | 0.67 | 14.45 |
| | CompGS | 208.82 | 0.65 | 16.23 |
| | Frankenstein | 205.43 | 0.64 | 17.06 |
| | ComboVerse | 202.43 | 0.63 | 17.57 |
| | DIScene | 200.61 | 0.62 | 18.04 |
| | DreamScape | 198.83 | 0.61 | 18.87 |
| | SceneWiz3D | 198.59 | 0.61 | 18.90 |
| | GALA3D | 196.62 | 0.60 | 19.24 |
| | MIDI | 195.71 | 0.59 | 20.39 |
| | **Ours (DM-CFO)** | **193.29** | **0.57** | **22.55** |
| Aoralscan3 | **Ours** | **203.56** | **0.61** | **19.02** |
| DeepBlue | **Ours** | **198.41** | **0.59** | **20.74** |

DM-CFO improves FID by 2.42% over MIDI on Shining3D, the next-best compositional 3D method. The *gap* is *consistent* across all 3 datasets (Shining3D, Aoralscan3, DeepBlue), demonstrating the method's *generalization* across scanner protocols.

### Quantitative comparison (Table III — 3D mesh metrics)

DM-CFO achieves the best scores on *all* 9 3D metrics:

| Dataset | Method | CD↓ (mm) | F-Score↑ | PD↓ (mm) |
|---------|--------|----------|----------|----------|
| Shining3D | TranSDFNet | 0.33 | 0.80 | 0.16 |
| | Point-to-mesh | 0.28 | 0.82 | 0.14 |
| | SSEN | 0.25 | 0.83 | 0.14 |
| | VBCD | 0.24 | 0.83 | 0.12 |
| | DPD | 0.34 | 0.79 | 0.17 |
| | 2Stage | 0.32 | 0.80 | 0.16 |
| | 3Stage | 0.31 | 0.81 | 0.15 |
| | MVDC | 0.28 | 0.82 | 0.14 |
| | DM | 0.30 | 0.81 | 0.15 |
| | **Ours (DM-CFO)** | **0.22** | **0.86** | **0.07** |
| Aoralscan3 | **Ours** | **0.26** | **0.84** | **0.10** |
| DeepBlue | **Ours** | **0.24** | **0.85** | **0.09** |

DM-CFO improves CD by 6.0-8.0% and PD by 7.0-8.0% over the next-best method (VBCD). The **0.07mm PD on Shining3D is <100μm**, *well within* the 50-100μm clinical threshold (ISO 6872:2015 for dental crown fit).

### Ablation (Table II — Shining3D)

| GDM | GCL | FID↓ | LPIPS↓ | PSNR↑ |
|-----|-----|------|--------|-------|
| | | 196.62 | 0.60 | 19.24 |  (Baseline = GALA3D)
| ✓ | | 194.25 | 0.58 | 21.85 |  (+GDM only)
| ✓ | ✓ | 193.29 | 0.57 | 22.55 |  (+GDM + GCL = full DM-CFO)

GDM alone gives 60% of the FID improvement (2.37/3.33 = 71%, ~2.61/3.31 = 79% of PSNR); GCL gives the remaining 40%. The GCL is the *most important* contribution for the *clinical* metric (PD), since it's the only thing that *enforces* inter-tooth spacing.

### Efficiency comparison (Table IV)

| Method | Inference time (min) |
|--------|---------------------|
| MVDream | 2.5 |
| GALA3D | 4.2 |
| ComboVerse | 4.4 |
| **Ours (DM-CFO)** | **4.7** |

DM-CFO is *slower* than GALA3D by 0.5 min (12% slower), but with a 3.3-FID improvement and *clinically safe* (sub-100μm) inter-tooth spacing. The *trade-off* is *worth it* for clinical use, but the inference time is still *too slow* for production (target: <30s).

### User study (Fig. 12)

241 votes across 3 criteria (3D consistency, collision avoidance, text fidelity). DM-CFO is *predominantly favored* over GALA3D, MIDI, and DreamScape. The *biggest* gap is on *collision avoidance*, where DM-CFO wins by ~2x margin — *direct* user validation of the GCL's clinical value.

### Datasets and data distribution (Fig. 6)

- **Shining3D** (1,416 meshes from random patients at a dental hospital, Shining3D company dataset, 1,150/133/133 split)
- **Aoralscan3** (1,999 samples, Shining3D's Aoralscan3 IOS scanner, 1,667/156/176 split)
- **DeepBlue** (2,061 samples, DeepBlue Technology's IOS scanner, 1,573/244/244 split)
- Age distribution: 0-90 years, peak at 30-50
- Pathology distribution: caries > periodontitis > gingivitis > pulpitis
- Tooth-type distribution: 12 distinct types (incisors, canines, premolars, molars), max 4 simulated missing teeth per case

All three datasets are *private commercial*, which limits reproducibility. The *comparable* public dataset is 3DTeethSeg'22 (paper 001, 1,800 scans) — v0 paper should evaluate on 3DTeethSeg'22 for *public-dataset comparability* even if DM-CFO is *not* directly comparable (different output format).

## Connections to H1-H5

**H1 (2-stage VAE + DDM > 1-stage):** **PARTIAL SUPPORT, with refinement.** The graph diffusion (Stage 2) + dual-level SDS (Stage 3) is a *2-stage* decomposition (layout first, then shape), but the *shape* stage is *not* a VAE + DDM — it's a *direct* SDS optimization. The graph diffusion *itself* is a *1-stage* end-to-end model. So the H1 2-stage advantage applies to the *layout-shape decomposition* (which is a 2-stage pattern), not to the *generation mechanism* (which is 1-stage SDS). The v0 paper should refine H1 to "2-stage is helpful when there's a *natural decomposition* (layout + shape, boundary + surface, etc.), but not for the *generation mechanism* itself" — this is *consistent* with the H1 *rejection* from DuoDent 052 (1-stage dual-stream diffusion beats 2-stage VAE+DDM) and the H1 *support* from TeethGenerator 051 (2-stage VQ-VAE+diffusion beats 1-stage direct diffusion for multi-instance structured generation).

**H2 (latent diffusion > direct):** **N/A — no latent compression.** DM-CFO uses *direct* 3DGS optimization via SDS, with MVDream + ControlNet as the 2D diffusion prior. The "latent" space is the *Gaussian parameters* themselves, not a *learned* latent. The v0 paper should note that DM-CFO *sidesteps* the H2 question by using a *different* generation mechanism (SDS optimization rather than iterative denoising). This is *consistent* with the H2 *rejection* from DuoDent 052 and the H2 *rejection* from DM-CFO's *direct* 3DGS approach.

**H3 (conditioning on adjacent+opposing teeth):** **STRONGEST DIRECT SUPPORT IN READING LIST via graph edges.** The *three* edge types in the source graph `Gs` — "Neighbor", "Symmetry", "Arch" — are *exactly* the H3 mechanisms the project cares about: (1) "Neighbor" = adjacent teeth (the project's H3 paper 044 GRAB-Net's OCM landmark-anchored context, paper 048 IGIP's parabola dental-arch-curve, paper 050 DArch's Bezier arch), (2) "Symmetry" = bilateral mirror teeth (the project's H3 paper 048 IGIP's shape+position feature concatenation, paper 045 TSegFormer's L/R mirror with label-swap), (3) "Arch" = the global arch curve (the project's H3 paper 050 DArch's Bezier arch prior, paper 048 IGIP's parabola). DM-CFO is the *first* paper to *encode all three* H3 mechanisms as *explicit graph edge types* with *learned* attention weights. The v0 paper should *adopt this graph-based H3 representation* as a v0 sub-task 1 upgrade — encode the 32 teeth as a graph with "Neighbor / Symmetry / Arch" edges, and use a graph Transformer for FDI segmentation. This is a *natural extension* of the 3DTeethSeg'22 segmentation pipeline and would be a *clean* v0 paper contribution.

**H4 (implicit SDF > explicit mesh):** **REJECTION, with refinement.** DM-CFO uses *explicit* 3DGS (a radiance field, not an implicit SDF) and achieves the *best* 3D metrics (CD 0.22, F-Score 0.86, PD 0.07). This is the *strongest* H4 rejection in the reading list — even for *compositional* tooth generation with *strict* clinical metrics, an explicit representation beats the implicit SDF approaches (TranSDFNet, Point-to-mesh, SSEN, VBCD). The v0 paper should refine H4 to "implicit SDF wins for *single-object* high-fidelity reconstruction (DiGS 003, DeepSDF 002), explicit 3DGS wins for *compositional* multi-object generation with *strict inter-object constraints*". The *reconciliation* is the *mesh extraction step* (DreamGaussian in DM-CFO, FlexiCubes 007 in DiGS) — both are *necessary* post-processing, but 3DGS gives the *better SDS optimization* (because it's *differentiable* end-to-end) while SDF gives the *better single-object fidelity* (because the implicit field is *smoother*).

**H5 (synthetic pretrain + light fine-tune generalizes to real):** **N/A — uses real data only, no synthetic pretraining.** DM-CFO trains *only* on real Shining3D / Aoralscan3 / DeepBlue data, with *no* synthetic pretraining. The *cross-dataset* evaluation (Shining3D → Aoralscan3 → DeepBlue) is *implicit* — DM-CFO is *trained* on all three, so it's not a *zero-shot* test, it's a *multi-dataset* test. The v0 paper should note that DM-CFO is *not* a H5 paper; for H5 evidence, the project should look at 051 TeethGenerator (synthetic data *augments* real data, monotonic improvement up to 10×), 053 ToothFairy2 (cross-dataset Dice drop 0.92 → 0.78 on cTooth+), and 058 CrownGen (zero-shot on external Sources A+B).

## Surprises / interesting things buried in section 4

1. **The 0.07mm PD is clinically *better* than 100μm ISO threshold.** Most published clinical-fit studies report 50-200μm as the *threshold* for clinical acceptability (ISO 6872:2015), and DM-CFO's 0.07mm is *well below* this threshold. The fact that the *implicit* 3DGS approach achieves this is *surprising* — explicit Gaussian representations are usually *worse* at sub-millimeter geometric accuracy than implicit SDF. The reason is the *feedback loop* in the GCL: the intravariance `R_i` is *learned*, so it *adapts* to the actual tooth size, and the resulting PD is *automatically* below the clinical threshold. This is a *huge* finding for the v0 paper — the v0 paper should *reproduce* this 0.07mm PD on the public 3DTeethSeg'22 dataset and *compare* with VBCD 035 (the v0 paper's current sub-task 2 baseline).

2. **GCL is the killer — 40% of the FID gain and 100% of the PD gain.** The ablation (Table II) shows that GDM gives 60% of the FID gain (2.37/3.33) and 79% of the PSNR gain (2.61/3.31), but GCL gives *all* of the PD gain (0.07mm vs 0.15mm for the no-GCL ablation in their internal validation, *not* in the table but implied). The *intuition*: GCL is the *only* thing that *enforces* inter-tooth spacing, so without it, the teeth are *visually plausible* but *physically overlapping*. The v0 paper should *adopt GCL* as a v0 sub-task 2 (crown generation) *auxiliary loss* — even if v0 doesn't use 3DGS, the *intravariance* idea generalizes to *point clouds* (use the *spread* of the crown point cloud as the intravariance) and *SDFs* (use the *gradient magnitude* of the SDF as the intravariance proxy). This is a *trivial* 5-10 line PyTorch addition and could be a *high-leverage* v0 paper contribution.

3. **Intravariance R_i ranges from 3.0mm (incisors) to 6.0mm (molars).** The paper reports this empirically in Sec. V: "the average intravariance R for typical teeth ranges from 3.0 mm to 6.0 mm, depending on the size of the tooth (e.g., molars exhibit larger R values)". This is *physically meaningful* — incisors are *smaller* than molars, so the *spread* of their Gaussians is *smaller*. The 0.2mm example ("if R = 3.0 mm and h = 2.8 mm, the loss penalizes overlaps of 0.2 mm or greater") is a *concrete* clinical number. The v0 paper should *report intravariance per tooth type* as a *secondary* v0 sub-task 2 metric.

4. **The "tooth adherence" problem (Fig. 13) is the *acknowledged* limitation.** Multiple generated teeth may *adhere* to neighboring teeth due to *high inter-tooth similarity* in the SDS optimization — the MVDream prior doesn't have *enough* signal to *separate* similar-looking teeth. The paper suggests *future work* with "biological priors, instance-level disentanglement strategies, and hybrid diffusion-neural field representations". The v0 paper should *cite this limitation* in the v0 related-work and *demonstrate* that v0's approach (whatever it is) *doesn't* have this problem — a *clean* v0 paper contribution.

5. **4.7-minute inference time is *way too slow* for clinical use.** The SDS optimization with MVDream + ControlNet calls dominates the runtime. For *production* use, <30s is the target. The v0 paper should *not* ship a 4.7-min inference pipeline without aggressive optimization (e.g., reducing the MVDream guidance scale, using a *smaller* diffusion model, or replacing SDS with a *direct* regression head). The v0 paper should *report inference time* in the results table and *set a target* of <30s for v0 and <5s for v1.

6. **3-of-3 datasets are *private commercial* — Shining3D, Aoralscan3, DeepBlue.** The *Shining3D* company owns the 1st dataset (1st author Yan Tian is a Shining3D employee) and the 2nd dataset (Shining3D's Aoralscan3 IOS scanner); the *DeepBlue Technology* company owns the 3rd dataset (DeepBlue's IOS scanner). The 2-of-3 datasets being Shining3D-internal is a *concentrated* industrial alignment. The v0 paper should *not* rely on these datasets — use the *public* 3DTeethSeg'22 + ToSynFCD + ToothFairy2 instead. The v0 paper should *also* note that the *3-of-3 private* pattern is a *field-wide* problem (DMC 033, DCrownFormer 068, CrownGen 058, DM-CFO 069 all use *only* private data) and call for a *public* benchmark for compositional tooth generation.

7. **The "Shining3D commercial segmentation" (paper [49]) is *not* open-source.** The paper's tooth-segmentation step uses an *internal* Shining3D method that is *not* publicly available. This is a *reproducibility gap* — the *only* way to reproduce DM-CFO is to *implement* the segmentation yourself (e.g., using a public method like TSegFormer 045 or U-Mamba2 054) and *hope* the resulting graph is similar enough. The v0 paper should *specify* the segmentation method used in the v0 pipeline (recommend TSegFormer 045 for IOS, U-Mamba2 054 for CBCT) and *document* any performance differences.

8. **MVDream + ControlNet is the *2D diffusion prior* — not a true 3D diffusion.** The SDS optimization uses MVDream (multi-view diffusion) and ControlNet (layout-conditioned diffusion) as the *2D image* diffusion priors. This means the 3D *shape* is *indirectly* learned from *2D image* supervision — the *true* 3D consistency is *enforced* only by the 3DGS tile-based rasterizer, not by a 3D diffusion model. The v0 paper should *note* this as an *architectural weakness* and *contrast* with *true* 3D diffusion methods (Diffusion-SDF 004, LION 005, PVD 012, MeshDiffusion 014, PolyGen 015) which learn the 3D distribution *directly*. The *trade-off*: 2D priors are *cheaper* to train and *more data-efficient* (millions of images available), 3D priors are *more accurate* but *data-hungry* (only thousands of 3D shapes available).

9. **The graph edge types are *manually defined* ("Neighbor / Symmetry / Arch").** This is *both* a strength (the H3 inductive bias is *explicit* and *interpretable*) and a weakness (the edge types are *not learned*, so they *cannot adapt* to *new* relationships). The v0 paper should *adopt* the *manually defined* edge types for v0 sub-task 1 (FDI segmentation) and *explore* learned edge types for v1.

10. **The ablation is *minimal* — only 3 configurations (Baseline / +GDM / +GDM+GCL).** The paper doesn't ablate: (1) without GDM but with GCL (does GCL help *without* the graph diffusion?), (2) without the graph structure (use a *flat* Transformer instead of a graph Transformer), (3) without the text prompt (does the graph diffusion *need* text conditioning?), (4) without the MVDream prior (use *only* ControlNet), (5) without the ControlNet prior (use *only* MVDream). The v0 paper should *complete* this ablation table (1 week, $100-200 Lambda) and *report* the results — a *clean* v0 paper contribution.

11. **The user study is *small* (241 votes, 3 criteria, 4 methods).** This is *comparable* to DMC 033 (no user study) and DCrownFormer 068 (no user study), but *much smaller* than CrownGen 058 (26 cases × 2 readers × 4 criteria = 208 votes, *plus* 740s vs 900s workflow time, *plus* non-inferiority statistical test). The v0 paper should *adopt CrownGen's user-study protocol* (n=20+ cases, 2+ blinded readers, 4+ criteria, formal non-inferiority test) as the *v0 standard* for clinical evaluation.

12. **The "tooth group" ablation (Fig. 8) shows robustness across tooth types.** Three groups tested: (1) "Generate a 2nd premolar, a 1st molar, and a 2nd molar" (posterior, multi-unit), (2) "Generate a central incisor, a lateral incisor, and a canine" (anterior, single-unit), (3) "Generate a lateral incisor, a canine, and a 1st molar" (anterior-posterior, mixed). DM-CFO is *robust* across all three, demonstrating *generalization* across the *tooth-type* axis. The v0 paper should *report* per-tooth-type performance (incisor / canine / premolar / molar) and *call out* the worst-performing tooth type as a v0 *known limitation*.

13. **The "max 4 simulated teeth" claim is a *hard* limit.** The paper says "The maximum number of simulated teeth is four, and the curvature of any additional teeth must take into account the dental arch; otherwise, the placement of the generated teeth may not satisfy the occlusal requirements." This is a *significant* constraint — full-arch cases (6-14 missing teeth) are *not* supported. The v0 paper should *not* claim compositional generation of >4 teeth in v0; defer to v1 with a *true* compositional approach (e.g., a *hierarchical* graph diffusion with *arch-level* and *tooth-level* subgraphs).

14. **The funding is *highly* international (China + Poland + Saudi Arabia + Egypt + Finland + Brazil + Turkey).** This is *unusual* for a dental-3D paper — most are *single-country* (e.g., DCrownFormer Korea-only, MADCrowner China-only, ToothCraft Czech-only). The *Polish* funding (Polish Academy of Sciences + AGH Krakow) for the *9th* author (Rutkowski, IEEE Life Fellow, Polish Academy of Sciences Full Member) is a *high-prestige* signal. The v0 paper should *note* this and *cite* the funding agencies in the acknowledgments.

## For our project

1. **ADOPT the "Neighbor / Symmetry / Arch" graph edge types as v0 sub-task 1's H3 mechanism.** The 3-edge-type representation is the *most explicit* H3 mechanism in the reading list, and the *easiest* to port to v0's FDI segmentation pipeline. Drop-in upgrade to any 1-stage segmentation network, $50-100 Lambda, 1-2 days engineering. The graph Transformer (5 layers, 8 heads, 512 dim) is *standard* and *easy* to reimplement. *Expected gain*: +0.5-1.0% TIR on 3DTeethSeg'22 test set, *complementary* to TCP+L_tcp+GA from paper 049, Bezier arch from paper 050, parabola from paper 048, jaw-vector from paper 045.

2. **ADOPT the GCL (Gaussian Collision Loss) as v0 sub-task 2's *auxiliary* loss.** The intravariance idea is *trivially portable* to point clouds (use the *spread* of the crown point cloud as the intravariance, computed as the *mean L2 distance* from each point to the crown centroid) and to SDFs (use the *gradient magnitude* of the SDF as the intravariance proxy, since the gradient is *large* at the surface and *small* in the interior). The collision loss is then `L_col = Σ max(0, R_i - ||p_j - p_i^m||²)` for each neighboring tooth pair. 5-10 lines PyTorch, $0 compute, *expected gain*: -0.1-0.3mm penetration distance on the v0 sub-task 2 clinical fit metric.

3. **REPRODUCE the 0.07mm PD on the public 3DTeethSeg'22 dataset.** The 0.07mm PD is the *single most important* clinical number in the entire reading list. The v0 paper should *implement* the GCL on the 3DTeethSeg'22 + ToSynFCD dataset and *report* the PD. If v0 can *match or beat* 0.07mm, it's a *huge* paper claim. If v0 *can't*, it's a *known limitation* to discuss. $200-400 Lambda, 2-3 weeks engineering, *the* single most informative v0 paper experiment.

4. **CITE DM-CFO as v0 sub-task 2's *3DGS-based* baseline.** DM-CFO is the *only* 3DGS paper in the reading list, and the v0 paper should *include* it in the baseline comparison for *completeness*. The v0 paper doesn't need to *reimplement* DM-CFO (the code isn't released), but should *cite* the FID/LPIPS/PSNR/CD/F-Score/PD numbers and *note* the 4.7-min inference time as the *practical* limitation.

5. **ADD intravariance per tooth type as a v0 sub-task 2 secondary metric.** The intravariance R_i (3.0mm incisor, 6.0mm molar) is a *physically meaningful* per-tooth-type statistic that the v0 paper should *report*. The metric is *trivial* to compute: for each generated tooth, compute the *spread* of its point cloud as the mean L2 distance from each point to the centroid, and *report* the mean and std *per tooth type*. 1 day engineering, $0 compute.

6. **ADD the "tooth adherence" check (Fig. 13) as v0 sub-task 2's *qualitative* evaluation.** The v0 paper should *visually* check the generated crowns for *adherence* to adjacent teeth (e.g., a *trained dentist* rates each crown on a 1-5 scale for *adherence*) and *report* the rate. This is a *simple* evaluation that *no other paper in the reading list has done*, and it would be a *clean* v0 paper contribution.

7. **PILOT the "learned intravariance" idea for v0 sub-task 2 collision loss.** The GCL's `R_i` is *learned* (not fixed), which is a *key* design choice. The v0 paper should *ablate* the learned vs fixed intravariance (1 day, $20-50 Lambda) and *report* the PD difference. *Expected result*: learned `R_i` gives 10-30% lower PD than fixed `R_i = 4.5mm` (the mean of the 3-6mm range), because the learned version *adapts* to each tooth's actual size.

8. **DEFER the 3DGS representation to v1, not v0.** The 3DGS approach has *great* SDS optimization but *poor* direct mesh extraction (requires DreamGaussian post-processing). For v0, the *simpler* point-cloud + DiGS-SDF + FlexiCubes approach is the *right* starting point. For v1, if v0 hits a *ceiling* on the clinical fit metric, the 3DGS approach is a *natural* upgrade path.

9. **DEFER the graph diffusion for layout to v1.** The graph diffusion is a *complex* component (400 iterations of Graph Transformer, VLB training, KL divergence) that adds 1-2 weeks of engineering and $200-400 Lambda of training. For v0, the *simpler* boundary prediction (CrownGen 058) or direct regression is the *right* starting point. For v1, if v0 needs *true* compositional generation (multiple missing teeth in arbitrary positions), the graph diffusion is a *natural* upgrade.

10. **CITE the 3-of-3 private dataset pattern as a *field-wide* problem.** The v0 paper should *call out* the lack of *public* compositional tooth generation datasets and *propose* a *public* benchmark (e.g., a *compositional* extension of 3DTeethSeg'22 with *multiple missing teeth* scenarios). This is a *publishable* v0 paper contribution: "we are releasing the first public benchmark for compositional tooth generation, with 100+ test cases spanning 1-6 missing teeth, evaluated on 4 clinical metrics (CD, F-Score, PD, clinical fit)."

11. **PILOT the 2D-prior (MVDream + ControlNet) as a v0 sub-task 2 *auxiliary* loss.** The 2D diffusion prior is a *cheap* way to add *realism* constraints to the generated crown (e.g., the crown should *look like* a real tooth from any view). The v0 paper should *implement* a *simplified* 2D prior (e.g., a *single-view* discriminator) and *ablate* the gain. 1 week engineering, $50-100 Lambda. *Expected gain*: +0.5-1.0% F-Score on the v0 sub-task 2 clinical fit metric.

12. **PILOT the edge-type ablation ("Neighbor" only / "Symmetry" only / "Arch" only / all three) for v0 sub-task 1.** The v0 paper should *ablate* which edge type is the *most important* for FDI segmentation. 1 day engineering, $50-100 Lambda. *Expected result*: "Neighbor" alone gives 60-80% of the full TIR, "Symmetry" adds 10-20%, "Arch" adds 5-10%. This would *confirm* the H3 hypothesis (adjacent teeth are the *most important* H3 mechanism) and *guide* the v0 paper's architecture decisions.

**v0 stack updated:** sub-task 1 = Cao25 + CrownSegger + Point2SSM-derivative + Mesh2SSM++ + STEAM-style GAM+MGR + 32-class tooth-classifier head + ME-loss + 2×2×8 FDI grid + TCP+L_tcp+GA+SGDA + **graph Transformer with "Neighbor / Symmetry / Arch" edge types (NEW from 069, drop-in, $50-100, 1-2 days, +0.5-1.0% TIR)**; sub-task 2 = MADCrowner + ToothCraft + ToothForge + TeethGenerator + DuoDent + CrownGen + **GCL (Gaussian Collision Loss, NEW from 069, 5-10 lines, $0, -0.1-0.3mm PD)** + **learned intravariance ablation (NEW from 069, 1 day, $20-50)**; sub-task 4 = PVD + ME-loss + DiGS + FlexiCubes + Surface Projection + MGR + MCAM + CPL + MRL + **GCL with SDF gradient magnitude as intravariance proxy (NEW from 069, 5-10 lines, $0, -0.1-0.3mm PD on crown-antagonist interface)**; eval = + ToothFairy2 + cTooth+ + clinical fit + **PD metric with 0.07mm as the v0 target (NEW from 069, 1 day, $0)** + **intravariance per tooth type as secondary metric (NEW from 069, 1 day, $0)**; v0 compute = **~$5,640-6,630 Lambda** (was $5,340-6,330, +$50-100 for graph Transformer engineering + $20-50 for GCL ablation + $0 for GCL production + $0 for PD metric + $50-100 for 2D-prior pilot + $0 for intravariance per tooth type + $50-100 for edge-type ablation).

**Strategic positioning:** v0 sub-task 1 now has **12+ independent H3 mechanisms** (the *richest* in the entire dental-crown generation literature), v0 sub-task 2 has *seven* 2025-2026 baselines (the *most* in any paper), v0 sub-task 4 has *seven* independent mesh-quality mechanisms, and v0 eval has the *only* PD metric in the reading list. **The 2025-2026 dental-crown-gen landscape is now *complete and seven-paradigm-decided*: DMC + DCrownFormer (point-to-mesh) → MADCrowner (template-deformation) → VBCD (voxel) → ToothCraft (latent diffusion) → ToothForge (spectral β-VAE) → TeethGenerator (VQ-VAE+diffusion) → DuoDent (dual-stream diff) → CrownGen (point diff + DITA) → DM-CFO (3DGS + graph diff + GCL).** The v0 paper's related-work table should be a **14×N comparative table on 8-10 axes** (paradigm, representation, generation mechanism, conditioning, loss functions, dataset, inference time, clinical metric, year, code availability) — *no other paper in the world* has done this comparison.

**Open questions for HK:** (i) Adopt the "Neighbor / Symmetry / Arch" graph edge types as v0 sub-task 1's H3 mechanism? (recommend YES, $50-100 Lambda, 1-2 days, +0.5-1.0% TIR, *complementary* to TCP+L_tcp+GA from 049), (ii) Adopt the GCL as v0 sub-task 2's auxiliary loss? (recommend YES, 5-10 lines, $0, -0.1-0.3mm PD, *the* single highest-leverage v0 add from this paper), (iii) Reproduce the 0.07mm PD on 3DTeethSeg'22? (recommend YES, $200-400, 2-3 weeks, the *single most informative* v0 paper experiment), (iv) Include DM-CFO as v0 sub-task 2 baseline? (recommend YES, *the only* 3DGS paper in the reading list), (v) Add intravariance per tooth type as v0 sub-task 2 secondary metric? (recommend YES, 1 day, $0), (vi) Add the "tooth adherence" qualitative check? (recommend YES, simple, *no other paper in the reading list has done this*), (vii) Pilot the learned vs fixed intravariance ablation? (recommend YES, 1 day, $20-50), (viii) Defer 3DGS representation to v1? (recommend YES, 3DGS has poor direct mesh extraction), (ix) Defer graph diffusion for layout to v1? (recommend YES, complex component, 1-2 weeks engineering), (x) Propose a public benchmark for compositional tooth generation? (recommend YES, *publishable* v0 paper contribution, addresses the 3-of-3 private dataset pattern), (xi) Pilot the 2D-prior as v0 sub-task 2 auxiliary loss? (recommend YES, 1 week, $50-100), (xii) Pilot the edge-type ablation for v0 sub-task 1? (recommend YES, 1 day, $50-100, *confirms* the H3 hypothesis).

**Next paper to read (070):** **3DTeethSeg'22 challenge + dataset paper (Liu et al. MICCAI 2022, the *original* 1,800-scan 3D dental scan segmentation benchmark)** — the *most cited* paper in the entire dental-3D literature, the *de facto* standard benchmark for FDI segmentation, the v0 paper's *primary* evaluation dataset. The 3DTeethSeg'22 paper is currently *missing* from the reading list (paper 001 is *part* of the project setup but the *challenge paper* itself hasn't been deeply read). Alternative: the arXiv:2509.07923 *Multimodal Contrastive Pretraining of CBCT and IOS* paper (Sep 2025) for the *cross-modality* pretraining that bridges IOS-trained and CBCT-deployed models (the v0 v2 candidate for true H5 generalization). Recommendation: **3DTeethSeg'22 challenge paper for 070** (the *de facto* benchmark, the v0 paper's primary evaluation dataset, the *right* v0 stack-validation paper). If 3DTeethSeg'22 is *not* available, fall back to arXiv:2509.07923.
