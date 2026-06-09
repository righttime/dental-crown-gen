# Paper 086 Digest — Utonia (Zhang et al. 2026, arXiv preprint)

**Date:** 2026-06-09 09:35 KST
**Paper:** `papers/086-utonia-zhang26.md`
**Authors:** Yujia Zhang, Xiaoyang Wu, Yunhan Yang, Xianzhe Fan, Han Li, Yuechen Zhang, Zehao Huang, Naiyan Wang, Hengshuang Zhao (HKU + CUHK + Xiaomi — *same* Pointcept team as Sonata 084 + Concerto 085)
**Venue:** arXiv 2603.03283 v1 (3 Mar 2026) | CC-BY-4.0 | cs.CV
**Code/weights:** github.com/Pointcept/Concerto (Apache-2.0, inference merged) + github.com/Pointcept/Pointcept (Apache-2.0, pretrain) + HuggingFace `Pointcept/Concerto` (CC-BY-NC-4.0 weights, *restricted by HM3D/ArkitScenes NC*)

## TL;DR

**The *first* single self-supervised Point Transformer V3 encoder trained jointly on *all five* point cloud domains** (remote sensing, outdoor LiDAR, indoor RGB-D, object CAD, video-lifted) with **adaptive color/normal input handling** — the *culmination* of the **Sonata→Concerto→Utonia trilogy (084-085-086)**, the *first* 3D foundation model that does *not* need domain-specific pretraining per downstream task. Uniform SOTA on **indoor (ScanNet 81.1% / S3DIS 78.1%)**, **outdoor (NuScenes 82.2% / Waymo 71.4% / SemanticKITTI 72.0%)**, **object (ModelNet40 92.4% / ScanObjectNN-H 95.0%)**, **part-seg (ShapeNetPart 86.3% / PartNetE 62.7%)**, **open-world (PartObjaverse-Tiny 57.95% vs Sonata 55.57%, +2.4pp)**, **robotics (82.1%, +2.1pp over Concerto)**, and **spatial-reasoning (ScanRefer 54.0% / Multi3DRefer 54.1% / Scan2Cap 83.9% / ScanQA 30.5%, best on all 4)**.

**Three enabling contributions:**
1. **Causal Modality Blinding** — per-data + per-point random dropout of color/normal during pretraining. *First* principled treatment of inconsistent modality availability across point cloud domains. **Killer result:** ScanNet w/o color (linear probe) = **77.0% Utonia vs 36.8% Concerto (+40.2pp)**, the *single largest* modality-robustness gain in 3D literature. *Direct* mechanism for v0 v2 cross-IOS-scanner deployment.
2. **Perceptual Granularity Rescale** — rescale each point cloud to a canonical observing granularity. *Direct* answer to v0 v2's "intra-oral scan (mm-scale) vs model scan (cm-scale) vs CBCT-surface-extraction (sub-mm-scale)" problem.
3. **RoPE on Granularity-Aligned Coordinates** — split channels into `[u_x; u_y; u_z]`, apply 1D RoPE on rescaled coordinates with axis-wise jitter γ + isotropic scaling η (DINOv3-style). *First* paper demonstrating RoPE + granularity alignment fixes cross-domain density shifts. *Practical* v0 v2 mechanism for "use 3D RoPE in any custom dental transformer".

**Training data:** 250k cross-domain + 1M Cap3D objects (subsampled 90k/epoch). **Training cost:** $3000-5000 Lambda (64× H20 × 2 stages × ~3 days = ~150 GPU-days, **2× Concerto**). **Channel constraint:** divisible by 6 (3 axes × 2 RoPE pairs) — minor implementation gotcha.

## Hypothesis connections (H1-H5)

- **H1 (2-stage > 1-stage):** **MILD SUPPORT** — Utonia is itself a 2-stage system; uniform SOTA on 4 indoor + 3 outdoor + 4 object benchmarks with *only* a linear probe on top of frozen pre-trained encoder is the *strongest* H1 evidence in the 86-paper reading list. A single 2-stage pipeline matches or exceeds single-stage + supervised-auxiliary baselines on every domain.
- **H2 (diffusion on point clouds > mesh-based VAE):** **NEUTRAL** — Utonia is a *pretraining* method, not a generation method. Indirect: Utonia's cross-domain features are richer than Sonata's; downstream H2 generation models (LION 005, Diffusion-SDF 004, PVD 012) using Utonia features as latent should beat the same models with Sonata features.
- **H3 (conditioning on opposing + adjacent teeth improves outer surface):** **STRONG SUPPORT** — Utonia's *entire* contribution is the mechanism that makes H3 work: a single cross-domain (scanner/granularity/modality) encoder is the *direct* solution to "conditioning on a *different* scanner's representation of opposing/adjacent teeth". Killer v0 v2 mechanism: pre-train on dental-specific mixture (50k clinical + 50k CBCT + 50k intra-oral photos + 1M Cap3D-style dental CAD) to learn a *dental-specific* cross-domain representation.
- **H4 (implicit SDF > explicit mesh):** **NEUTRAL** — purely point-based (no SDF, no mesh). Indirect: Utonia features are the *encoder* for downstream H4 implicit-SDF models (DiGS 003, Diffusion-SDF 004). Concrete v0 v1 mechanism: use Utonia features as conditioning input to DiGS for outer surface generation, mesh extraction (FlexiCubes 007) unchanged.
- **H5 (synthetic data from existing CAD libraries can bootstrap training):** **STRONGEST SUPPORT — Cap3D mixture story.** Utonia includes **1,006,782 Cap3D objects** (subsampled 90k/epoch), the *largest* synthetic data source in reading list. Table 7f ablation: adding 1M Cap3D to 83k cross-domain → +0.4pp ScanNet200 lin, **+1.2pp Waymo lin (biggest gain)** — direct evidence synthetic Cap3D *complements* real data. Crucial v0 v1 mechanism: pre-train a *dental-specific* Utonia on 1M Cap3D dental CAD + 50k clinical + 50k CBCT.

## For our project (v0 v1 / v0 v2)

**Utonia is the *single highest-leverage* paper in the 86-paper reading list for v0 v1/v2** — the *founder unified multi-domain SSL reference*. The 38M ablation model is competitive with 137M at the *exact* v0 data budget (83k), and the released weights are *directly usable* (Apache-2.0 inference).

**Top 7 v0 v1 / v0 v2 actions:**

1. **ADOPT Utonia 38M as v0 v1 default backbone for all 5 sub-tasks** ($0, 1-week plumbing, $200 Lambda fine-tune, target ≥0.95 F1 on 3DTeethSeg22). The 38M model is *fast enough* for chair-side inference (<5s/crown on A100).
2. **APPLY Causal Modality Blinding to v0 training** (1-day transform, 3-day train, $100 Lambda, target <5pp degradation when color/normal missing). Use `RandomModalityDropout(p=0.3)` for color and normal channels.
3. **APPLY Perceptual Granularity Rescale to v0** (1-day, $100 Lambda). Implement `PerceptualGranularityRescale(target_grid_size=0.001)` (1mm) for v0 — intra-oral scans (mm), model scans (cm), CBCT-surface-extraction (sub-mm) all need granularity alignment.
4. **USE 3D RoPE for any custom dental transformer** (architecture choice, $0). Use RoPE base=10, ensure channel counts divisible by 6.
5. **For v0 v1, use 38M model (not 137M).** For v0 v2, scale to 137M with full 1.15M dental mixture.
6. **SKIP v0 v2 multi-modal pretraining stage for now** — Utonia-style from-scratch requires 64 H20 GPUs, not feasible for v0 v1. Linear-probed 38M on dental data is the practical v0 v1 path.
7. **PRE-TRAIN v0 v2 from Utonia 38M init on dental-specific data** ($500 Lambda, 1-week, the *killer* v0 v2 experiment): use 50k clinical + 50k CBCT + 50k intra-oral photos + 1M Cap3D dental = 1.15M dental-specific mixture, continue pretraining for 50 epochs. ~10× cheaper than from-scratch.

**CRITICAL LEGAL ISSUE for HK:** released Utonia weights are CC-BY-NC-4.0 (restricted by HM3D/ArkitScenes NC datasets). For v0 v1 *commercial* deployment, recommend **(c) use the architecture (PTv3 + Causal Modality Blinding + Perceptual Granularity Rescale + 3D RoPE) and pre-train from scratch** on permissive-license data (Cap3D + PartNet + ScanObjectNN + NuScenes + Waymo + SemanticKITTI + RE10K = 1.15M, no HM3D/ArkitScenes). **HK to confirm with legal counsel.**

## Next paper (087)

**(a) VGGT** (Wang et al. CVPR 2025 Best Paper, arXiv:2503.11651) — the *feed-forward* video-to-point-cloud reconstruction model that *both* Concerto and Utonia use for video-lifted pretraining. The *right* paper for understanding how 46,282 RE10K video-lifted point clouds in Utonia are generated and whether v0 can generate 200k dental-arch videos for v0 v2's *dental-specific* pretraining.

**(b) DINOv3** (Simeoni et al. 2025) — the *next* gen of DINOv2 that Utonia's RoPE augmentation is *borrowed* from. The *right* paper for understanding *why* the axis-wise jitter + isotropic scaling works on RoPE frequencies.

**(c) 3D-Mesh pretraining** (MeshMAE / MeshGPT / UniMesh) — the *direct* H4 mesh foundation model, the *right* v0 sub-task 4 mesh-output paper that Utonia does *not* cover (purely point-based).

**Recommendation:** Utonia *ends* the 3D-Point-Foundation-Model line. Next papers should *pivot* to (a) video-lifting, (b) image SSL, or (c) mesh foundation.
