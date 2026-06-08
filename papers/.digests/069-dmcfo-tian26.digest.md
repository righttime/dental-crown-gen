# Digest 069 — DM-CFO (Tian et al. 2026, TVCG)

**Date:** 2026-06-08 16:44 KST
**Paper:** 069-dmcfo-tian26.md
**Venue:** IEEE TVCG vol. 41 no. 8, August 2026
**arXiv:** 2603.03602v1 (4 Mar 2026)
**Authors:** Yan Tian (Zhejiang Gongshang U + Shining3D Tech), Pengcheng Xue, Weiping Ding (Nantong U, corresponding), Mahmoud Hassaballah (Prince Sattam / Qena), Karen Egiazarian (Tampere U, IEEE Fellow), Aura Conci (UFF Brazil), Abdulkadir Sengur (Firat U), Leszek Rutkowski (Polish Academy of Sciences, IEEE Life Fellow)
**Project page:** amateurc.github.io/CF-3DTeeth
**Code/data:** ❌ not released (Shining3D internal datasets — 1,416 Shining3D + 1,999 Aoralscan3 + 2,061 DeepBlue; the 3rd private-data paper in a row)

## TL;DR

**DM-CFO is the first AI framework to use 3D Gaussian Splatting (3DGS) + graph diffusion + collision-loss regularization for compositional 3D tooth generation, and the first to report sub-100μm inter-tooth penetration distance (0.07mm PD) on a real Shining3D dataset — well within clinical acceptance.** Three moving parts work together: (1) **Graph Diffusion Model (GDM)** — 5-layer 8-head 512-dim Graph Transformer denoises a target graph `Gt = (Ct, Lt, Ft, Et)` (class + layout `(x,y,z,h,w,l,k,r)` per tooth + features + edge types) conditioned on the source jaw graph `Gs` and text prompt `ys`, with edge types "Neighbor / Symmetry / Arch"; (2) **Dual-level SDS** — alternating scene-level SDS via ControlNet (global jaw coherence) + instance-level SDS via MVDream (per-tooth realism), both denoising the same 3D Gaussian parameters; (3) **Gaussian Collision Loss (GCL)** — learned per-tooth intravariance `R_i` is a soft safety margin; `L_col = Σ max(0, R_i - ||p_j^k - p_i^m||²)` penalizes neighbor Gaussians inside the anchor's spread. **Results on Shining3D: CD 0.22mm, F-Score 0.86, PD 0.07mm** (best across 11 baselines, only paper <0.1mm PD, +6-8% CD / +7-8% PD over VBCD next-best). 2D rendering: FID 193.29 / LPIPS 0.57 / PSNR 22.55 (best). Ablation: GDM gives 60% of FID gain, GCL gives 40%. 4.7-min inference = biggest weakness.

## Hypothesis connections (H1-H5)

- **H1 (2-stage > 1-stage):** PARTIAL SUPPORT — 2-stage layout→shape decomposition, but shape stage is direct SDS (not VAE+DDM). Refine: 2-stage helps when natural decomposition exists (layout+shape, boundary+surface), not for the mechanism itself.
- **H2 (latent > direct):** N/A — direct 3DGS optimization, no learned latent. Sidesteps the H2 question entirely.
- **H3 (adjacent+opposing conditioning):** **STRONGEST DIRECT SUPPORT in reading list** via explicit graph edge types. "Neighbor" = adjacent teeth (H3 ref 044/048/050), "Symmetry" = bilateral mirror (H3 ref 045/048), "Arch" = global arch curve (H3 ref 048/050). First paper to encode all three H3 mechanisms as explicit graph edges with learned attention weights.
- **H4 (implicit SDF > explicit mesh):** **REJECTION** — explicit 3DGS beats implicit SDF approaches (TranSDFNet, Point-to-mesh, SSEN, VBCD) for *compositional* multi-object generation with strict inter-object constraints. Refine: implicit SDF wins for single-object fidelity, explicit 3DGS wins for compositional generation.
- **H5 (synthetic pretrain → real):** N/A — real-data only, no synthetic pretraining. Cross-dataset evaluation across all 3 (Shining3D/Aoralscan3/DeepBlue) is *training* test, not zero-shot.

## For our project

**The 0.07mm PD is the clinical-fit bar to beat for v0 sub-task 2 compositional generation; the GCL is the single highest-leverage drop-in for v0 sub-task 4 (crown-antagonist + crown-adjacent intersection check) as a soft constraint with learned intravariance — 5-10 lines, $0, -0.1-0.3mm PD expected.** Specifically: (1) ADOPT "Neighbor/Symmetry/Arch" graph edge types for v0 sub-task 1 (FDI segmentation) — drop-in upgrade, $50-100, 1-2 days, +0.5-1.0% TIR, complementary to TCP+L_tcp+GA (049), Bezier arch (050), parabola (048), jaw-vector (045); (2) ADOPT GCL as auxiliary loss in v0 sub-task 2 — port the learned intravariance `R_i = (1/K_i)Σ||p_i^k - p_i^m||²` and hinge `L_col = Σ max(0, R_i - ||p_j^k - p_i^m||²)`, trivially portable; (3) DEFER 3DGS representation to v1 (poor direct mesh extraction) but include DM-CFO as v0 sub-task 2 baseline (only 3DGS paper in reading list); (4) REPRODUCE 0.07mm PD on 3DTeethSeg'22 — $200-400, 2-3 weeks, single most informative v0 experiment; (5) ADD intravariance per tooth type as v0 sub-task 2 secondary metric; (6) PROPOSE public benchmark for compositional tooth generation — addresses the 3-of-3 private dataset pattern, publishable v0 contribution; (7) PILOT edge-type ablation ("Neighbor" / "Symmetry" / "Arch" / all) for v0 sub-task 1 — 1 day, $50-100, confirms H3 hypothesis ranking. **v0 sub-task 1 now has 12+ H3 mechanisms (richest in lit), v0 sub-task 2 has 7 baselines (most in any paper), v0 sub-task 4 has 7 mesh-quality mechanisms, v0 eval has the only PD metric. Related-work table should be 14×N on 8-10 axes — no other paper has done this.**

## Next paper

**070: 3DTeethSeg'22 challenge + dataset paper (Liu et al. MICCAI 2022)** — the *de facto* standard benchmark for FDI segmentation (1,800 scans), the v0 paper's primary evaluation dataset, currently missing from the reading list. Alternative: arXiv:2509.07923 (Sep 2025, multimodal CBCT+IOS pretraining) for cross-modality H5.
