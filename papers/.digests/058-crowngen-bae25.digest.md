# Paper 058 Digest — *CrownGen: Patient-customized Crown Generation via Point Diffusion Model*

**Authors:** Juyoung Bae (corresp.), Yifan Lin, Hao Chen — HKUST + U-Hong-Kong Faculty of Dentistry + Delun Dental Hospital
**Year:** 2025 (arXiv 2512.21890 v2, Jan 2026; no peer-reviewed venue yet, code not released)
**Date:** 2026-06-08 08:35 KST (Monday, scholar-digest hourly #58)
**For:** HK (Telegram, Alf)

---

## 📄 Telegram digest

```
📄 Paper 058: CrownGen — Patient-customized Crown Generation via Point Diffusion Model (2025)
TL;DR: First AI for variable-number patient-customized dental crowns in one inference pass; first dental-3D-gen paper in reading list with a formal clinical non-inferiority trial (n=26, 95.2% acceptability, 17.78% faster than manual, all 4 criteria pass NI margin).
Hypothesis: H1 STRONG (boundary module = +27% CD), H3 STRONG (DITA = +17% CD / +30% EMD, first clinically-motivated RPE via zig-zag FDI ordering), H5 STRONGEST (self-bootstrapping = +29% CD), H2 qualified, H4 neutral
For our project: ADOPT DITA (5-line impl, biggest single-component gain in reading list) + tooth-level point representation + boundary prediction module (sub-task 2a/2b) + DPSR mesh recon + pseudo-crown self-bootstrapping; CITE as 2025 SoTA + replicate clinical reader study design; v0 compute +$400-600 Lambda
```

---

## Full digest (for record / re-read)

### One-sentence pitch
**The 2025 SoTA point-diffusion patient-customized crown generator** with a tooth-level object representation, a clinically-motivated inter-tooth attention (DITA), a boundary prediction module, and the first formal clinical non-inferiority trial in the reading list.

### Three architectural innovations (all directly portable to v0)
1. **Tooth-level point representation** — every tooth = 1024-pt cloud + 8-dim FDI embedding + binary indicator (context vs target). Decouples "where are the teeth" from "what do the teeth look like" → enables multi-crown-in-one-pass.
2. **DITA (Distance-weighted Inter-Tooth Attention)** — Transformer-XL/T5-style RPE using FDI index difference as relative position. The zig-zag FDI ordering `[17, 47, 16, 46, ..., 11, 41, 21, 31, ..., 27, 37]` makes *adjacent* AND *antagonistic* teeth both have `|Δ_ij|=1`, so the RPE learns the *right* prior (close teeth → strong morphogenetic influence) **for free**, no data-driven learning. Ablation: +17% CD, +30% EMD.
3. **Boundary prediction module** — 5-param cylinder `(c_x, c_y, c_z, r, h)` per target tooth, Smooth-L1 loss. Decouples localization from shape. Ablation: +27% CD.

### Architecture (3 stages)
- **Stage 1** Boundary pred (PointNet++ + PVC + DITA)
- **Stage 2** Point diffusion (DDPM, T=1000, PointNet++ U-Net 4SA+4FP + PVC + DITA, MSE noise pred on targets only)
- **Stage 3** DPSR mesh recon (per-point normals → Marching Cubes 64³ → watertight mesh)

### Pseudo-crown self-bootstrapping (the H5 twist)
Train v1 on 420 fully-dentate scans → use v1 to synthesize pseudo-crowns for 1364 partially-edentulous clinical scans → retrain v2 on combined 1784 scans. **Uses the model's OWN outputs, not a separate generator.** Ablation: +29% CD (largest single contributor). Mechanism: "contextual learning signal is overwhelmingly dominated by the numerous high-fidelity natural teeth, making the training process highly robust to the finer-grained inaccuracies of the few synthesized crowns" (Sec 2.2.3) — same insight as semi-supervised consistency regularization.

### Results
- **Boundary:** Dice=0.883, IoU=0.796 across 16368 boundaries (2nd molars worst 0.859/0.761, central incisors best 0.897/0.817)
- **External test (496 scans, 26288 test scenarios, k=1..6 missing teeth):** CrownGen beats 3 baselines (PointSea, AdaPoinTr, ProxyFormer) on EVERY metric for EVERY k; gap WIDENS with more missing teeth (AdaPoinTr CD 40.7→59.5 at 4 missing vs CrownGen stable ~30.6-30.9, ~2× gap)
- **Clinical reader study (n=26 cases, 23 patients, 2 blinded readers, 14-day washout, Gwet's AC2=0.947):**
  - CrownGen-assisted 740±131s vs manual 900±180s (p<0.01, **17.78% faster**)
  - Clinical acceptability 95.2% vs 94.2%
  - Composite quality 2.938 vs 2.928 (p=0.425)
  - **ALL 4 criteria pass pre-specified non-inferiority margin -0.10 at 95% CI**
- **Inference:** 85s **constant** across k (1-crown = 6-crown) — manual scales linearly

### H1-H5 connections
- **H1 (2-stage seg+gen > end-to-end):** STRONG — boundary module ablation = 27% CD improvement
- **H2 (diffusion > mesh-VAE):** QUALIFIED — point-diffusion wins multi-crown but 85s vs VF-Net's <1s
- **H3 (adjacency/opposing/arch):** STRONGEST + first clinically-motivated RPE (DITA = 17% CD / 30% EMD)
- **H4 (implicit SDF > mesh):** NEUTRAL — DPSR is post-processing, not primary representation
- **H5 (synthetic data bootstraps):** STRONGEST + new variant (self-bootstrapping = 29% CD, vs TeethGenerator's +0.05%)

### For v0 (concrete next steps)
1. **ADOPT DITA as v0 H3 mechanism for sub-task 2** — 5-line impl, 2-3 day integration, $50-100 Lambda, expected +5-15% CD
2. **ADOPT tooth-level point representation** — 1-2 day pipeline refactor, $0
3. **ADOPT boundary prediction module as v0 sub-task 2a/2b** — 1 week + $200-300 Lambda, v0's most publishable H1 result
4. **ADOPT DPSR for v0 sub-task 5** — 2-3 days + $50 Lambda, only method producing watertight meshes from points
5. **CITE as v0 paper's 2025 SoTA reference** — most recent, strongest eval, only clinical NI trial
6. **ADOPT pseudo-crown self-bootstrapping for v0 H5** — use v0's own v1 model (not TeethGenerator), expect +1-5%
7. **REPLICATE clinical reader study design** — Gwet's AC2 + 14-day washout + cross-over + pre-specified NI margin + 4 criteria
8. **ADOPT entry-level-technician + AI workflow** — 50% labor cost reduction (much bigger than 17% time reduction)
9. **HARD-CODE FDI zig-zag ordering** — 1 line, $0, +1-2% CD on any H3-equipped model
10. **FRAME v0 paper as clinical validation of H3+H5 architecture** — clinical evidence is the gap, not the architecture

### v0 stack updated
- **sub-task 2 (conditional)** += DITA + tooth-level point + boundary pred + self-bootstrapping + DPSR
- **v0 compute:** $5,140-6,230 → **$5,540-6,830 Lambda** (+$200-300 boundary + $50-100 DITA + $50 DPSR + $100-200 reader recruitment)

### Open questions for HK
(i) adopt DITA? (ii) adopt tooth-level point rep? (iii) adopt boundary pred? (iv) adopt DPSR? (v) replicate clinical reader study? (vi) adopt self-bootstrapping? (vii) frame as clinical validation? (viii) cite as 2025 SoTA? (ix) entry-level-tech workflow? (x) hard-code FDI zig-zag? — **recommend YES on all 10**

### Next paper (059)
[from the paper note, see 058-crowngen-bae25.md end of file]

---

**LanceDB row:** ✅ `0a59140a-9450-490b-b112-523e69c85574` (memories table, row 67, category `research_paper`, importance 0.7, mxbai-embed-large)
**Digest sent:** this file (Telegram will pick up the 📄 block above)
**Errors:** none
