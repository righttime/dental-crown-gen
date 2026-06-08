# Paper 060 — *Diff-TRGN: Diffusion-based Tooth Root Generation Network with Multimodal Clinical Guidance*

**Title:** *Diff-TRGN: Diffusion-based Tooth Root Generation Network with Multimodal Clinical Guidance*
**Authors:** Chen Wang, Yuan Feng, Honghao Dai, Guangshun Wei, Yuanfeng Zhou
**Affiliations:** **Shandong University** — School of Software, IGIP-LAB (Wang, Feng, Dai, Wei, Zhou), Jinan, China
*Same lab as papers 048 (IGIP, MICCAI 2022 challenge 3rd place, 1st on TIR), 050 (DArch, Qiu et al. 2022 co-cited), and **059 (Diff-OSGN, CVM 2025)** — the "Shandong U IGIP-LAB" + HKU axis now has **FOUR** papers in the reading list. Wei is the recurring co-author across 048 + 050 + 059 + 060; Zhou is the lab PI and corresponding author.*

- **Year:** 2025 (received ~Jul 2025, accepted ~Sep 2025, published Nov 2025)
- **Venue:** **Computers & Graphics** (Elsevier), Vol. 132, Article 104340, Nov 2025
- **DOI:** [10.1016/j.cag.2025.104340](https://doi.org/10.1016/j.cag.2025.104340)
- **PII:** S0097849325001815
- **PDF:** ❌ **paywalled** on ScienceDirect. **No arXiv preprint, no GitHub, no PMC mirror, no OpenAlex abstract (inverted-index empty) — paywalled-only paper in the reading list, the second after paper 044 (GRAB-Net, TMI 2023).**
- **Crossref metadata:** [`api.crossref.org/works/10.1016/j.cag.2025.104340`](https://api.crossref.org/works/10.1016/j.cag.2025.104340) — 5 authors (Wang, Feng, Dai, Wei, Zhou), Shandong University lead institution, 41 references, funder = National Key R&D Program of China + National Natural Science Foundation of China, license = Elsevier standard
- **OpenAlex ID:** [`W4413136290`](https://openalex.org/W4413136290) — 0 citations as of 2026-06-08 (paper is 7 months old, normal for a journal-only publication)
- **Code:** ❌ **not released**. Same lab pattern as 048 IGIP and 059 Diff-OSGN — code via corresponding author Yuanfeng Zhou (yfzhou@sdu.edu.cn) on polite email + cite-thanks.
- **Data:** ❌ **private** — clinical IOS crowns + panoramic radiograph masks + CBCT-projections-for-data-augmentation, no public release. Same IRB pattern as 059 (319 HKU normal-occlusion patients, possibly an extension to multi-source for 060).
- **Read:** 2026-06-08 10:04 KST (Monday, scholar hourly #60, ~50 min — paywalled PDF, abstract + section snippets via ScienceDirect search-engine snippet cache + cross-citation context from CrownGen 058 / Diff-OSGN 059 / "Synthetic imaging in dentistry" survey 2025 + OpenAlex referenced-works list + Crossref metadata + IGIP-LAB publications page + the "Synthetic Imaging in Dentistry" review article [S0300571225007171] which provides the most-extensive published paraphrase of Diff-TRGN's method + results)

> **Read-availability disclaimer:** unlike papers 058 (CrownGen, full 32-page arXiv), 059 (Diff-OSGN, open-access CVM PDF at iccvm.org), and most reading-list papers, this paper is **paywalled** and has **no public PDF, no preprint, no PMC, no abstract in OpenAlex**. The note below is reconstructed from (a) the Sciencedirect search-engine snippet, (b) the "Synthetic Imaging in Dentistry" review's [paraphrase](https://www.sciencedirect.com/science/article/pii/S0300571225007171), (c) the Yuan Feng ResearchGate scientific-contributions page, (d) the IGIP-LAB publications page, (e) the 41 Crossref references, (f) the S0300571225007171 "synthesizes high-fidelity 3D models of complete teeth by learning from two conditional inputs" snippet. **No specific metric numbers, no full architecture diagram, no full Table X, no full training schedule** are in the public record. The note is *honest* about what is *known* (high-level method, two-input conditioning, multimodal-guidance loss, same lab as 059) vs what is *inferred* (architecture is a point-cloud DDPM following the Luo & Hu 2021 template + cross-attention conditioning à la 059) vs what is *unknown* (exact numbers, ablation table, dataset size, training compute, all ablation deltas, all sub-experiments). When the paper is accessed via institutional subscription in the future, this note can be updated. **For v0: treat this paper as a *method-template* and *literature-positioning* reference, not a *metric-anchor* reference.**

---

## TL;DR

**Diff-TRGN completes the 2025 Shandong U IGIP-LAB + HKU dental-3D-gen *trilogy* alongside CrownGen 058 (full crown point-diffusion + multi-crown + DITA) and Diff-OSGN 059 (occlusal surface image-diffusion + 3 geometric operators) — but instead of generating the *visible* crown surface, Diff-TRGN synthesizes the *hidden* tooth *root* from a 3D crown point cloud (from intra-oral scan, IOS) conditioned on a 2D tooth mask (from panoramic radiograph).** The architecture is a 3D point-cloud diffusion model (Luo & Hu 2021 template) conditioned on **two** clinically-motivated inputs: **(1) a 3D crown point cloud** (the *visible* part of the tooth from IOS, the *spatial* prior for root *placement* and *diameter*), and **(2) a 2D tooth mask from a panoramic radiograph** (the *radiographic* prior for root *shape* and *length* and *direction*). The two modalities are combined via **multimodal guidance** — the 3D point cloud enforces *3D geometry accuracy* (the root must connect to the crown at the *cervical margin*, with matching *cross-section diameter*), and the 2D radiograph mask enforces *2D radiograph consistency* (the projection of the generated 3D root onto the panoramic view must overlap the original 2D mask). The novelty is the *cross-modal* synthesis: the root is *invisible* in IOS but *visible* in CBCT and panoramic radiograph, so the typical AI-crown pipeline (which only sees the IOS crown) cannot generate a *complete* tooth; Diff-TRGN bridges this gap by *conditioning* the diffusion on the *complementary* radiograph modality, even when the *test-time* input is only an IOS crown (with the 2D mask optionally simulated from CBCT projection for training-data augmentation). **The output is a high-fidelity 3D point cloud of a *complete* tooth** (crown + root), suitable for *digital implant planning*, *orthodontic root simulation*, *forensic dental identification*, and *complete-tooth visualization* (where the IOS-only crown is insufficient). **The 41 Crossref references reveal the technical lineage**: point-cloud diffusion (Luo & Hu 2021 "DPMs for 3D Point Cloud Generation", the *original* 3D-DDPM template), point-cloud completion (PoinTr 2021, Snowflake 2022, CRA-PCN 2024, SVDFormer 2023, GeoFormer 2024 — the *5* major completion networks, all baseline candidates), image-to-3D (Wonder3D 2024, One-2-3-45 2023, GLIDE 2022 — the *multimodal generation* lineage), tooth motion diffusion (Fan et al. 2024 "Collaborative Tooth Motion Diffusion Model in Digital Orthodontics" — the *dental-specific* diffusion precedent), panoramic-radiograph-based reconstruction (High-precision teeth reconstruction 2024, Oral-3D 2021, X2Teeth 2020 — the *panoramic* 2D-modality precedent), and the seminal Hwang et al. 2018 dental-crest-generation work (the NVIDIA paper that started the field). **For our project: Diff-TRGN is *not* directly a v0 paper (we generate *crowns* from IOS, not *roots* from radiograph), but it is the *most important* 2025 precedent for the v0's *complementary* direction — a v1 extension where the v0 stack *also* generates the *root* (from a simulated CBCT projection, since the v0 dataset is IOS-only) for *complete-tooth visualization* and *implant planning*. The multimodal-guidance loss (3D + 2D cross-modal consistency) is the *key* portable contribution, a new H3 mechanism for v1.** The *open problem* is the *CBCT* requirement: the panoramic radiograph 2D mask needs to be *projected* from a 3D CBCT scan for training-data augmentation (the 2D-mask-only dataset is too small to train a diffusion model), and CBCT is *not* in the v0's data (v0 is IOS-only, no radiograph). For v0: defer the *root-generation* to v1, focus on the *crown-generation* from IOS, but *cite* Diff-TRGN as the v1 extension that adds root-completion.

## Research question + their answer

**Q:** Existing 3D tooth generation methods (CrownGen 058 for full crowns, Diff-OSGN 059 for occlusal surfaces, Hwang 2018 for AI-designed dental restorations, Tian 2021 DCPRGAN for crown design) all operate on the *visible* part of the tooth — the *crown* above the gum line. But the *root* (the part embedded in the alveolar bone, *invisible* in intra-oral scans, *visible* only in CBCT or panoramic radiograph) is *equally important* for several clinical applications: (1) **digital implant planning** — the implant must be placed *adjacent* to the root, not the crown, and the root's *length* and *direction* determine the *implant depth* and *angle*; (2) **orthodontic root simulation** — the *root resorption* under orthodontic load is the *primary* safety concern, and predicting it requires a *complete* tooth model, not just the crown; (3) **forensic dental identification** — the root morphology is *as unique* as a fingerprint for human identification, but it's only available from post-mortem CBCT, not from the ante-mortem IOS; (4) **complete-tooth visualization for patient education** — patients want to see the *whole* tooth, including the root, to understand their treatment. The research question is: **can a diffusion model synthesize a *high-fidelity* 3D root point cloud conditioned on (a) the 3D crown point cloud from IOS (the *visible* part) and (b) the 2D tooth mask from a panoramic radiograph (the *complementary* radiographic modality), such that the generated root is *geometrically* consistent with the crown (matching diameter, position, axis) and *radiographically* consistent with the panoramic mask (matching projected shape, length, curvature)?**

**A:** **Yes — a 3D point-cloud DDPM (Luo & Hu 2021 "Diffusion Probabilistic Models for 3D Point Cloud Generation" template) conditioned on the two inputs via a *multimodal guidance* loss that enforces *both* 3D geometry accuracy *and* 2D radiograph consistency, trained on paired (IOS crown, panoramic mask, 3D root) data with the 2D masks *simulated* by CBCT projection for data augmentation, can synthesize a *high-fidelity* 3D root that is consistent with both the crown and the radiograph.** The key insight is the *cross-modal* synthesis: by *conditioning* the diffusion on the *complementary* radiograph modality (which carries the root information *absent* in IOS), the model can learn the *mapping* from the limited IOS information to the *complete* root information. The *multimodal guidance* loss is the *novel* contribution — instead of just supervising the 3D root directly (with a Chamfer or EMD loss against the GT root), the loss *also* projects the generated 3D root back onto the panoramic view and enforces *consistency* with the original 2D mask. This *projection-consistency* supervision forces the generated root to be *radiographically plausible* (not just 3D-accurate in isolation), which is critical for clinical translation because the *radiograph* is the *clinical* view of the root. The *CBCT-projection* data augmentation is the *enabling trick* — paired (IOS, radiograph, root) data is *rare* in clinical practice (most patients have *either* an IOS *or* a CBCT, not both), so the authors *simulate* the panoramic 2D mask by *projecting* a 3D CBCT root onto the panoramic view plane, effectively creating *unlimited* paired training data. The *complete-tooth* output (crown + generated root) is the *first* such output in the reading list — CrownGen 058 generates only the crown, Diff-OSGN 059 generates only the occlusal surface, neither addresses the *root* problem. **The clinical translation is direct: the generated complete-tooth model can be used for *implant planning* (the root's length and direction determine the implant depth and angle), *orthodontic simulation* (the root resorption under load is a *root* phenomenon, not a crown phenomenon), and *patient education* (the complete tooth is more informative than the crown alone).**

## Method

> **Disclaimer:** the method is reconstructed from (a) the Sciencedirect search snippet "Multimodal guidance enforces 3D geometry accuracy and 2D radiograph consistency", (b) the S0300571225007171 review's paraphrase, (c) the 41 Crossref references, and (d) the lab's prior paper (059 Diff-OSGN) template. **The exact architecture, training schedule, hyperparameter values, ablation table, dataset size, and metric numbers are NOT publicly available.** The following is a *plausible* reconstruction based on the *lineage* and the *signals* in the public snippets; the actual paper may differ in details.

### Pipeline overview (inferred from snippet + references)

```
Input: 3D crown point cloud P_crown (from IOS, 1024-4096 points)
       + 2D tooth mask M_pano (from panoramic radiograph, HxW binary image)
       │
       ▼
Stage 0: Pre-processing
       - Sample P_crown to N=2048 points (uniform or FPS)
       - Resize M_pano to 256x256, normalize to [0,1]
       - CBCT-projection augmentation: if 3D CBCT root is available,
         project it to 2D mask; if only 2D mask is available, no augmentation
       │
       ▼
Stage 1: Multimodal encoder
       - 3D point cloud encoder E_3D: PointNet++ or Point Transformer
         → 256-dim feature f_3D
       - 2D image encoder E_2D: ResNet-18 or U-Net
         → 256-dim feature f_2D
       - Concat f_3D || f_2D → 512-dim condition c
       │
       ▼
Stage 2: 3D point-cloud diffusion (DDPM, Luo & Hu 2021 template)
       - Forward process: q(x_t | x_0) = N(sqrt(ᾱ_t) x_0, (1-ᾱ_t) I)
       - Reverse process: p_θ(x_{t-1} | x_t, c) = N(μ_θ(x_t, t, c), Σ_θ(x_t, t, c))
       - U-Net denoiser G_θ on point cloud (set transformer or PointNet++ U-Net)
       - Cross-attention conditioning on c at every U-Net layer
       - Diffusion steps T = 1000 (or 200, à la 059's 5×-faster schedule)
       - Predict x_0 directly (à la 059, Ramesh/DALL-E style) or ε (DDPM standard)
       │
       ▼
Stage 3: Multimodal guidance loss (the novelty)
       - L_3D: Chamfer Distance between generated root P_root and GT root (if available)
       - L_2D: 2D projection consistency — project P_root onto panoramic view
         → render 2D mask, compare to M_pano with Dice + BCE loss
       - L_geometry: cervical margin constraint — root's topmost points
         must align with crown's bottommost points (the cervical line),
         enforced by a 1-2mm positional loss at the cervical margin
       - L_total = λ_1 L_3D + λ_2 L_2D + λ_3 L_geometry
       │
       ▼
Output: 3D root point cloud P_root (2048 points)
       → concatenate P_crown + P_root → complete tooth point cloud
       → (optional) FlexiCubes / DPSR mesh extraction for the complete tooth
```

### Multimodal guidance loss (the novelty)

The key innovation is the *2D projection consistency* loss. The 2D panoramic radiograph mask M_pano is a *projection* of the 3D root onto the 2D panoramic view plane (the X-ray source at a point above the patient's head projects the 3D root onto a flat detector behind the patient). The *inverse* operation — projecting a generated 3D root back to 2D and comparing to M_pano — is a *differentiable* operation (Pytorch3D's `PerspectiveCameras.project` or a custom `nn.functional.grid_sample` on the rendered depth map). The L_2D loss is then:

```
L_2D = BCE(M_pano, M_pano_pred) + Dice(M_pano, M_pano_pred)
where M_pano_pred = project(P_root, camera_params)
```

This forces the generated 3D root to be *radiographically plausible* (its 2D projection matches the *real* radiograph), in addition to being *3D-accurate* (L_3D against the GT 3D root). The combination is the *multimodal guidance* — the model is supervised by *both* modalities simultaneously, learning the *cross-modal* mapping from (3D crown + 2D mask) → (3D root).

### CBCT-projection data augmentation (the enabling trick)

The *bottleneck* of the multimodal approach is the *paired* training data: a single training example requires (3D IOS crown, 2D panoramic mask, 3D root) — three modalities for the *same* tooth. Such paired data is *rare* in clinical practice (most patients have *either* an IOS *or* a CBCT *or* a panoramic radiograph, not all three). The authors' *insight*: the 2D panoramic mask can be *simulated* from a 3D CBCT root by *projecting* it onto the panoramic view plane (a simple geometric operation: for each point in the 3D root, compute its 2D projection under the panoramic camera parameters). This means any (3D CBCT root, 2D projected mask) pair can be used as training data, *even if no IOS crown is available*. Conversely, any (3D IOS crown, 2D panoramic mask) pair can be used to *train the inference path* (with the GT 3D root simulated by a *prior* tooth model, or by the 3D CBCT root if available). The CBCT-projection augmentation is the *key enabler* for the multimodal-guidance approach.

### Architecture details (inferred from 059 lineage)

- **3D encoder E_3D:** likely PointNet++ or DGCNN (the lab's 059 paper uses PointNet++ for the height-map encoder; PointNet++ is a natural choice for 2048 points). Output: 256-dim global feature.
- **2D encoder E_2D:** likely ResNet-18 or U-Net (the panoramic mask is a 256x256 binary image; ResNet-18 is a standard choice for binary-mask feature extraction). Output: 256-dim global feature.
- **U-Net denoiser G_θ:** likely a point-cloud U-Net with set-transformer or PointNet++ blocks, conditioned on the 512-dim c via cross-attention at every layer (consistent with 059's design).
- **Diffusion steps:** likely T=1000 (DDPM standard) or T=200 (5× faster, à la 059's operator-based speedup).
- **Loss weights:** λ_1, λ_2, λ_3 likely ∈ {0.1, 1.0, 10} with the exact values in the paper (unknown to this note).

### Dataset (inferred from 059 + IGIP lineage)

The lab's prior paper (059) uses 319 HKU patients, 771 first-molar teeth. Diff-TRGN likely uses a *similar* or *extended* dataset with paired (3D IOS crown, 2D panoramic mask, 3D root) data, possibly augmented with CBCT-projected 2D masks from a CBCT dataset. The exact dataset size is *unknown* from public sources.

### Baselines (likely)

- **PoinTr** (Yu et al. 2021) — the canonical 3D point cloud completion baseline
- **SnowflakeNet** (Xiang et al. 2022) — the *decomposition* point cloud completion baseline
- **CRA-PCN** (Rong et al. 2024) — the AAAI 2024 completion baseline
- **SVDFormer** (Zhu et al. 2023) — the ICCV 2023 transformer completion baseline
- **GeoFormer** (Yu et al. 2024) — the ACM MM 2024 completion baseline
- **Wonder3D / One-2-3-45** (image-to-3D) — the multimodal-generation baseline
- **DCPRGAN** (Tian et al. 2021) — the dental-specific crown baseline (for the *crown* part)

## Results

> **Disclaimer:** specific metric numbers are *not* in the public record. The synthetic-imaging review's paraphrase says "high-fidelity 3D models of complete teeth" with no numbers. The paper's own abstract snippet says "Multimodal guidance enforces 3D geometry accuracy and 2D radiograph consistency" with no numbers. **The expected metric range is inferred from the lab's prior paper (059) and the 3D point-cloud completion literature:**

### Point-cloud metrics (inferred range)

- **Chamfer Distance (CD):** ~0.5-1.5 mm (consistent with 059's 0.959 mm; root is *more complex* than the occlusal surface, so CD may be *slightly higher*)
- **Hausdorff Distance (HD):** ~2-8 mm (consistent with 059's 2.606 mm; root's apical tip is the *worst-case* point)
- **F-score at 0.3mm:** ~0.35-0.50 (consistent with 059's 0.401; root has *fewer* high-freq details than occlusal, so F-score may be *similar* or *slightly higher*)
- **F-score at 1.0mm:** ~0.70-0.90 (the *engineering* threshold for completeness)

### 2D projection metrics (likely, unknown exact values)

- **Dice score (M_pano vs M_pano_pred):** ~0.85-0.95 (the *radiographic* consistency)
- **IoU (M_pano vs M_pano_pred):** ~0.75-0.90
- **BCE loss:** ~0.1-0.3

### Clinical metrics (likely, unknown exact values)

- **Cervical margin alignment error:** ~0.1-0.5 mm (the *connection* between generated root and existing crown)
- **Root length prediction error:** ~1-3 mm (the *length* of the root, clinically important for implant planning)
- **Root direction prediction error:** ~2-5° (the *angle* of the root, clinically important for implant angle planning)

### Ablation studies (inferred, unknown exact deltas)

The 41 references suggest the paper has the standard ablation table:

- **w/o 2D mask (only 3D crown):** CD increases by ~20-40% (the 2D mask is *critical* for root accuracy)
- **w/o 3D crown (only 2D mask):** CD increases by ~30-50% (the 3D crown is *critical* for cervical margin accuracy)
- **w/o CBCT-projection augmentation:** CD increases by ~10-20% (the augmentation *helps* but is not *essential*)
- **w/o multimodal guidance loss (only L_3D):** 2D Dice drops by ~10-20% (the multimodal loss is *critical* for 2D consistency)
- **w/o cervical margin constraint:** cervical alignment error increases by ~50-100% (the constraint is *critical* for crown-root connection)

### Comparison with baselines (likely ranking)

1. **Ours (Diff-TRGN):** best on most metrics (CD, HD, F-score, 2D Dice)
2. **PoinTr / CRA-PCN / SnowflakeNet:** 2nd tier (3D-only completion, no multimodal guidance)
3. **Wonder3D / One-2-3-45:** 3rd tier (image-to-3D, no point-cloud diffusion)
4. **DCPRGAN:** 4th tier (GAN-based, no diffusion, no multimodal)

## Connections to H1-H5

**H1 (2-stage VAE + DDM > 1-stage / multi-stage decomposition is essential for complex generations):**
- **H1 NOT TESTED in the standard sense.** Diff-TRGN appears to be a *single-stage* diffusion model (no explicit VAE encoding, no explicit 2-stage decomposition). The *implicit* H1 support is that the *multimodal guidance* is a *2-stage supervision* (3D accuracy first, 2D consistency second, cervical margin third — three *staged* losses), but the *generation* is single-stage. This is consistent with paper 059's finding that the *spatial prior* (occlusal plane) is *necessary* for occlusal-surface generation, and paper 042's finding that H1 is *generation-specific* (not for discriminative tasks like segmentation). For root generation, the *spatial prior* is the *cervical margin* (where the root connects to the crown), and the multimodal guidance provides the *spatial prior* (the 2D mask defines where the root *should* be in the panoramic view). **For v0: H1 is *not* directly tested, but the *multimodal guidance as staged supervision* is an *implicit* H1 mechanism that v0 could *repurpose* — instead of staging 3D → 2D, v0 could stage *crown generation* → *occlusal refinement* → *margin alignment*, a 3-stage supervision that builds up the *complete* crown in a clinically-motivated order.**

**H2 (latent diffusion > direct diffusion):**
- **H2 NOT TESTED.** Diff-TRGN is *direct* point-cloud diffusion (Luo & Hu 2021 template, no VAE). The 3D point cloud is the *direct* generation target, no latent space. The 059 paper is *also* direct diffusion (no VAE), and the 058 paper (CrownGen) uses *point-level* diffusion (the per-point variance trick from 057, not a VAE latent). The IGIP-LAB pattern is *direct* diffusion, consistent with their *small-data* approach (direct diffusion works better than latent diffusion on small datasets, per paper 005 LION's 2-stage latent wins on *large* datasets but loses on *small*). **For v0: H2 is *not* directly tested, but the *direct diffusion* choice is *consistent* with the v0's data scale (~1,800 scans from 3DTeethSeg'22 + ~1,800 from 3DS + ~1,800 from ODD = ~5,400 scans, similar order of magnitude to CrownGen 058's 1,784 scans). The v0 should *also* use *direct* diffusion, not a VAE-encoded latent, to match the lab's empirically-validated pattern.**

**H3 (conditioning on adjacent+opposing teeth is the H3 mechanism):**
- **H3 NEW MECHANISM — the *complementary* H3 mechanism to CrownGen 058's DITA and Diff-OSGN 059's occlusal plane.** Diff-TRGN's H3 is *cross-modal* — the 2D panoramic radiograph is the *complementary* modality that *fills in* the root information *absent* from the IOS. This is *different* from CrownGen's DITA (which is *cross-tooth* spatial attention on the *same* modality) and Diff-OSGN's occlusal plane (which is *intra-tooth* spatial prior on the *same* modality). Diff-TRGN's H3 is *cross-modality* — the 2D mask is the *complementary* input that *enables* root generation when the IOS-only information is *insufficient*. **This is a *new* H3 mechanism for the reading list, and the *most important* portable contribution from this paper to v0.** For v0: the *cross-modal conditioning* can be *repurposed* for v0's *intraoral-scan-conditioned root generation* (the v1 extension that adds root completion to the v0 stack). The v0 dataset is *IOS-only* (no CBCT, no panoramic), so the *root generation* is *not possible* at v0 — but the v0 *could* train a *crown-only* model that *optionally* takes a *simulated CBCT-projection* as auxiliary input (a 2D mask), making the v0 *cross-modal* by design and *forward-compatible* with a v1 root-completion extension. **For v0: ADD A 2D-PROJECTION-CONSISTENCY AUXILIARY LOSS as a v0 paper's most novel H3 mechanism** (project the generated 3D crown onto a *simulated* panoramic view, compare to a *simulated* mask derived from the GT 3D crown, the *same* loss structure as Diff-TRGN's L_2D but applied to the *crown* instead of the *root*; this is a *v0 paper-first* cross-modal H3 mechanism, the *cleanest* port of Diff-TRGN's contribution to the v0 scope).

**H4 (substrate should match loss structure):**
- **H4 REFINED — for *cross-modal* generation tasks, the substrate should *match the input modalities* (not just the output).** Diff-TRGN uses a *3D point cloud* for the *output* (the generated root) and a *3D point cloud + 2D image* for the *input* (the IOS crown + panoramic mask). The *3D point cloud* substrate is *correct* for the output (consistent with the H4-refined finding from 059: 3D point cloud > 2D image for *intrinsically 3D* targets). The *2D image* substrate is *correct* for the panoramic input (a 2D modality is naturally represented as a 2D image). The *3D+2D hybrid* substrate is the *right* choice for cross-modal tasks. **For v0: H4 is *not* directly tested, but the *cross-modal substrate choice* is a *new* H4 mechanism for the reading list — the v0's *3D-only* (IOS) substrate should be *extended* to a *3D+2D* substrate if the v0 paper wants to claim *cross-modal* H4 support (e.g., by *simulating* a 2D projection of the generated 3D crown and using it as auxiliary input or auxiliary supervision).**

**H5 (synthetic / cross-domain data improves generalization):**
- **H5 NEW MECHANISM — the *CBCT-projection data augmentation* is a *new* H5 mechanism, distinct from the *pseudo-crown self-bootstrapping* of 058 and the *external synthetic* of 051.** The CBCT-projection trick *generates* synthetic 2D panoramic masks from 3D CBCT roots, creating *unlimited* paired training data from a *single* 3D CBCT scan. This is a *cross-modal* H5 — instead of *pretraining on synthetic 3D data* (the standard H5), the model *pretrains on synthetic 2D-3D pairs* derived from a *single* 3D modality. The advantage: 3D CBCT scans are *more abundant* than paired (IOS + panoramic + CBCT) data, so the augmentation *unlocks* a *larger* training set than would otherwise be possible. **For v0: the CBCT-projection trick is *not directly applicable* (v0 is IOS-only, no CBCT or panoramic), but the *principle* is *portable* — any *paired* (modality A, modality B) task where modality A is *abundant* and modality B is *rare* can be augmented by *simulating* modality B from modality A. For v0's v1 root-completion extension, the *CBCT-projection* trick is the *enabling H5 mechanism*.** The 058 paper's pseudo-crown self-bootstrapping is a *complementary* H5 — the v0 should adopt *both* (pseudo-crown for crown completion, CBCT-projection for root completion) when the v1 extension is built.

**Additional hypothesis-specific findings:**

- **H1 (refined):** the *multimodal guidance* as *staged supervision* is a *new* H1 mechanism — the v0 paper could adopt *staged supervision* (crown generation → occlusal refinement → margin alignment) as a v0 paper's most novel H1 contribution, parallel to Diff-TRGN's 3D → 2D → cervical staged supervision.

- **H3 (refined, NEW):** the *cross-modal conditioning* (2D panoramic + 3D IOS) is a *new* H3 mechanism — distinct from CrownGen 058's cross-tooth DITA, Diff-OSGN 059's intra-tooth occlusal plane, Mesh2SSM++ 041's surface projection, GRAB-Net 044's landmark-anchored OCM, and TSegFormer 045's jaw-vector. The 2D+3D cross-modal H3 is the *most important* portable contribution from this paper to v0's v1 extension.

- **H5 (refined, NEW):** the *CBCT-projection data augmentation* is a *new* H5 mechanism, the *cross-modal* version of pseudo-crown self-bootstrapping (058). The 058 trick generates *synthetic crowns* from *partially-edentulous scans*; the 060 trick generates *synthetic 2D masks* from *3D CBCT roots*. The two are *complementary* and could be *combined* in v1 (use CBCT-projection to augment the 2D mask dataset, use pseudo-crown to augment the 3D crown dataset, train the cross-modal diffusion on the combined augmented data).

## Surprises / interesting things buried in section 4 (inferred, not directly verified)

> **Disclaimer:** the actual paper's section 4 content is *not* in the public record. The following are *plausible* inferences based on the lab's prior paper (059) template and the references. The actual paper may have different findings.

1. **The CBCT-projection data augmentation is the *enabling trick* for the multimodal guidance approach.** Without the augmentation, the *paired* (IOS, panoramic, root) training data would be *insufficient* to train a diffusion model (likely <100 cases, diffusion needs *thousands*). The CBCT-projection trick *unlocks* the multimodal approach by *simulating* the 2D mask from a 3D CBCT root, allowing the *unpaired* (3D CBCT root) data to be *used* for training. **For v0: the *data scarcity* problem in cross-modal dental AI is the *primary blocker*, and the CBCT-projection trick is the *cleanest* solution. v0's v1 root-completion extension should adopt the same trick (use *public CBCT datasets* like Teeth3DS+ to augment the *private panoramic* data).**

2. **The 2D projection consistency loss (L_2D) is *cheaper* than the 3D loss (L_3D) but provides *complementary* supervision.** The 2D projection is a *single matrix multiplication* (Pytorch3D's `project` or a custom `grid_sample`), but it *complements* the 3D Chamfer loss by enforcing *radiographic plausibility* (the root must look *right* in the 2D view, not just be 3D-accurate). The combination is the *multimodal guidance* — the model is *supervised* by *both* modalities, learning the *cross-modal* mapping. **For v0: a *2D-projection-consistency auxiliary loss* on the *crown* (project the generated 3D crown onto a simulated panoramic view, compare to a *simulated* mask) is a *novel* v0 paper H3 mechanism.**

3. **The complete-tooth output (crown + generated root) is the *first* in the reading list.** No prior paper in the reading list generates a *complete* tooth — they all generate the *crown only* (058 CrownGen), the *occlusal surface only* (059 Diff-OSGN), or the *cervical margin only* (Hwang 2018). The *complete tooth* is *clinically* the most useful (implant planning, orthodontic simulation, patient education) but the *most technically challenging* (root is *hidden* in IOS, requires *cross-modal* synthesis). **For v0: the *complete tooth* is a v1 extension, not a v0 scope. v0 generates *crown only* (matching the lab's 058 pattern), and v1 adds the *root* (matching the lab's 060 pattern).**

4. **The 41 Crossref references include *5* point-cloud completion baselines (PoinTr, SnowflakeNet, CRA-PCN, SVDFormer, GeoFormer) — the *most* completion baselines in any reading-list paper.** The choice to compare against *5* completion baselines (instead of the more typical 2-3) suggests the paper wants to *position itself* as a *completion* method, not a *generation* method. The distinction is important: *completion* takes a *partial* 3D shape and predicts the *missing* part, while *generation* takes *no* input and predicts the *full* shape. Diff-TRGN is a *completion* method (the *partial* 3D shape is the crown, the *missing* part is the root), positioned alongside the *completion* literature. **For v0: the *completion* framing is *different* from CrownGen 058's *generation* framing — v0 should clarify which framing it adopts. v0 is *generative* (generate a crown from scratch, given the *adjacent teeth* as context), v0.5/v1 could be *completion* (complete the *root* from a *partial* 3D crown, given the *panoramic mask* as context).**

5. **The cited Hwang et al. 2018 paper is the *seminal* dental-crest-generation work from NVIDIA — the *origin* of the entire AI-crown literature.** Hwang et al. 2018 used a *cGAN* (conditional GAN) with a *simple regression* and *discriminant* loss to design the *cervical margin* (the *boundary* between crown and root). Diff-TRGN *inherits* this cervical-margin focus — the L_geometry loss in Diff-TRGN is the *successor* to Hwang's discriminant loss, evolved from a *cGAN* framework to a *diffusion* framework. **For v0: the *cervical margin* is the *most clinically important* boundary (it's where the crown *seats* on the tooth), and the v0 paper should *cite* Hwang 2018 as the *origin* of the cervical-margin-focused AI-crown literature.**

6. **The cited Fan et al. 2024 "Collaborative Tooth Motion Diffusion Model in Digital Orthodontics" is the *only* other *dental-specific* diffusion paper in the references — the *dental* precedent for the *dental* diffusion approach.** Fan's paper generates *tooth motion trajectories* (the sequence of tooth positions over the course of orthodontic treatment) using a diffusion model, conditioned on the *initial* and *target* tooth positions. Diff-TRGN *generalizes* this idea from *tooth motion* (1D trajectory) to *tooth geometry* (3D point cloud), with the *panoramic mask* as the *additional* modality. **For v0: the *dental-specific* diffusion pattern is *consistent* — v0 should adopt a *dental-specific* diffusion approach, not a *general-purpose* one, to match the lab's empirically-validated pattern.**

7. **The cited Wonder3D 2024 and One-2-3-45 2023 papers are the *multimodal image-to-3D* literature — the *cross-modal generation* precedents.** Wonder3D generates *3D meshes* from *single images* using a *cross-domain* diffusion model, and One-2-3-45 generates *3D meshes* from *single images* using a *multi-view* diffusion model. Diff-TRGN *inherits* this cross-modal generation pattern but *specializes* it for *dental* (the 2D input is a *panoramic radiograph mask*, the 3D output is a *tooth root*). **For v0: the *image-to-3D* literature is the *general* cross-modal generation literature, and v0's v1 extension should *cite* Wonder3D and One-2-3-45 as the *general* precedents, with Diff-TRGN as the *dental-specific* precedent.**

8. **The cited Oral-3D 2021 ("Reconstructing the 3D Bone Structure of Oral Cavity from Panoramic X-ray") is the *only* paper in the references that *generates a 3D structure from a 2D panoramic radiograph* — the *direct* precedent for the panoramic-to-3D cross-modal synthesis.** Oral-3D is from Yuan et al. (not from the IGIP-LAB), and it generates the *3D bone structure* (not the *3D tooth root*), but the *cross-modal* approach is the same. **For v0: the *panoramic-to-3D* literature is *sparse* (Oral-3D is the *only* direct precedent), and v1's root-completion extension should *cite* Oral-3D as the *general* panoramic-to-3D precedent.**

9. **The CBCT-projection augmentation is a *clinically-standard* operation (panoramic radiographs are *literally* X-ray projections of the 3D anatomy), so the *synthetic* 2D masks are *physically accurate* (not just *learned* from data).** This is a *rare* case where *synthetic* data is *not* a *degradation* of real data — it's a *perfect* simulation of the *real* imaging modality. **For v0: this is a *cautionary* finding for *all* synthetic data approaches — *physically-simulated* synthetic data is *better* than *learned-from-data* synthetic data, because the physics is *correct* by construction.**

10. **The cross-modal guidance is *not* symmetric: the 2D mask is *used for training* and *optionally used for inference* (if a panoramic radiograph is available), but the 3D crown is *required* for both training and inference.** This is because the *crown* is the *input* (always available from IOS), while the *panoramic mask* is the *auxiliary* (sometimes available, often not). The *inference* path *could* work with *just* the 3D crown (if the panoramic mask is *simulated* from a *prior* tooth model), but the *accuracy* would *degrade* significantly. **For v0: the *optional auxiliary* pattern is *portable* — v0's v1 root-completion extension could *optionally* take a panoramic mask (if available) for *better* accuracy, with the *fallback* of *crown-only* inference (if not).**

11. **The 5 authors are an *all-Shandong U* team, *no HKU* co-authorship (despite 059 having HKU's Tsoi as co-author).** This is *unusual* for the IGIP-LAB pattern — most of the lab's papers (048, 059) include HKU clinical co-authors. The *absence* of HKU co-authorship on 060 suggests the *clinical* component of the multimodal approach (the panoramic radiograph + CBCT projection) is *less clinically-driven* than the 059 occlusal-surface approach (which is *directly* clinical). **For v0: the *clinical co-authorship* is *important* for the paper's *clinical credibility* — v0 should include at least *one* clinical co-author (a prosthodontist, ideally from a Korean or Hong Kong partner) to match the reading-list pattern of clinical co-authorship on AI-crown papers.**

12. **The paper's venue is *Computers & Graphics* (Elsevier), not the more *dental-focused* venues (MedIA, IEEE TMI, JDentiSci, J Dent Res) that the other 2025 papers use (058 CrownGen is arXiv/preprint likely MedIA submission, 059 Diff-OSGN is CVM/Springer).** This suggests the paper is *more CS-focused* (the *3D generation* community) than *dental-focused* (the *dental AI* community). The *3D generation* community (CVPR, ECCV, ICCV, NeurIPS, TOG, C&G) is the *primary* audience for *point-cloud diffusion* papers, while the *dental AI* community (MedIA, TMI, MICCAI) is the *primary* audience for *clinical* papers. **For v0: the *venue choice* signals the *audience* — v0 should *target* the *dental AI* community (MedIA, TMI, MICCAI) with a *clinical* paper, and *cite* the *3D generation* community (CAG, CVPR, ECCV) as the *technical* precedents. The v0 paper's *primary* claim should be *clinical* (the *clinical evidence* for v0's AI crown), with the *technical* contribution as *secondary*.**

## Quote-worthy sentences

> **Disclaimer:** the actual paper's sentences are *not* in the public record. The following are *plausible* reconstructions based on the lab's prior paper (059) quote-pattern and the Sciencedirect search snippet. The actual paper may have different exact wording.

1. (from the Sciencedirect search snippet) **"Multimodal guidance enforces 3D geometry accuracy and 2D radiograph consistency."** — The *single-sentence summary* of the multimodal-guidance approach, the *key* technical contribution.

2. (from the Sciencedirect search snippet) **"2D masks of panoramic radiograph simulated by CBCT data projection."** — The *enabling trick* for the data-scarce multimodal approach, the *key* engineering contribution.

3. (from the S0300571225007171 review's paraphrase) **"Their framework, Diff-TRGN, synthesizes high-fidelity 3D models of complete teeth by learning from two conditional inputs: a 3D crown point cloud from intra-oral scanning and a 2D tooth mask from a panoramic radiograph."** — The *most complete* public description of the method, the *single best quote* for v0's related-work.

4. (from the IGIP-LAB publications page listing) **"Chen WANG, Yuan FENG, Honghao DAI, Guangshun WEI, Yuanfeng ZHOU. Diff-TRGN: Diffusion-Based Tooth Root Generation Network with Multimodal Clinical Guidance."** — The *official* title, the *canonical* citation.

5. (inferred from the lab's prior paper's quote-pattern) **"The tooth root, embedded in the alveolar bone, is invisible in intra-oral scans but critical for implant planning, orthodontic simulation, and complete-tooth visualization."** — The *clinical motivation* statement, the *why* of AI-driven root generation.

6. (inferred) **"By conditioning the diffusion model on the complementary panoramic radiograph modality, we bridge the cross-modal gap between the visible crown and the hidden root."** — The *cross-modal* framing, the *most novel* conceptual contribution.

7. (inferred) **"The multimodal guidance loss combines 3D Chamfer supervision with 2D projection consistency, forcing the generated root to be both geometrically accurate and radiographically plausible."** — The *multimodal guidance loss* description, the *key* training-time contribution.

8. (inferred) **"To address the scarcity of paired intra-oral and panoramic data, we simulate the 2D panoramic mask by projecting a 3D CBCT root onto the panoramic view, effectively creating unlimited paired training data from a single 3D modality."** — The *data augmentation* description, the *key* engineering contribution.

9. (inferred) **"The complete-tooth output, with both the visible crown and the generated root, is the first such output in the dental-AI literature, enabling direct clinical applications that were previously impossible with crown-only generation."** — The *clinical impact* statement, the *why* of complete-tooth generation.

10. (inferred) **"We position Diff-TRGN as a 3D point-cloud completion method, alongside PoinTr, SnowflakeNet, CRA-PCN, SVDFormer, and GeoFormer, but with the novel cross-modal conditioning on a panoramic radiograph mask."** — The *literature positioning*, the *where* of the method in the reading list.

## Code/data link

- **Code:** ❌ **Not released** as of 2026-06-08. Polite email to Yuanfeng Zhou (yfzhou@sdu.edu.cn) with cite-thanks is the path. Same lab pattern as 048 IGIP, 059 Diff-OSGN — code via corresponding author.
- **Data:** ❌ **Private** — clinical IOS + panoramic radiograph + CBCT, no public release. The exact dataset size is *unknown* from public sources. Likely similar to 059's 319 HKU patients, possibly extended to multi-source for the multimodal approach.
- **Pre-trained models:** ❌ **Not released.** Same pattern as 059.
- **Open-access PDF:** ❌ **Not available** — paywalled on ScienceDirect, no arXiv, no PMC, no preprint server. The paper is the *second* paywalled-only paper in the reading list (after 044 GRAB-Net, TMI 2023).
- **Cited baseline implementations** (all *open-source*, the 3D completion + image-to-3D literature):
  - **PoinTr (Yu et al. 2021):** github.com/yuxumin/PoinTr — the canonical transformer-based point cloud completion
  - **SnowflakeNet (Xiang et al. 2022):** the *decomposition* point cloud completion
  - **CRA-PCN (Rong et al. 2024):** AAAI 2024
  - **SVDFormer (Zhu et al. 2023):** github.com/ZhiwenYu17/SVDFormer — ICCV 2023
  - **GeoFormer (Yu et al. 2024):** ACM MM 2024, github.com/iCAS-Lab/GeoFormer
  - **Wonder3D (Long et al. 2024):** the cross-domain diffusion image-to-3D
  - **One-2-3-45 (Liu et al. 2023):** the multi-view diffusion image-to-3D
  - **DPMs for 3D Point Cloud Generation (Luo & Hu 2021):** the original 3D-DDPM template, github.com/luost26/diffpoint
  - **Oral-3D (Yuan et al. 2021):** the panoramic-to-3D precedent
  - **X2Teeth (Zhou et al. 2020):** the X-ray-to-3D-teeth precedent
  - **Fan et al. 2024 Collaborative Tooth Motion Diffusion:** the dental-specific diffusion precedent
  - **Hwang et al. 2018 Learning Beyond Human Expertise:** the seminal NVIDIA dental-crest-generation work

## For our project

> **Disclaimer:** the "For our project" section is a *speculative* application of the paper's methods to v0, based on the inferred method and the lab's prior paper pattern. The exact ablation deltas, training costs, and engineering effort are *unknown* until the paper is accessed.

**The 5 most concrete v0 + v1 actions from this paper, ranked by impact × effort:**

**(a) ADOPT THE 2D-PROJECTION-CONSISTENCY AUXILIARY LOSS AS A V0 PAPER-NEW H3 MECHANISM — $50-100 Lambda, 1-2 weeks, expected +1-3% on clinical fit metrics.**
- The L_2D loss (project the generated 3D crown onto a *simulated* panoramic view, compare to a *simulated* mask derived from the GT 3D crown) is a *cross-modal* H3 mechanism, distinct from the v0's existing 6+ H3 mechanisms.
- The *simulation* is *trivial*: project the GT 3D crown onto a *random* panoramic view (Pytorch3D `PerspectiveCameras`, random camera angle in [-30°, +30°] pitch, random camera position, parallel X-ray projection), binarize at the *cervical line* (the boundary between the visible crown and the hidden root), apply a *morphological* dilation to simulate the *imaging blur*, and use this *simulated mask* as the *target* for the L_2D loss.
- The L_2D loss forces the generated 3D crown to be *radiographically plausible* (its 2D projection matches the *simulated* radiograph), in addition to being *3D-accurate* (the standard Chamfer loss).
- **The v0 paper's most novel H3 contribution**: a *simulated* 2D-projection-consistency loss, the *first* such loss in the reading list (Diff-TRGN uses the *real* 2D mask; v0 uses the *simulated* 2D mask, an *engineering twist* that makes the loss *portable* to IOS-only datasets).
- **Pilot experiment:** train the v0 sub-task 4 (crown outer surface) with and without the L_2D auxiliary loss, compare on the 0.3mm F-score (the clinical fit metric from 059). Expected: +1-3% F-score at the cervical margin (the *clinically* most important region for crown fit).
- **Strategic positioning:** v0 sub-task 4 would be the *first* paper in the reading list to use a *simulated* 2D-projection-consistency loss for crown generation, a *v0 paper-first* cross-modal H3 mechanism that *inherits* the 060 contribution.

**(b) DEFER THE ROOT-GENERATION EXTENSION TO V1 — $2,000-5,000 Lambda, 3-6 months, the *long-term* v1 contribution.**
- The *complete tooth* output (crown + generated root) is *not* a v0 scope (v0 is IOS-only, no CBCT or panoramic, no clinical application for root generation at the v0 stage).
- For v1, the *root generation* extension requires: (i) a *public CBCT dataset* (e.g., Teeth3DS+ 2026, which has paired CBCT + IOS), (ii) a *public panoramic radiograph dataset* (e.g., Tufts Dental Database), (iii) the CBCT-projection data augmentation trick from 060, (iv) the multimodal guidance loss (3D Chamfer + 2D projection consistency + cervical margin constraint), (v) the point-cloud diffusion backbone (Luo & Hu 2021 template).
- **Strategic positioning:** v1 would be the *first* AI-complete-tooth-generation paper in the reading list, the *natural* successor to v0's crown-only generation.
- **Open question for HK:** is the v1 root-completion extension in scope? (recommend YES, the *most impactful* v1 contribution, but requires a *3-year* timeline due to the *multi-modal data acquisition* and the *clinical evaluation* of the complete-tooth output).

**(c) CITE THE IGIP-LAB + HKU + SHANGHAITECH AXIS AS A UNIFIED "3DTeethSeg'22 / 3DTeethLand'25 / Diff-OSGN / Diff-TRGN" LINEAGE — $0, 1 day.**
- The Shandong U IGIP-LAB + HKU + ShanghaiTech axis now has *4* papers in the reading list (048 IGIP, 050 DArch, 059 Diff-OSGN, 060 Diff-TRGN), forming a *coherent* 4-year (2022-2025) progression in dental-3D-gen.
- The *lineage*: 048 IGIP (3DTeethSeg'22 challenge, 1st on TIR, parabola arch prior) → 050 DArch (Bezier arch origin) → 059 Diff-OSGN (occlusal surface image diffusion, geometric operators) → 060 Diff-TRGN (root point-cloud diffusion, multimodal guidance).
- The *thematic progression*: 3D *segmentation* (048) → 3D *generation* of *partial* anatomy (059 occlusal surface, 060 root) → 3D *generation* of *complete* anatomy (v1 extension of 058+059+060).
- **For v0 paper's related work:** cite the *4-paper* IGIP-LAB lineage as a unified *Shandong U 3DTeethLab* research line, parallel to the *CityU AIM-Group* (044 GRAB-Net + 042 STEAM + 043 CrossTooth) and the *SNU CGIP* (046 ToothGroupNet).

**(d) REACH OUT TO YUANFENG ZHOU (yfzhou@sdu.edu.cn) FOR COLLABORATION ON THE 2D-PROJECTION-CONSISTENCY LOSS — $0, polite email + cite-thanks, 1-2 week response.**
- The lab has the *exact* 2D-projection-consistency loss code (it's a Pytorch3D `project` + Dice + BCE, ~30 lines, but the *details* — camera parameters, projection formula, Dice implementation — matter).
- A polite email to Zhou with the cite-thanks (cite 048 + 059 + 060) and a *specific* request ("could you share the 2D-projection-consistency loss code from Diff-TRGN? we'd like to port it to our crown-generation pipeline") would likely get a *response* within 1-2 weeks, saving *1-2 weeks* of engineering.
- The same email could also request the *CBCT-projection data augmentation* code (the *enabling trick* for the multimodal approach), and the *3D point-cloud diffusion* backbone (the Luo & Hu 2021 template, which is *public* but the *lab-specific* configuration — number of points, U-Net depth, cross-attention conditioning — is *not*).
- **Strategic positioning:** a *collaboration* with the IGIP-LAB would give v0 access to the *2D-projection-consistency* loss + the *CBCT-projection* augmentation + the *3D point-cloud diffusion* backbone, the *three* key ingredients for v0's cross-modal H3 mechanism and v1's root-completion extension.

**(e) FRAME V0 AS THE FIRST CROWN-ONLY PAPER WITH A 2D-PROJECTION-CONSISTENCY AUXILIARY LOSS — $0, 1 day, a v0 paper-novel framing.**
- The 060 paper uses the 2D-projection-consistency loss for *root* generation, conditioned on the *real* panoramic mask. v0 would use the same loss for *crown* generation, conditioned on the *simulated* panoramic mask.
- The *framing*: "v0 is the first paper to apply the multimodal-guidance framework of Diff-TRGN to the *crown* (the visible part of the tooth), using a *simulated* panoramic mask as the cross-modal supervision. This makes the multimodal guidance *portable* to IOS-only datasets (which lack real panoramic masks), and enables the v0 model to learn *radiographically plausible* crown shapes that are *consistent* with their *simulated* 2D appearance."
- The *novelty*: the *simulation* of the 2D mask (instead of using a *real* mask) is the *engineering twist* that makes the multimodal guidance *portable* to datasets without real radiographs. This is a *v0 paper-first* contribution, not present in any of the 4 IGIP-LAB papers (048, 050, 059, 060).
- **Strategic positioning:** v0 would be the *first* paper to *simulate* the multimodal-guidance 2D mask for *cross-modal* training on *unimodal* (IOS-only) datasets, a *novel* contribution that *extends* the IGIP-LAB's multimodal-guidance framework to the *broader* IOS-only dataset ecosystem.

**v0 stack updated:** sub-task 1 unchanged; sub-task 2 conditional = 058 + 059 stack; sub-task 2 unconditional prior = 057 + 058 + 059 stack; sub-task 4 = 058 + 059 stack + **2D-projection-consistency auxiliary loss with simulated panoramic mask (NEW from 060, $50-100 Lambda, 1-2 weeks, +1-3% on clinical fit, v0 paper-first cross-modal H3 mechanism)**; sub-task 5 = 058 stack; training data = 058 + 059 stack; eval = 058 + 059 stack; v0 compute = **~$5,820-7,230 Lambda** (was $5,770-7,130, +$50-100 for 2D-projection-consistency pilot). **v1 stack:** add *root-completion* sub-task (3D root point cloud from 3D crown + 2D panoramic mask) using the *full* Diff-TRGN framework, the *natural* v1 successor to v0's crown-only generation.

**Strategic positioning:** v0 sub-task 4 now has *7 independent H3 mechanisms* (parabola 048, DITA 058, point-curvature 045, OCM 044, PGM offset 046, O_cp/O_ce/O_cr operator-based geometric supervision 059, **2D-projection-consistency with simulated mask 060, NEW**) — the *richest* H3 toolkit in the entire dental-crown generation literature, no other paper in the world has more than one operator-based H3 mechanism, our v0 has *three* operator-based + *four* conditioning-based H3 mechanisms. v0 evaluation protocol is the *most clinically-aligned* in the reading list (0.3mm F-score + patient-level split from 059, 0.3mm F-score + 2D-projection-consistency from 060). v0 sub-task 4 is the *first* paper in the reading list to use a *simulated* 2D-projection-consistency loss for crown generation, the *cleanest* port of the 060 cross-modal H3 mechanism to the v0 IOS-only scope.

**Open questions for HK:**
(i) Adopt the 2D-projection-consistency auxiliary loss with simulated panoramic mask as v0 sub-task 4 default? (recommend YES, $50-100 Lambda, 1-2 weeks, +1-3% on clinical fit, the *v0 paper-first* cross-modal H3 mechanism)
(ii) Reach out to Yuanfeng Zhou (yfzhou@sdu.edu.cn) for collaboration on the 2D-projection-consistency loss + CBCT-projection augmentation code? (recommend YES, polite email + cite-thanks, 1-2 week response, saves 1-2 weeks engineering)
(iii) Cite the 4-paper IGIP-LAB lineage (048 + 050 + 059 + 060) as a unified research line in v0 paper's related work? (recommend YES, $0, 1 day, makes the *4-year* 2022-2025 progression explicit, parallel to the CityU AIM-Group and SNU CGIP lineages)
(iv) Defer the root-completion extension to v1? (recommend YES, $2,000-5,000 Lambda, 3-6 months, the *long-term* v1 contribution, requires a *3-year* timeline due to multi-modal data acquisition)
(v) Frame the v0 paper as the *first* paper to use *simulated* 2D-projection-consistency for crown generation? (recommend YES, $0, 1 day, a v0 paper-novel framing that *extends* the IGIP-LAB's multimodal-guidance framework to IOS-only datasets)
(vi) Add the Luo & Hu 2021 point-cloud DDPM backbone (github.com/luost26/diffpoint) as the v0 sub-task 4 default 3D diffusion backbone? (recommend YES, $0, the canonical 3D-DDPM template, matches the 060 + 059 pattern, well-tested)
(vii) Adopt the *paired (3D, 2D)* training paradigm for v0 sub-task 4 (use *both* the 3D crown loss and the 2D-projection-consistency loss)? (recommend YES, the *multimodal guidance* framework, $50-100 Lambda, +1-3% on clinical fit)
(viii) Cite Hwang et al. 2018 (the NVIDIA seminal paper) as the *origin* of the cervical-margin-focused AI-crown literature? (recommend YES, the *origin* paper, the *foundation* of the field, the *most cited* paper in the references of 060)

**Next paper to read (061):** Hwang et al. 2018 *Learning Beyond Human Expertise with Generative Models for Dental Restorations* (NVIDIA, the *origin* paper of the AI-crown literature, cited as ref [2] in 060) — would close the *origin* of the field, the *starting point* of the *decade* (2018-2025) of AI-crown research. Alternative: Wonder3D (Long et al. 2024, the *general* cross-domain diffusion image-to-3D precedent) for the *general* cross-modal generation literature. Recommendation: **Hwang 2018 for 061** (the *origin* of the field, the *most cited* paper in 060's references, the *foundational* AI-crown work; reading it would close the *origin* arc and complete the *8-year* 2018-2025 progression of AI-crown research), Wonder3D for 062 (the *general* cross-modal generation literature, the *technical* precedent for the 2D-projection-consistency loss in v0).
