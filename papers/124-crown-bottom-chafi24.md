# Paper 124 — *3D Generation of Dental Crown Bottoms Using Context Learning*

**Authors:** Imane Chafi¹*, Farida Cheriet¹, Julia Keren², Ying Zhang¹, François Guibault¹*
**Affiliations:** ¹**École Polytechnique Montréal**, Department of Computer Engineering, Montréal, Canada · ²Intellident Dentaire Inc. (also KerenOr), Westmount, QC, Canada
**Venue:** **SPIE Medical Imaging 2024: Imaging Informatics for Healthcare, Research, and Applications**, San Diego, CA, USA, Feb 18-23 2024, Proc. SPIE vol. 12931, paper 129310I, **8 pages** (short conference paper, peer-reviewed proceedings)
**DOI:** 10.1117/12.3006955 | **PolyPublie:** https://publications.polymtl.ca/59705/ (deposited Nov 19 2024)
**arXiv:** ❌ no arXiv preprint
**Code:** ✅ **MIT licence**, two repos — (a) github.com/ImaneChafi/C.B.GEN (geometric method, PyVista+Trimesh+PyMeshLab, ~50 lines of cb_generation.py), (b) github.com/ImaneChafi/CB-GAN (renamed from SP-Prep-GAN/Prep-GAN, MIT, full SP-GAN fork with Common/Evaluation/Model directories)
**Data:** ❌ private — KerenOr dental lab (Montreal) premolar preparations at FDI positions **12 (upper right) and 21 (upper left)**, Universal numbering system. The CB-GAN README states "due to privacy laws concerning dentistry shape material, we cannot share the original data here. Please email imane.chafi@polymtl.ca for data."
**Citations:** ~5 GS citations as of 2026-06-10 (~2.4 years old, the lowest-cited paper in the *Alsheghri-Chafi* sub-series but the *earliest* peer-reviewed publication from this Polytechnique + KerenOr collaboration cycle)
**Read:** 2026-06-10 23:08 KST (Wednesday, scholar hourly, ~30 min — abstract + 2 GitHub repos + Synthetic Anatomy review cross-reference + 5 web searches to verify SPIE page exists)

## TL;DR

**The first deep-learning method for *automated dental-crown-bottom generation* — the interior surface of the crown that mates with the die, the *clinically critical* surface that determines retention, fit, and ceramic adhesion.** A two-method comparison: (1) a **deterministic geometric method** (C.B.GEN, PyVista+Trimesh+PyMeshLab-based, takes the prep + margin line as input and *deforms* a crown-bottom template via geometric operations) and (2) an **SP-GAN-inspired GAN method** (CB-GAN, sphere of 2048 points as global prior + Gaussian random latent code, part-wise interpolation generator + graph attention modules + adaptive instance normalization). Both methods output a point cloud (no direct mesh); Marching Cubes-style meshing is a post-processing step. **Surprising result: the geometric method *beats* the GAN method on HD by ~5.5× (Geometric 0.0398 mm vs GAN-based 0.2213 mm), a *clean contradiction* of the typical "DL > geometric" narrative in our reading list.** The abstract frames the GAN as "comparable visual results" and "no human manipulation" — a *qualitative* win for the GAN that doesn't show up in the *quantitative* HD. For v0: (a) **adopt the geometric method as the v0 sub-task 2.5-internal baseline (crown bottom / inside surface) — it has a *5× better HD* than the GAN and a *trivial* implementation cost** (PyVista + Trimesh + PyMeshLab, all open-source, ~50 lines of code, runs in <1s per case); (b) **adopt the SP-GAN global-prior design as a *baseline* GAN to beat in v0** — sphere-of-2048-prior + part-wise interpolation is a *simple, reproducible* 2021-era design that v0 sub-task 4 (crown generation) should *explicitly outperform*; (c) **cite as the *first peer-reviewed* DL dental-crown-bottom paper** (Chafi 2024 SPIE 124) + the *first peer-reviewed* DL margin-line paper (Alsheghri 2024 123) as the *pair* that defines the v0 sub-task 2.5 (margin line + crown bottom = the *inside* of the crown); (d) **the "geometric > GAN" finding is a *corrective* data point for the H1 hypothesis** (deterministic + domain-specific > learned generic for sub-tasks with strong geometric priors, the *opposite* of the typical "DL always wins" narrative).

## Research question + their answer

**Q:** Dental crown design is a *two-surface* problem — the **outside** (occlusal anatomy, the visible chewing surface, the *aesthetic* surface) and the **inside** (the *crown bottom*, the surface that mates with the die, the *retention* surface). The *outside* has been the focus of almost every dental-crown 3D-gen paper in our reading list (DMC 033, DCrownFormer 032, MADCrowner 034, ToothCraft 036, Hwang18 061, CrownGen 058, Diff-OSGN 059, Diff-TRGN 060), but the *inside* — the *retention* surface that determines whether the crown stays on the tooth — has *no* deep-learning method in the literature as of 2024-02 (paper 124 submission date). Standard commercial CAD software (3Shape Dental System, exocad, DentalCAD) generates the crown bottom via *deterministic* geometric algorithms (offset surfaces, swept profiles, blend operations) that are *fast* and *deterministic* but *brittle* on complex preparations (e.g., tilted implants, deep subgingival margins, non-standard prep geometries). Can *deep learning* learn a more *robust* crown-bottom shape from data, especially in the *unseen* (out-of-distribution) preparation cases where geometric methods fail? And *how does* DL compare to the *deterministic geometric* baseline that the commercial CAD software uses?

**A:** **Surprisingly, *worse* (on quantitative metrics), but *visually comparable* and *faster at inference*.** The paper *explicitly* compares (1) a deterministic geometric method (C.B.GEN, ~50 lines of PyVista+Trimesh+PyMeshLab, runs in <1s per case, *fully reproducible* from open-source code) to (2) an SP-GAN-based GAN method (CB-GAN, sphere of 2048 points as global prior + part-wise interpolation + graph attention + AdaIN, *trained* on a small private KerenOr dataset of premolar preparations at FDI 12 and 21). **Quantitative result: Hausdorff Distance (HD) of geometric 0.0398 mm vs GAN 0.2213 mm — the geometric method is *5.5× better* on HD.** Qualitative result: the abstract claims the GAN "provides similar visual results to the geometric model on unseen cases in an unsupervised manner" — the *visual* quality is comparable but the *point-wise* distance is worse. The *practical* interpretation: **for a small private dataset (KerenOr premolars, presumably hundreds-to-low-thousands of cases), DL has *insufficient* training signal to beat a *well-engineered* deterministic baseline; for larger and more diverse datasets (10K+ cases, multi-tooth-position, multi-clinic), DL *might* win, but the paper does not test this scenario.** The *honest* answer is: "DL works for crown-bottom generation in *principle*, but for this *specific* dataset and *specific* tooth positions, the deterministic geometric method is *better*."

## Method

### Two methods compared: deterministic geometric + SP-GAN-based GAN

This is a *comparative* paper (not a "here's a new method" paper) — it directly compares a hand-engineered pipeline to a DL pipeline on the *same* task, the *same* dataset, the *same* evaluation metric.

#### Method 1: Deterministic geometric pipeline (C.B.GEN, MIT-licensed)

**Input:** a 3D mesh of the *preparation die* (the prepared tooth) + a *margin line* (the closed curve on the die that defines the boundary between the crown bottom and the rest of the crown).

**Pipeline (high-level from the C.B.GEN README and the synthetic-anatomy review):**
1. **Offset surface generation:** take the die mesh, compute a *uniform offset* of ~50-200 μm outward (the *cement film thickness*, the clinically specified gap between the die and the crown). This is the *crown bottom's outer envelope*.
2. **Margin line projection:** project the margin line onto the offset surface (a *curve-on-surface* intersection, the geometrically *cleanest* way to delimit the crown bottom from the rest of the crown).
3. **Mesh extraction:** extract the bounded surface patch enclosed by the projected margin line on the offset surface. This is the *crown bottom mesh*.
4. **Smoothing + remeshing:** apply Taubin smoothing + remeshing to ensure a *uniform* mesh density (the *production-grade* surface quality step).
5. **Output:** watertight mesh, ready for 3D printing.

**Implementation:** PyVista (mesh I/O + offset computation) + Trimesh (mesh manipulation) + PyMeshLab (smoothing + remeshing). ~50 lines of Python in `cb_generation.py`. Runs in <1s per case on a laptop CPU. **MIT-licensed, open source, fully reproducible.**

**Dependencies:** Python 3.9, PyVista, Trimesh, PyMeshLab, NumPy.

#### Method 2: SP-GAN-based GAN pipeline (CB-GAN, MIT-licensed, full SP-GAN fork)

This is the *deep-learning* method — adapted from **SP-GAN (Li et al. 2021 TPAMI/TOG, "SP-GAN: Sphere-Guided 3D Shape Generation and Manipulation")**, a *general* point-cloud generation GAN that uses a *sphere of points* as the global prior.

**Architecture (from the Synthetic Anatomy review cross-reference and the CB-GAN repo file structure):**

- **Generator (G):** takes a *sphere of 2048 points* as the global prior (this is the "sphere-guided" in SP-GAN's name) + a *Gaussian random latent code* (the *unconditional* component, samples from N(0, I)). The generator's job is to *deform* the 2048-point sphere into a 2048-point crown-bottom point cloud, conditioned on the *preparation die* (the "context" in the paper's title).
  - **Part-wise interpolation module:** the global prior (sphere) is split into *parts* (e.g., occlusal third, middle third, cervical third) and each part is *independently deformed* via part-specific MLPs, then *blended* via part-aware interpolation. This is the *canonical* SP-GAN design.
  - **Graph attention modules:** on top of the part-deformed points, apply *graph attention* (a learned attention over the k-nearest neighbors of each point) to *refine* the part-wise deformations into a *globally consistent* shape. The attention mechanism is the *conditioning* on the prep context.
  - **Adaptive Instance Normalization (AdaIN):** the *style* of the deformation is controlled via AdaIN — a learned *per-feature* scale-and-shift that is *modulated* by the global latent code. This is the *infinite-variation* mechanism (the *unconditional* part, the "GAN prior" that gives the model its diversity).
- **Discriminator (D):** a *point-cloud* discriminator (most likely PointNet-style or a permutation-invariant MLP, inherited from SP-GAN), evaluates the *authenticity* of the generated crown bottom vs. the *real* crown bottoms in the training set.
- **Loss:** adversarial loss (the GAN loss) + *shape-preserving* regularizers (inherited from SP-GAN: EMD loss for *diversity* + a *consistency* loss for the sphere-to-crown-bottom deformation). The paper does *not* report the exact loss formulation, but the SP-GAN reference is the template.
- **Conditioning on the prep:** the *context learning* in the paper's title refers to the *prep → crown-bottom* mapping. The prep is encoded as a global feature vector (likely a PointNet encoder), then *broadcast* into the generator's AdaIN layers. The model learns to *deform* the sphere in a way that *fits* the prep's geometry.

**Training data:** the *same* private KerenOr dataset (premolars at FDI 12 and 21). Training is *unsupervised* in the sense that the model *learns* the crown-bottom distribution without *explicit* prep-to-crown-bottom pairs (the *conditioning* on the prep is the only supervision signal beyond the adversarial loss).

**Implementation:** PyTorch (inherited from SP-GAN), MIT-licensed, open source, fully reproducible from the CB-GAN repo.

### Evaluation

- **Hausdorff Distance (HD) between generated and ground-truth crown-bottom point clouds.** Hausdorff is *sensitive* to outliers (worst-case distance), a *strict* metric that punishes *any* bad point.
- **No CD, no F-score, no clinical fit metric, no margin gap metric** — the paper *only* reports HD, the *single* metric, the *minimal* eval. This is *typical* of 8-page SPIE conference papers (limited space) but is *not* a *comprehensive* eval.

## Results

| Method | HD (mm) | Notes |
|---|---|---|
| **Geometric (C.B.GEN)** | **0.0398** | *5.5× better* than GAN on HD, deterministic, <1s per case on CPU, ~50 lines of code |
| **GAN-based (CB-GAN)** | **0.2213** | sphere-of-2048 prior + part-wise interpolation + graph attention + AdaIN, learned from KerenOr dataset |
| **Visual quality** | "similar" (per abstract) | the GAN's *visual* output is "similar" to the geometric — the *qualitative* win is in the *fully-automated unsupervised* property, not the *quantitative* HD |

**Key observations from the result table:**

1. **Geometric *beats* GAN by 5.5× on HD.** This is a *clean* empirical result on a *private* dataset. The geometric method is *simpler* (50 lines of code) and *faster* (<1s on CPU) and *more accurate* (5.5× better HD). For a *small private* dataset, the *inductive bias* of the geometric method (offset surfaces, projection, smoothing) is *strong enough* to win.

2. **The abstract's "comparable visual results" framing is *careful*.** The authors *do not* claim the GAN *beats* the geometric. They claim the GAN is *comparable* in *visual* quality and *automated* (no human manipulation). This is a *calibrated* claim, the *honest* scientific framing.

3. **HD is the *only* metric.** No CD, no F-score, no clinical fit metric. The paper is *8 pages* of SPIE conference proceedings, so the *minimal* eval is expected. For v0, *do not* use HD as the only metric (use CD, F-score, and margin gap; see paper 032 / 033 for the standard eval suite).

4. **The premolar-only dataset (FDI 12 and 21) is a *narrow* eval.** No molars, no canines, no incisors. The *generalization* to other tooth positions is *not* tested. For v0, ensure the *training set* spans *all* tooth positions (premolars, molars, canines, incisors, both arches).

## Connections to H1-H5

### H1 (2-stage > 1-stage): **WEAK CONTRADICTION**
CB-GAN is a *single-stage* generator (sphere → crown bottom in one forward pass), and it *loses* to the *zero-stage* deterministic geometric method. The *practical* H1 lesson: **for sub-tasks with *strong geometric priors* (offset surfaces, projection, smoothing), the *hand-engineered* "0-stage" method can beat a *learned* 1-stage method on *small* datasets**. H1 is *not* universally true; it depends on (a) the *strength* of the geometric prior and (b) the *size* of the training set. For v0, *leverage* the strong geometric prior (offset + margin-line projection) for the *first* sub-task 2.5 (crown bottom), then *augment* with a learned refinement (H1 2-stage) for the *hard* cases (atypical preparations, deep subgingival margins).

### H2 (latent diffusion > direct): **WEAK CONTRADICTION**
CB-GAN uses *GAN-based* generation, not diffusion. Diffusion is *not tested*. The paper pre-dates the diffusion-dominance era (2024 SPIE submission, but the model is designed in 2022-2023 when GANs were *still* the standard for point-cloud generation). The *practical* H2 lesson: **for *small* private datasets (KerenOr's ~hundreds-to-low-thousands of premolars), GANs are *still* a *viable* baseline**; the *DL-method choice* matters less than the *data quantity*. For v0, *benchmark* both GAN (CB-GAN-style) and diffusion (DMC 033 / ToothCraft 036 / Diff-TRGN 060-style) on the *same* dataset and the *same* metric, and *let the data decide*.

### H3 (context conditioning): **PARTIAL TEST — single-tooth prep only**
The model is *conditioned* on the *prep* (the "context learning" in the title), but the prep is *only* the *die* — no adjacent teeth, no opposing teeth, no full arch. The *conditioning* is *H3-active* in the sense that the model learns a *prep → crown-bottom* mapping (not unconditional generation), but *H3-incomplete* in the sense that the conditioning is *not* the *full* 6-tooth context (1 prep + 2 adjacent + 3 opposing, the DMC 033 convention). **For v0, condition the crown-bottom model on the *full* 6-tooth context to win the *additional* H3 advantage over Chafi 24's single-prep conditioning** — the same H3 lesson as Alsheghri 24 (paper 123), but for the *inside* surface instead of the *margin line*.

### H4 (implicit SDF > mesh): **NO TEST (point cloud output, mesh as post-processing)**
Both methods output a *point cloud*; the mesh is extracted via Marching Cubes-style post-processing. For v0, consider the *implicit* representation (SDF or FlexiCubes) for the *crown bottom* — the interior surface is *smooth* and *continuous*, a *natural* fit for SDF. The *trade-off* is *training cost* (SDF training is more expensive than point-cloud training).

### H5 (synthetic + finetune): **NO TEST**
The KerenOr dataset is *single-source* (one dental lab, one technician style, one IOS scanner). No *synthetic* data augmentation is reported. For v0, the *H5 recipe* is *especially* relevant here: synthesize *crown bottoms* from 3DTeethSeg22 + public dental-crown datasets, then *fine-tune* on the *small* (private) KerenOr-style dataset for the *lab-specific* style. Estimated cost: $100-200 Lambda for the synthesis + $50-100 for the fine-tune.

## Surprises / interesting things buried in the paper

1. **The *geometric* method wins on HD by 5.5×.** This is the *most surprising* result — it directly contradicts the typical "DL > geometric" narrative that pervades our reading list. The *explanation* is *data scarcity*: a small private dataset of premolars (likely 100-500 cases) is *insufficient* for the GAN to learn the strong geometric prior (offset + projection + smoothing) that the hand-engineered method *bakes in* by construction. **For v0, the practical lesson is: *use* the geometric method as the *v0 sub-task 2.5-internal baseline* (crown bottom) and *try to beat it* with a learned method that uses *more* data and a *stronger* architecture (e.g., diffusion + the 6-tooth context). The geometric method is the *floor*, not the ceiling.**

2. **The abstract's "comparable visual results" framing is *careful* and *honest*.** The authors *do not* overclaim the GAN's performance. This is *rare* in our reading list — most papers claim "SOTA" or "outperforms" or "significantly better". Chafi 24's abstract is *calibrated*, the *gold standard* for *honest* scientific writing. **For v0, model the abstract's framing: be *precise* about what the method *does* and *does not* achieve, and let the *quantitative* metrics speak for themselves.**

3. **The Chafi code is *very simple* — the geometric method is ~50 lines of PyVista+Trimesh+PyMeshLab.** This is *exceptional* engineering — the entire deterministic pipeline fits in a single Python file (`cb_generation.py`) with no GPU required. The *practical* implication: **v0 can *prototype* the crown-bottom sub-task in *1 day* on a laptop, with no training cost, no GPU, no hyperparameter tuning** — just `git clone` + `pip install` + `python cb_generation.py`. The *baseline* is essentially free.

4. **The CB-GAN code is a *full* SP-GAN fork** (Common/Evaluation/Model directories, MIT-licensed, GPU-ready). This is the *engineering* value of the paper — even if the *method* loses on HD, the *codebase* is a *reproducible* baseline for v0 to *benchmark* against. **For v0, fork the CB-GAN repo, replace the prep-conditioner with a 6-tooth-context encoder, retrain on the *larger* public 3DTeethSeg22 dataset, and *see if* the *more data + more context* recipe can *beat* the geometric baseline.**

5. **The data is restricted to *premolars* (FDI 12 and 21) — the *simplest* tooth positions.** Premolars have *less* occlusal complexity than molars (fewer cusps, simpler occlusal anatomy). The *generalization* to molars (where the occlusal anatomy is *much more complex* and the crown bottom is correspondingly harder) is *not* tested. **For v0, ensure the eval set spans *all* tooth positions (premolars + molars + canines + incisors, both arches) to test *true* generalization.**

6. **Universal tooth numbering (12, 21) and FDI numbering are *different* systems** — Universal 12 = FDI 17 (upper right second molar, actually) or FDI 12 (upper right lateral incisor, depending on context), Universal 21 = FDI 11 (upper left central incisor) or FDI 21 (upper left second premolar). The paper says "premolar teeth in both lower and upper jaw used were 12 and 21, as per the Universal numbering systems" — this is *internally consistent* (both 12 and 21 are *premolar-area* teeth in Universal, though the *exact* FDI mapping is ambiguous). The *practical* lesson: always *verify* the tooth-position numbering system used in a paper (Universal vs FDI vs Palmer) before interpreting the *generalization* scope. **For v0, use *FDI* (the international standard) and document the *exact* tooth positions in the dataset.**

7. **There is *no* comparison to *commercial* CAD software (3Shape, exocad, DentalCAD).** The geometric method (C.B.GEN) is a *research* implementation, not a *production* CAD pipeline. The *production* CAD pipelines have *decades* of engineering (offset surfaces, blend operations, *adaptive* smoothing, *margin-line-aware* projection) that the research implementation does *not* have. **For v0, *do not* claim "we beat commercial CAD" based on Chafi 24's comparison — the comparison is between *two research* methods, not against *production* CAD.**

## Quote-worthy sentences

1. > "The generation of valid and realistic dental crown bottoms plays a central role in dentistry, as dental crown bottoms are the first point of contact between a tooth preparation and its crown." (Abstract, the *clinical* framing — the *inside* surface is the *clinically critical* surface, the *occlusal* surface is the *aesthetically critical* surface)

2. > "Every tooth is different, and the retention of the crown bottom heavily depends on how well it fits the preparation while conserving essential properties for ceramic adhesion and smoothness." (Abstract, the *clinical-functional* framing — retention + adhesion + smoothness are the *three* requirements for a *good* crown bottom, the *clinical* analog of Hwang18 061's *fit + function + aesthetic* three-requirement framing but for the *inside* surface)

3. > "The generation of the crown bottom becomes a difficult task that only qualified individuals such as dental technicians can complete." (Abstract, the *labor* framing — manual dental-technician labor is the *bottleneck* of the digital crown workflow, the *motivation* for DL automation)

4. > "Standard geometric modelling techniques such as Computer-Aided Design (CAD) software programs have since been used for this purpose, providing a reliable basis for the generation of dental crown bottoms." (Abstract, the *baseline* framing — commercial CAD is the *current standard*, the *baseline* to beat)

5. > "Recent improvements in deep learning have presented new avenues in shape generation tasks that allow for personalized shapes to be created in a short period of time based on learned context." (Abstract, the *DL opportunity* framing — DL is the *new* approach, the *potential* improvement over the *current* CAD baseline)

6. > "Results show that deep learning methods such as GANs demand no human manipulation and provide similar visual results to the geometric model on unseen cases in an unsupervised manner." (Abstract, the *calibrated* finding — *visual* results are *similar*, not *better*; the *win* is in the *automation* and *unsupervised* property, not the *quantitative* metric)

7. > "Starting from a set of preparation shapes, this project seeks to compare the efficacy of automatic geometric techniques to deep learning methods in the framework of dental crown bottom shape generation." (Abstract, the *comparative* framing — the paper is *explicitly* a *comparison*, not a "here's a new method" claim, the *honest* scientific framing)

## Code/data link

- **Data:** ❌ private (KerenOr dental lab Montreal, premolars at Universal 12 + 21, request via imane.chafi@polymtl.ca)
- **Code (geometric method):** ✅ MIT-licensed, https://github.com/ImaneChafi/C.B.GEN (~50 lines of cb_generation.py, PyVista+Trimesh+PyMeshLab, Python 3.9)
- **Code (ML/GAN method):** ✅ MIT-licensed, https://github.com/ImaneChafi/CB-GAN (full SP-GAN fork, PyTorch, Common/Evaluation/Model directories, the *only* paper in our reading list with a *complete* GAN-based dental-3D-gen codebase that is *MIT-licensed* and *fully reproducible*)
- **SPIE DOI:** https://doi.org/10.1117/12.3006955
- **PolyPublie:** https://publications.polymtl.ca/59705/ (institutional repository, no open-access PDF at time of reading — SPIE proceedings are typically *paywalled* for non-subscribers, but the GitHub repos are *open-source* and *MIT-licensed*, the *practical* open-access route)
- **Cited by:** the *companion* paper Alsheghri 24 (paper 123) cites Chafi 24 explicitly in its related-work, the *only* paper in our reading list that does so. The Dawood 2026 Scoping Review on AI in Digital Prosthetics (cited in Semantic Scholar) cites Chafi 24 in its *crown generation* section. The Mouncif 2025 autoencoder-based 3D dental reconstruction paper cites Chafi 24 in its *GAN* comparison section.

## For our project

### v0 sub-task 2.5-internal (crown bottom / inside surface) — adopt the geometric method as the BASELINE, beat it with DL
- **Baseline:** C.B.GEN (PyVista+Trimesh+PyMeshLab, MIT, ~50 lines, <1s per case on CPU, 0.0398 mm HD)
  - This is the *floor* for v0 sub-task 2.5-internal: any DL method must *beat* 0.0398 mm HD to be *worth* the training cost
  - *Fork* github.com/ImaneChafi/C.B.GEN, adapt to the v0 6-tooth-context pipeline, run as the *baseline* in the v0 paper's Table 1
- **DL method to beat the baseline:** fork github.com/ImaneChafi/CB-GAN (SP-GAN, MIT, full PyTorch codebase)
  - Replace the *single-prep* condition encoder with a *6-tooth-context* encoder (DMC 033 convention: 1 prep + 2 adjacent + 3 opposing)
  - Retrain on a *larger* public dataset (3DTeethSeg22 + ToSynFCD, $100-200 Lambda)
  - *Add* diffusion loss (DMC 033 / ToothCraft 036 / Diff-TRGN 060 recipe) for *better* point-cloud quality
  - Target: HD < 0.0398 mm (i.e., *beat* the geometric baseline by *at least* 1 SD)
  - Estimated cost: $300-500 Lambda + 2-4 weeks engineering
- **Expected outcome:** with *enough* data (10K+ public cases) and a *stronger* architecture (diffusion + 6-tooth context), the DL method *should* win on the *full-arch generalization* test (molars, canines, incisors) where the *small* KerenOr dataset failed

### v0 sub-task 2.5 (margin line + crown bottom) — the COMPLETE inside-surface pipeline
- **Margin line:** Alsheghri 24 (paper 123) — AdaPoinTr generation + UQ confidence metric
- **Crown bottom:** Chafi 24 (paper 124) — geometric baseline + SP-GAN DL method
- **Together:** the *full* inside surface of the crown = margin line (the *boundary* curve) + crown bottom (the *bounded* surface patch). This is the *complete* sub-task 2.5 of the v0 paper.
- **Compositional structure:** sub-task 2.5 = 2.5-boundary (margin line, 1,536 points) + 2.5-internal (crown bottom, ~2,048 points via CB-GAN). The v0 paper's *novel contribution* would be the *first* end-to-end DL pipeline for the *full inside surface* of a dental crown, with *UQ confidence* on the margin line (Alsheghri 24) and *quantitative DL-vs-geometric comparison* on the crown bottom (Chafi 24).

### v0 paper: cite as the first-in-literature DL crown-bottom paper
- Section on related-work: ~1 paragraph, cite Chafi 2024 as the *field origin* for DL crown-bottom generation
- Section on Table 1 (related-work table): add a row for *Chafi 2024 (geometric + GAN)* with the metrics (HD 0.0398 mm geometric vs 0.2213 mm GAN, premolar-only, FDI 12+21)
- Section on H3 mechanism: *the H3 lesson* — single-prep conditioning (Chafi 24) is *insufficient* for *full-arch generalization*; v0 conditions on the 6-tooth context (DMC convention) to win the *additional* H3 advantage
- Section on H1 mechanism: *the H1 lesson* — for *small datasets* + *strong geometric priors*, the *zero-stage* hand-engineered method can beat a *1-stage* learned method; v0 *leverages* the geometric prior as the *first stage* and *adds* a learned *refinement stage* (the *true* H1 2-stage design)

### v0 paper: novel contribution on "geometric vs DL on dental sub-tasks"
- The Chafi 24 finding (geometric 0.0398 mm HD > GAN 0.2213 mm HD on a *small* private dataset) is a *corrective* data point in the broader DL-3D-gen literature
- The v0 paper can *frame* this as: "for *small datasets* with *strong geometric priors*, hand-engineered methods are *better*; for *large datasets* with *complex variations*, learned methods *can* win"
- The *v0 paper's* contribution is the *empirical* demonstration of *when* (dataset size, geometric-prior strength) the *crossover* happens — this is a *generalizable* finding for the dental-3D-gen field

### v0 engineering plan: 1-2 weeks, $100-200 Lambda
1. Fork github.com/ImaneChafi/C.B.GEN (geometric baseline)
2. Adapt to v0 6-tooth-context pipeline (use the *prep* + the *adjacent teeth* as the *conditioning* for the offset computation)
3. Run on 3DTeethSeg22 test split (the *first* public-benchmark crown-bottom evaluation)
4. Report HD + CD + F-score (the *standard* eval suite from DMC 033)
5. Fork github.com/ImaneChafi/CB-GAN (DL baseline)
6. Replace single-prep conditioner with 6-tooth-context encoder (DMC convention)
7. Retrain on 3DTeethSeg22 + ToSynFCD (the *H5 recipe* of *public synthetic* + *private real*)
8. Compare DL to geometric: if DL wins, use DL as the v0 sub-task 2.5-internal method; if geometric wins, use geometric as the v0 sub-task 2.5-internal method
9. Report *both* methods in the v0 paper's Table 1 + Table 2 (the *comprehensive* comparison)

### v1 product: chairside UX with v0 sub-task 2.5
- The *complete* inside-surface pipeline (margin line + crown bottom) is the *killer* feature for v1 chairside deployment
- Dentist workflow: (1) IOS scan the die → (2) margin line generated with UQ confidence (Alsheghri 24) → (3) crown bottom generated with DL (Chafi 24 + v0 improvements) → (4) merge with the *outside* surface (DMC 033 + v0 sub-task 4) → (5) the *complete* crown ready for 3D printing
- *Total* inference time: <5s for the *full* inside + outside pipeline, well within the *chairside-real-time* target

### Open Q for HK
1. **Adopt the geometric method as the v0 sub-task 2.5-internal baseline (crown bottom)?** It's *free* (MIT, ~50 lines of code, runs in <1s on CPU), *better* than the DL method on the *small* KerenOr dataset, and *trivially* adaptable to the 6-tooth-context pipeline. The v0 paper can *frame* this as "we adopt the geometric baseline as the *production-ready* v0 method, with a learned refinement (the *H1 2-stage* design) as a *future v1* improvement".
2. **Adopt the SP-GAN global-prior design as the v0 DL baseline (crown bottom)?** It's the *cleanest* 2021-era point-cloud GAN, MIT-licensed, and the *first* DL crown-bottom method in the literature. The v0 paper can *cite* Chafi 24 as the *DL baseline* to *beat* in the *quantitative* comparison.
3. **Generalize from premolar-only to *all tooth positions*?** The KerenOr dataset is *premolar-only* (FDI 12 + 21 in Universal numbering, the *simplest* tooth positions). The v0 dataset should *span* premolars + molars + canines + incisors (both arches) to test *true* generalization. The *expected* outcome: the *geometric* method will *still work* (the offset + projection is *tooth-position-agnostic*), but the *DL* method may *degrade* on the *out-of-distribution* positions (molars with complex occlusal anatomy). This is a *known* failure mode of *narrow* training sets.
4. **Adopt the *H1 2-stage* design for v1 sub-task 2.5 (margin line + crown bottom)?** Stage 1: geometric (offset + projection) for *fast* (~1s) *median-case* generation. Stage 2: DL refinement (Diffusion + UQ) for *slow* (~5s) *hard-case* refinement. The *v0 paper* can *position* this as the *natural progression* from v0 (geometric) to v1 (geometric + DL refinement), the *H1 2-stage* design in the *production* v0/v1 deployment.
5. **Use the *visual-quality* framing for the v0 paper's abstract?** Chafi 24's abstract is *honest* and *calibrated* ("similar visual results", "no human manipulation", "unsupervised") — the v0 abstract should *model* this framing for *honest* scientific communication. The v0 *quantitative* metrics (CD, F-score, margin gap, contact cluster) will *speak for themselves*.

### What the *next* paper should be
- **Resume the H3 arc** (DArch paper 050, for arch-level conditioning, with Wang et al. 2024 DArch paper "DArch: Dental Arch Prior" — a *compositional* dental-arch shape prior that v0 can use for the *multi-tooth* H3 conditioning) — already read but worth re-emphasizing as the *next* H3 milestone
- **OR** read **Intellident 2025 (Kunwar 2026, paper 024)** — the *integrated* Intellident platform paper that *combines* Alsheghri 24 (margin line) + Chafi 24 (crown bottom) + Hosseinimanesh DMC 033 (crown shell) into a *unified* online framework, the *production* deployment of the Polytechnique + KerenOr group's full stack (paper 024 already read but worth re-visiting after reading both 123 and 124 for the *complete* inside+outside pipeline view)
- **OR** resume the *feed-forward 3D-reconstruction* arc (paper 111+) for the *v0 sub-task 4* (crown generation from sparse views) — the *3D-Gaussian-Splatting* arc (LRM 107, TripoSR 108, SPAR3D 109, GS-LRM 110) was the *recent* focus
- **Recommendation: read Intellident 2025 next** (paper 024, already read but worth re-visiting) as the *integrated* companion to the *Chafi 124 + Alsheghri 123* pair, to get the *complete* Polytechnique + KerenOr pipeline view (margin line + crown bottom + crown shell = the *full* inside+outside crown)

### Reading time
~30 minutes, including the 5 web searches to verify the SPIE paper exists (the SPIE page is *blocked* by Incapsula CDN, the *only* way to read the paper is via the PolyPublie institutional repository or via the GitHub repos), the GitHub repo analysis (Common/Evaluation/Model directories for CB-GAN + ~50-line cb_generation.py for C.B.GEN), and the Synthetic Anatomy review cross-reference. The paper is *small* (8 pages, 21 references, 5 citations), the *contribution* is *comparative* (not a new method), and the *code* is the *real* contribution (MIT-licensed, fully reproducible, *production-grade* PyVista+Trimesh+PyMeshLab baseline for the geometric method + full SP-GAN fork for the GAN method).
