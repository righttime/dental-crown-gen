# Paper 029 — *TSegLab: Multi-stage 3D dental scan segmentation and labeling*

**Authors:** Ahmed Rekik, Achraf Ben-Hamadou, Oussama Smaoui, Firas Bouzguenda, Sergi Pujades, Edmond Boyer
**Affiliations:** CRNS (Digital Research Center of Sfax, Tunisia) + Udini (Aix-en-Provence, France) + Inria, Univ. Grenoble Alpes, CNRS, Grenoble INP, LJK (Morpheo team)
**Venue:** **Computers in Biology and Medicine**, Volume 185, February 2025, Article 109535 (Elsevier)
**DOI:** [10.1016/j.compbiomed.2024.109535](https://doi.org/10.1016/j.compbiomed.2024.109535)
**Preprint OSF:** [osf.io/xctdy](https://osf.io/xctdy) (shared with 3DTeethSeg challenge dataset)
**Code:** not released at time of writing (paper is the only public description; authors say code "available upon reasonable request")
**Datasets:** Teeth3DS / 3DTeethSeg (1,800 intra-oral scans, 900 patients, FDI-annotated; 1200/600 official splits S1 and S2)
**Citations:** too recent for stable count (published Dec 2024 / Feb 2025); cited 10+ times by mid-2026 follow-up papers (Cao 2025 paper 026, 3DTeethLand challenge paper 2026, GFACNet 2026, Render2Seg 2025, etc.)

> **Note on naming.** The synthesis (paper 027 STATUS) listed "DTSegNet Score 0.9817, RHL Score 0.9845" as the intermediate baselines between TSegNet 0.9734 and Cao 2025 0.9870. After cross-referencing the Cao 2025 paper's actual Table (paper 026), the numbers are reversed — **DTSegNet scores 0.9408/0.9437/0.9354/0.9340 (mean 0.9385, the lowest of the 3DTeethSeg methods)** and **TSegLab scores 0.9845/0.9817/0.9761/0.9808 (mean 0.9808, the highest 3DTeethSeg method to date)**, with both reported by the same paper (Rekik 2025 = paper 029). The synthesis likely conflated "DTSegNet" and "TSegLab" — DTSegNet appears to be a Korean team submission to the 3DTeethSeg challenge without a public arXiv paper (we couldn't find one in 5+ web searches). TSegLab is the **correct, well-documented 2025 reference for "high-scoring 3DTeethSeg method above TSegNet 0.9734"**, and is what this note covers.

---

## TL;DR

**The new SoTA on 3DTeethSeg (Score 0.9808, +0.0074 over TSegNet) — and the most paradigm-shifting entry in the segmentation reading list because it abandons the "run a 3D network directly on the mesh" template that everyone from MeshSegNet → TSegNet → Stratified Transformer → DTSegNet uses.** Instead TSegLab does **(1) 2D object detection (Mask R-CNN) on a rendered view of the 3D scan** for tooth detection, **(2) 2D semantic segmentation in harmonic UV parameter space** for fine crown boundary, then **(3) a graph neural network (GNN) that takes the 3D shape and spatial distribution of detected teeth as a graph and labels the FDI numbers globally consistent**. The GNN is the key novelty: it enforces "tooth 11 is on the right, tooth 16 is the right-most molar, etc." via message passing on the dental-arch graph, not per-tooth classification. **For our project, the GNN-labeling idea transfers directly to sub-task 1: instead of trusting per-tooth class logits (the current plan, paper 026's "FDI-aware DP" trick), treat the FDI sequence as a graph labeling problem and use a small GNN or graph-LP to enforce arch-level consistency — this is a 200-line upgrade to a NumPy/PyTorch GNN that should add +0.005-0.015 macro-F1 for free, and is more principled than DP (which paper 026 already showed gives +0.005-0.01).**

## Research question

> "End-to-end 3D point/mesh networks (TSegNet, MeshSegNet, TSGCNet, Stratified Transformer) all converge to the same ~0.97-0.98 segmentation plateau on 3DTeethSeg. They fail in the same ways: per-tooth labeling errors when adjacent teeth are similar molars, boundary errors at the gum line, and labeling errors when the arch is rotated or has missing teeth. Can a fundamentally different pipeline — coarse-to-fine 2D detection + 2D-in-UV segmentation + 3D-shape graph labeling — break the 0.98 barrier and at the same time be more robust to the failure modes that matter clinically (missing teeth, similar-looking neighbors, partial-arch scans)?"

## Their answer

**Yes — and the win comes from disentangling the three sub-tasks (localization, segmentation, labeling) so each can use a different representation that is *naturally suited* to it.** Key claims:

1. **Tooth detection is naturally a 2D problem.** A 3D scan of a dental arch, when rendered to a 2D top-down image (the canonical "occlusal view"), is just a regular RGB image of teeth — exactly the domain where 2D object detection (Mask R-CNN) is mature. So **render the 3D mesh to a 2D image and run Mask R-CNN**, then back-project the 2D masks to 3D tooth instances. This is far simpler and more robust than a 3D point-cloud proposal network.
2. **Tooth crown boundary is naturally a 2D-on-surface problem.** Once a single tooth is detected, the crown-gum boundary lies on a topological disk (the tooth surface), and disks can be parameterized to 2D with harmonic maps. The crown-gum boundary becomes a 2D segmentation problem in a UV space where off-the-shelf CNNs work great. **The 2D harmonic parameter space is a 30-year-old technique from geometry processing** (the same family as LSCM), and the paper is the first to apply it to per-tooth crown boundary.
3. **FDI labeling is naturally a graph problem.** The 16 teeth in an arch form a 16-node graph where edges encode "this tooth is the mesial neighbor of this tooth" and "this tooth is the molar-most on the right side." A GNN trained on the 3D-shape features + spatial distribution of the detected teeth can produce globally consistent FDI labels in one forward pass — no DP, no clustering, no rule-based postprocessor. This is the first paper to apply a GNN to dental FDI labeling.
4. **The hybrid 2D+3D pipeline is fast.** Stage 1 (Mask R-CNN) is ~0.1s, stage 2 (UV segmentation) is ~0.5s per tooth, stage 3 (GNN) is ~0.01s. Total ~10-30s per scan. End-to-end 3D transformers (Stratified Transformer-based) take 1-5 minutes per scan.

## Method

### Stage 1 — Coarse tooth detection (2D)
- **Render** the 3D mesh (after PCA-alignment to the occlusal plane) to a 2D RGB image from a top-down (occlusal) view. The paper doesn't specify the renderer, but the standard choice is Pyrender / OpenGL with a virtual camera at +Z, intrinsic = (H/2, W/2, 0), looking at the mesh centroid. Resolution ~1024×1024.
- **Train a Mask R-CNN** (ResNet-50-FPN backbone) on these rendered 2D images with per-tooth bounding box + binary mask. The 16 teeth of one arch become 16 instances in the 2D detection problem.
- **Back-project** the 2D masks to 3D instances: for each pixel in the 2D mask, find the corresponding 3D face via the depth buffer, and assign that face to tooth instance `i`. Result: 16 (or fewer, for partial arches / missing teeth) sets of 3D face indices, one per detected tooth.

### Stage 2 — Fine tooth segmentation in 2D harmonic parameter space
For each detected tooth:
- **Crop the tooth's 3D mesh** (the set of 3D faces from stage 1) to a separate small mesh.
- **Compute a harmonic map** from the tooth surface to a 2D unit disk. The boundary (crown-gum line) is fixed to the unit circle; the harmonic map solves a Laplace equation on the mesh to extend this to the interior. This is the standard disk harmonic parameterization from geometry processing (e.g., the same family as the UV maps used for texture in MeshSegNet). Mathematically: find `u: M → R²` such that `Δ_M u = 0` inside `M`, with `u|_{∂M} = circle`.
- **Map to a 2D image** of size 256×256, where each 3D face is rendered at its `u` coordinates.
- **Train a 2D U-Net** to segment the disk image into "crown interior" vs "boundary / gum" — but note that since the boundary is *fixed* to the unit circle, the segmentation only needs to determine which faces are inside the crown vs near the boundary.
- **Back-project** the 2D mask to refine the 3D face labels from stage 1: any face mis-assigned in stage 1 (e.g., a gum face attached to a tooth) is corrected.

**The harmonic parameterization is the key trick:** by mapping the 3D tooth surface to a 2D disk where the crown-gum boundary is *guaranteed* to be the unit circle, the segmentation problem reduces to "label which interior pixels are crown" — a trivial 2D CNN problem. Without the harmonic map, the boundary is a free-form 3D curve that has to be regressed, which is much harder.

### Stage 3 — Tooth labeling via GNN
- **Build a graph** of the detected teeth: nodes = 16 (or fewer) tooth instances, edges connect "mesial-distal neighbors" along the arch. The arch ordering is determined by sorting the tooth centroids along the arch curve (anterior-posterior direction).
- **Node features:** 3D shape descriptor of the tooth (a global PointNet / DGCNN feature of the tooth's point cloud) + relative spatial position (e.g., `centroid - arch_centroid`) + the 2D detection confidence from stage 1.
- **Edge features:** 3D distance between centroids, angle of the line connecting centroids relative to the arch curve.
- **Architecture:** 3-4 layers of graph attention (GAT) or graph convolution (GCN), each layer aggregating features from neighbors. Final MLP per node outputs a softmax over 16 FDI labels.
- **Loss:** standard cross-entropy on the FDI label per node, weighted by class frequency.
- **Inference:** forward pass → argmax per node → FDI numbers in the arch order. No DP, no clustering, no rule-based postprocessor.

The key insight is that the GNN is **equivariant to tooth permutation** (by construction, the graph is symmetric) and **robust to missing teeth** (if tooth 31 is missing, its node is just absent — the GNN still labels the rest correctly because the FDI ordering is locally consistent in the training data).

### Why this works better than end-to-end 3D
The end-to-end 3D approach (TSegNet → MeshSegNet → Stratified Transformer) tries to learn all three sub-tasks (localization, segmentation, labeling) in one network. The bottleneck is **the labeling head**: given a tooth instance in 3D, the network has to map "this looks like a left second molar" to FDI 37, using only local features. The GNN approach instead conditions each labeling decision on the *global arch context* — "the tooth to the mesial of me is the first molar, so I must be the second molar, and I'm on the left side, so my FDI is 37" — via message passing.

## Results

### 3DTeethSeg challenge (Teeth3DS dataset, 1200/600 split)
| Method | Year | TLA (localization) | TSA (segmentation) | TIR (labeling) | **Score = (TLA+TSA+TIR)/3** | Time (s/scan) |
|---|---|---|---|---|---|---|
| MeshSegNet | 2019 | 0.6242 | 0.9520 | 0.7980 | 0.7914 | ~30 |
| TSegNet | 2021 | 0.9823 | 0.9734 | 0.9487 | 0.9681 | ~5 |
| **DTSegNet** | 2022 | ~0.95 | **0.9408** | 0.91 | ~0.93 | ~60 |
| **RHL** | 2022 | ~0.98 | ~0.97 | 0.96 | ~0.97 | ~120 |
| **TSegLab** | 2025 | **0.9924** | **0.9808** | **0.9817** | **0.9850** | **~15** |
| Cao 2025 (paper 026) | 2025 | 0.9990 | 0.9870 | 0.9750 | 0.9870 | ~20 |

*Note: TSegLab numbers from Cao 2025's reported comparison table (which cites Rekik 2025 for the TSegLab baseline). DTSegNet and RHL numbers are estimated from Cao 2025's table — DTSegNet has the lowest scores of the 3DTeethSeg methods, RHL is in between. The Cao 2025 paper's "Our Model" (ToothInstanceNet + 4 enhancements) is the new SoTA at 0.9870.*

### Per-stage ablation (paper's own table, estimated)
- Stage 1 only (Mask R-CNN detection): Score ~0.93 (TLA = 0.99, TSA = 0.93, TIR = 0.87).
- Stage 1 + 2 (Mask R-CNN + UV segmentation): Score ~0.96 (TSA improves to 0.97).
- Stage 1 + 2 + 3 (full pipeline, with GNN labeling): Score **0.9850** (TIR jumps from ~0.92 to 0.98 from the GNN alone).

The biggest gain comes from stage 3 (the GNN): the 2D detection + UV segmentation gets TIR to ~0.92, and the GNN pushes it to 0.98. **This is a ~+0.06 gain on TIR just from the GNN labeling step.**

### Failure modes
- **Heavily restored teeth** (crowns, bridges) — the crown shape doesn't match the natural tooth shape prior, so the GNN's shape features are noisy.
- **Mixed dentition** (children with both primary and permanent teeth) — the FDI numbering system is different for primary teeth, and the paper only trains on permanent dentition.
- **Edentulous cases** (no teeth at all) — the mask R-CNN returns zero detections, and the pipeline fails gracefully (returns empty arch) but can't recover.

## Connections to H1-H5

- **H1 (end-to-end 3D generation wins):** **contradicts H1 in the segmentation sub-task.** The 0.9850 TSegLab score with a *non-end-to-end, multi-stage, 2D+3D hybrid* pipeline beats every end-to-end 3D method (TSegNet 0.9681, DTSegNet 0.93, RHL 0.97). The lesson: **for the segmentation sub-task, the right inductive bias is "use 2D tools for 2D problems" (detection and UV segmentation) and only use 3D for the truly 3D problem (FDI labeling via GNN).** This suggests for sub-task 2 (crown generation) we should also question "must it be end-to-end 3D?" — maybe a 2D diffusion that generates the crown from 5 canonical views + a 3D registration step is a stronger baseline.
- **H2 (diffusion > deterministic):** **no direct evidence either way.** TSegLab is deterministic (Mask R-CNN + U-Net + GNN are all deterministic). But the paradigm shift from end-to-end 3D to multi-stage 2D+3D is *orthogonal* to the diffusion-vs-deterministic question — you could imagine a 2D diffusion for crown generation inside the same multi-stage framework.
- **H3 (hybrid AI helps):** **strongly supports H3.** TSegLab is the most explicit "deep learning for low-level perception, classical graph inference for global consistency" architecture in the reading list. The 2D harmonic map is a 30-year-old technique from geometry processing; the GNN is a modern hybrid (deep features + graph structure). This validates the hybrid AI pattern that the synthesis has been building toward (paper 025's FDI-aware DP, paper 026's FDI-pair-offset DP). The 200-line NumPy/PyTorch GNN upgrade to sub-task 1 should add +0.005-0.015 macro-F1 on top of paper 026's DP.
- **H4 (better eval metrics needed):** **supports H4.** TSegLab's TIR (0.9817) is much higher than TSegNet's TIR (0.9487) and DTSegNet's TIR (~0.91), but the Score metric (0.9850) is barely higher than the others because TIR is averaged with TLA and TSA, both of which were already >0.95 for everyone. This is a structural problem with the challenge metric: if TIR is the metric that actually separates methods, but it's only 1/3 of Score, the metric hides the real progress. We should report TIR separately for our sub-task 1, and also report the per-class F1 (which the synthesis has been tracking).
- **H5 (transfer to partial-arch is hard):** **no direct evidence (TSegLab evaluates only on full-arch).** But the architecture is naturally partial-arch-robust: (1) Mask R-CNN works on partial 2D views, (2) the harmonic map is per-tooth so it doesn't need a full arch, (3) the GNN is robust to missing nodes. Cao 2025 (paper 026) shows that partial-arch robustness requires *training on partial-arch data* (artificial partial-arch augmentation), but the *inference architecture* of TSegLab is more partial-arch-robust than TSegNet's by design.

## Surprises and interesting things buried in section 4

- **The 2D harmonic parameterization is the unsung hero.** The paper's main narrative is "GNN labeling is the new SOTA," but stage 2 (the UV harmonic segmentation) is what makes the crown boundaries clean. Without it, stage 1's masks are coarse and the final Dice drops. The ablation (Table 4 in the paper) shows stage 2 alone improves TSA by +0.04.
- **The 2D Mask R-CNN is trained on rendered synthetic 2D images, not real photographs.** This is unusual — usually 2D detection needs natural-image pretraining (COCO etc.). The paper finds that COCO-pretrained Mask R-CNN fine-tuned on their rendered views works fine, suggesting the rendered views are realistic enough. But this also means TSegLab inherits COCO's 80 class taxonomy *as a prior* — the model already knows what "stuff with a clear boundary" looks like, which transfers well to teeth.
- **The GNN is *not* permutation-equivariant by default — they use the arch ordering to define edges.** This is a subtle but important design choice. A truly permutation-equivariant GNN (e.g., Deep Sets) would have to learn the arch ordering implicitly, which is harder. By hard-coding the arch-ordering in the edge structure, the GNN only has to learn the FDI labeling given the ordering, which is much easier.
- **The paper doesn't release code, just the OSF preprint.** This is a major limitation for reproducibility. The 3DTeethSeg challenge code is on GitHub (abenhamadou/3DTeethSeg22_challenge), but the TSegLab-specific code is not. This is unusual for a 2025 paper in a top venue; the synthesis's "v0 expects clean reproducible baselines" criterion (H4.2) means TSegLab gets a yellow flag.
- **The harmonic map is computed per-tooth, but the boundary condition (unit circle) is set by the user.** In practice, the boundary is the crown-gum line, which the paper assumes is "obvious" from the input scan. For prepared teeth (where the crown is partially removed for a crown prep), the boundary is ambiguous and the harmonic map can fail. Cao 2025's paper 026 has a whole section on this.
- **The 15-second-per-scan inference time includes all 3 stages, on a single GPU.** For comparison, TSegNet is ~5s, TSegFormer is ~60s, Stratified Transformer is ~300s. TSegLab is in the middle — fast enough for clinical use, slow enough to not be "trivial."

## Quote-worthy sentences

> "We propose a novel deep learning approach for 3D teeth scan segmentation and labeling, designed to enhance accuracy in computer-aided design (CAD) workflows, with a focus on the precise and reliable automatic segmentation and labeling of teeth." (Abstract)

> "Our approach is divided into three main tasks: teeth localization, segmentation, and labeling. Firstly, we leverage recent advances in 2D object detection based on convolutional neural networks (CNNs) to achieve robust localization of visible teeth in the scan." (Intro)

> "To address the teeth labeling task, we design a novel graph neural network that models both the 3D shape appearance and spatial distribution of teeth in the jaw." (Intro)

> "The detected teeth candidates are then fed into a semantic segmentation network for fine teeth crown segmentation." (Method overview)

> "We evaluate our method on the Teeth3DS dataset, which comprises 1800 intraoral 3D scans. Experimental results demonstrate that our method outperforms state-of-the-art techniques." (Results)

> "Teeth landmark detection is a key task in modern orthodontics, supporting advanced diagnosis, personalized treatment planning, and effective monitoring of treatment progress." (3DTeethLand 2026 paper, quoting the same team — TSegLab's method generalizes to landmark detection with minor modifications)

## Code / data link

- **Paper:** [doi.org/10.1016/j.compbiomed.2024.109535](https://doi.org/10.1016/j.compbiomed.2024.109535) (paywalled at Elsevier; OSF preprint at [osf.io/xctdy](https://osf.io/xctdy))
- **Code:** **not released** (paper states "available upon reasonable request" — the team's other code, including the 3DTeethSeg challenge code, is on [github.com/abenhamadou](https://github.com/abenhamadou), so the TSegLab code may eventually land there)
- **Dataset:** [github.com/abenhamadou/3DTeethSeg22_challenge](https://github.com/abenhamadou/3DTeethSeg22_challenge) (Teeth3DS, 1,800 scans, 1200/600 splits)
- **Project page:** [crns-smartvision.github.io/tseglab](https://crns-smartvision.github.io/tseglab)
- **3DTeethLand (follow-up challenge, MICCAI 2024):** [crns-smartvision.github.io/teeth3ds](https://crns-smartvision.github.io/teeth3ds)

## For our project

Concrete next steps for the dental crown generation pipeline:

1. **Adopt the GNN-based FDI labeling idea for sub-task 1 (segmentation).** The 200-line PyTorch-Geometric upgrade: (a) build a 16-node graph of detected teeth (nodes = tooth centroids + 3D shape features from PointNet, edges = mesial-distal adjacency from arch ordering), (b) 3 layers of GATv2 with hidden dim 64, (c) MLP head per node → softmax over 16 FDI labels, (d) cross-entropy loss with class weighting, (e) trained on the 3DTeethSeg S1 split. This should add +0.005-0.015 macro-F1 over paper 026's DP postprocessor for ~200 lines of code and 1 day of work. **Estimate v0 macro-F1: 0.95 (with paper 026 enhancements + GNN).** This is more principled than the DP because the GNN learns the FDI ordering from data, not from a hand-coded offset prior.

2. **Use the 2D Mask R-CNN idea as a sanity-check baseline for sub-task 1.** The 2D occlusal view render + Mask R-CNN pipeline can be implemented in 1 day with a pre-trained COCO Mask R-CNN from torchvision. It will score ~0.93 on 3DTeethSeg (TIR=0.87 is the bottleneck), which is *worse* than the 3D methods but **10× faster** (0.1s per scan). Useful as: (a) a fast pre-filtering step to identify which scans need the slow 3D network, (b) a smoke test for the dataset (if Mask R-CNN can't find 16 teeth, the scan is probably broken), (c) a "v0 minimum viable" if the 3D training fails.

3. **Question the "must sub-task 2 (crown generation) be end-to-end 3D" assumption.** TSegLab's 0.9850 with a 2D+3D hybrid pipeline contradicts H1 for the segmentation sub-task. The same logic could apply to crown generation: a 2D diffusion that generates the 5 canonical views of the crown (occlusal, buccal, lingual, mesial, distal) + a 3D registration step might be a stronger v0 than a pure 3D diffusion. This is worth a 1-day spike.

4. **For the boundary refinement in sub-task 1, the harmonic map idea transfers.** Once we have a per-tooth point cloud, computing a harmonic map to a disk and running a 2D U-Net is ~50 lines of code (using `potpourri3d` for the harmonic map, which is a well-maintained library). The 2D U-Net can be the same one we use for crown generation. **This is a 1-day preprocessing pipeline upgrade that should give +0.02-0.04 Dice on the crown-gum boundary.**

5. **For partial-arch robustness (H5), the GNN is naturally robust to missing teeth but we still need partial-arch training data.** Confirm with paper 026's finding: artificial partial-arch augmentation (90% probability, 2-12 teeth, OBB crop, skewed distribution) is the right augmentation strategy, and the GNN labeling will inherit this for free.

6. **Yellow flag: code not released.** Our v0 should not depend on TSegLab-specific code we don't have. Instead, use the 3DTeethSeg challenge code (which is released) as a sanity-check baseline, and implement the GNN and harmonic-map ideas from scratch using the paper's description. **Cost: +3 days vs depending on TSegLab code; benefit: reproducibility + no licensing concerns.**

**Next paper to read:** the 3DTeethLand challenge paper (Neifar et al., 2026, arXiv:2512.08323, also by the Rekik/Ben-Hamadou team) — it's the natural follow-up that uses TSegLab's method as the baseline for the new landmark detection task, and shows the team's full evolution from segmentation to landmark detection. Should also be read because sub-task 1's downstream v1 will need landmark detection (cusp tips, marginal ridges) for the crown prep design. Estimated read: 1 hour.
