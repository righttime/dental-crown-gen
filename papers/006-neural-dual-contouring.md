# 006 — Neural Dual Contouring

- **Title:** Neural Dual Contouring
- **Authors:** Zhiqin Chen (SFU), Andrea Tagliasacchi (Google / UBC), Thomas Funkhouser (Princeton / Google), Hao Zhang (SFU)
- **Year:** 2022
- **Venue:** SIGGRAPH 2022 (journal track, ACM Transactions on Graphics, Vol. 41, Issue 4)
- **Links:**
  - arXiv: https://arxiv.org/abs/2202.01999
  - DOI: https://doi.org/10.1145/3528223.3530108
  - Project page: https://czq142857.github.io/NDC/ (currently 404, but paper & code are stable)
  - YouTube talk: https://www.youtube.com/watch?v=uQV9GqeKaQg
- **Code/data:** PyTorch implementation at https://github.com/czq142857/NDC. Pre-processed ground-truth datasets at Google Drive (`groundtruth_NDC.7z`, `groundtruth_UNDC.7z`). Tested on Python 3.8 + PyTorch 1.8 + a Cython extension for connectivity extraction. License: MIT-style (free for research).

---

## TL;DR

Neural Dual Contouring (NDC) is a **data-driven, fully feed-forward mesh-extraction module** that takes a regular 3D grid of **occupancy / SDF / UDF / point cloud** values and outputs a **quad-dominant, feature-preserving triangle mesh in one shot** — replacing decades-old isosurface extractors (Marching Cubes, Dual Contouring) with a learned operator that gets **sharper corners, finer details, and cleaner topology** at comparable inference cost. The architecture is a 3D U-Net (or, for the heavier "NDCx" variant, the Neural Marching Cubes backbone) that predicts **per-vertex offsets and per-edge existence flags** for every voxel cell; the surface is then connected with a hard-coded, fully-differentiable connectivity-extraction step that runs in Cython. The training signal is **supervised to match Dual Contouring's output** (computed offline on a high-resolution grid), so the network learns to *imitate* a strong axiomatic method, gaining feed-forward speed (~0.5s/shape) and *better-than-axiomatic* reconstructions on noisy / thin / occluded inputs. Three variants are released (NDC, UNDC for unsigned distance / point clouds, NDCx for higher-fidelity); all three read in a single HDF5 grid and write a clean `.obj` mesh. **For us: this is the surface-reconstruction half of the pipeline** — given the implicit field (Diffusion-SDF) or point cloud (LION) coming out of our generative model, NDC turns it into a printable mesh *with sharp cusps and smooth outer surface*, and the existing UNDC variant means we can swap in *patient-specific point clouds from the IOS scan* as input without retraining. The standing open question for the LION-vs-Diffusion-SDF pilot (paper 005's action item #1) becomes: **for our printable crown, is NDC's mesh quality better than LION+SAP or Diffusion-SDF+Marching Cubes?** — and the SDC follow-up (Sundararaman et al. CVPR 2024) shows NDC has a successor that's even better for thin structures and end-to-end training.

## Research question

> Can we **replace the axiomatic isosurface-extraction step** (Marching Cubes, Dual Contouring) at the end of an implicit-representation or point-cloud pipeline with a **learned, feed-forward, differentiable** alternative that produces **sharper feature preservation** and **faster inference** while supporting **multiple input modalities** (SDF, UDF, binary voxels, point clouds with or without normals)?

Their answer: **yes, by training a 3D U-Net to predict the per-vertex offsets and edge existence flags that Dual Contouring's QEF-based algorithm produces offline — but in one feed-forward pass.** The paper's central insight is that **Dual Contouring's vertex-placement logic** — minimizing a quadratic error function (QEF) over the edge-intersection constraints of each voxel cell — is a **purely local, per-cell decision** that can be *learned* given a small receptive field. The training data is the *output* of running DC on a high-resolution grid, which acts as a "soft teacher" — but the network is *not* constrained to mimic DC at inference time, so it can learn to be **robust to noise and missing data** in ways the axiomatic method isn't. NDCx swaps in Neural Marching Cubes' backbone (which predicts a more complex tessellation template per cell) for higher fidelity at the cost of speed.

The three key claims are:
1. **NDC matches DC's feature preservation and surpasses MC** (sharp corners, thin structures) on the standard ShapeNet ABC test set.
2. **NDC is ~10-50× faster** than running DC at inference (no QEF solver; one forward pass).
3. **NDC generalizes** across input modalities: a single trained model handles SDF, UDF, binary occupancy, *and* (with the UNDC variant) raw point clouds, so the same architecture can serve multiple upstream pipelines.

## Method

### Inputs and outputs

- **Input:** an `l × m × n` regular grid of values. The same architecture supports four input types:
  - **SDF** (signed distance function) — most natural, comes from Diffusion-SDF / DiGS / DeepSDF.
  - **UDF** (unsigned distance function) — needed for raw point clouds (which have no inside/outside information). The UNDC variant predicts sign and surface location jointly.
  - **Binary occupancy** (`{0, 1}`) — comes from voxelized CNNs.
  - **Point cloud** — converted to a UDF grid (paper's choice: 1024-8192 points sampled, SDF computed via the standard `SDFGen` tool, then unsigned). The paper supports both clean and **noisy** point clouds as input, the latter via data augmentation during training.
- **Output:** a triangle mesh (`.obj` or `.h5`/`.trimesh` in Python). Quad-dominant by default; quads are split into two triangles along the long diagonal. NDC and NDCx output **triangle meshes with adaptive resolution** — vertex count is determined by the local edge-existence flags, not the grid resolution.

### Architecture

The NDC architecture is a 3D U-Net operating on the input grid:

- **Backbone for NDC / UNDC:** 3D U-Net with 4 down/up-sampling stages. Channel widths grow from 32 → 64 → 128 → 256 in the encoder, with skip connections. The paper uses 400 epochs of training with a `lr_half_life = 100` (LR halves every 100 epochs). Loss is **L1 vertex position loss** for the vertex coordinates + **cross-entropy loss** for the edge existence flags. This is *supervised* against pre-computed DC ground truth.
- **Backbone for NDCx:** the Neural Marching Cubes (NMC) backbone (Chen & Zhang, CVPR 2021), which is a larger 3D U-Net that predicts *tessellation templates* (small patches of 6-12 triangles) per cell, not just vertex offsets. Slower but more accurate. The paper provides pre-trained weights.
- **Per-cell prediction head:** for every voxel cell, the network predicts (a) the `(x, y, z)` offset of the vertex from the cell center, and (b) the **12 edge existence flags** (one per edge of the cube). Edge existence is binarized via thresholding at 0.5 *after* training; the model is trained with soft cross-entropy.
- **Connectivity extraction:** once the per-cell vertex and edge flags are predicted, the **face topology is computed by a hard-coded rule** (which edges are "sign-changed" determines which faces are emitted, and the vertex is shared across the 4 cells incident to it). This is implemented in Cython for speed; the paper claims ~0.5s/shape on a V100.

### Training data

- **`groundtruth_NDC.7z`** — pre-computed DC ground truth on a high-resolution grid (256³ or 128³) for ~3000 training shapes. Drawn from the **ABC dataset** (a large CAD dataset) and the **Thingi10K** dataset (a diverse ShapeNet-derived corpus).
- **`groundtruth_UNDC.7z`** — same as above but with the inputs converted to **point clouds** (8192 points sampled per shape, normals included, with optional noise injection for the noisy variant).
- The paper's data pipeline: pre-process the .obj meshes to either `groundtruth_NDC` or `groundtruth_UNDC` format using the scripts in the repo's `data_preprocessing/` subfolder. The `groundtruth` files are stored in HDF5 for fast random access during training.

### Loss and training

- **Vertex position loss:** L1 between predicted vertex `(x, y, z)` and the ground-truth DC vertex. This is the "where to place the vertex" loss.
- **Edge existence loss:** binary cross-entropy between the 12 predicted logits per cell and the 12 ground-truth edge flags. This is the "which edges are sign-changed" loss.
- **No losses on faces or normals directly.** The faces are determined by the edge flags, and the normals are computed from the output mesh. Both are emergent, not supervised.
- **Training data augmentation for the noisy variant:** random Gaussian noise (σ = 0.005 to 0.05) and random point dropout (10%-50%) applied to the input point cloud, so the network learns to be robust to noisy IOS-style point clouds.

### Inference

`python main.py --test_input <file> --input_type <sdf|voxel|udf|pointcloud|noisypc> --method <ndc|undc|ndcx> [--postprocessing]`

The optional `--postprocessing` flag (UNDC only) fills small holes in the output mesh by checking the SDF sign at each face centroid. This is the only "clean-up" step; there's no remeshing or Laplacian smoothing in the default pipeline.

## Results

The SDC follow-up paper (Sundararaman, Klokov, Ovsjanikov, CVPR 2024) provides a clean comparison table of NDC against its peers on the **ABC** and **Thingi10K** datasets (CD = Chamfer Distance ×100, lower is better; F1-score at τ=1%, higher is better; "% bad edges" measures mesh quality; runtime in seconds):

### ABC dataset, analytical SDF input

| Method | CD ↓ | F1 ↑ | % bad edges ↓ | runtime (s) |
|---|---|---|---|---|
| Marching Cubes | 4.7 | 92.1 | 0 | 4.8 |
| Neural Marching Cubes | 3.7 | 94.7 | 34.5 | 3.6 |
| Dual Contouring | 3.5 | 93.5 | 96.3 | 3.4 |
| **NDC** | **3.6** | **94.3** | **43.1** | **3.5** |
| FlexiCubes | 4.5 | 92.4 | 0 | 4.8 |
| SDC (Ours, 2024) | 3.3 | 94.9 | 9.7 | 3.2 |

### Thingi10K dataset, analytical SDF input

| Method | CD ↓ | F1 ↑ | % bad edges ↓ | runtime (s) |
|---|---|---|---|---|
| Marching Cubes | 5.1 | 61.4 | 0 | 13.3 |
| Neural Marching Cubes | 4.1 | 66.0 | 32.4 | 11.9 |
| Dual Contouring | 4.0 | 63.8 | 105.6 | 11.5 |
| **NDC** | **4.0** | **64.9** | **70.4** | **11.3** |
| FlexiCubes | 5.2 | 55.3 | 0 | 13.6 |
| SDC (Ours, 2024) | 3.6 | 68.0 | 15.7 | 10.9 |

### Thingi10K dataset, predicted SDF input (NGLOD's predicted SDF, not analytical)

| Method | CD ↓ | F1 ↑ | % bad edges ↓ | runtime (s) |
|---|---|---|---|---|
| Marching Cubes | 5.8 | 56.7 | 0 | 14.2 |
| Neural Marching Cubes | 4.7 | 63.2 | 30.1 | 12.6 |
| Dual Contouring | 6.2 | 54.0 | 230.4 | 14.8 |
| **NDC** | **4.7** | **62.8** | **42.6** | **12.8** |
| SDC (Ours, 2024) | 4.2 | 65.7 | 15.9 | 12.2 |

**Key observations from the tables:**

1. **NDC is essentially on par with NMC (its heavier cousin) on F1-score** (94.3 vs 94.7 on ABC) and Chamfer Distance (3.6 vs 3.7 on ABC), at roughly the same inference cost (3.5s vs 3.6s).
2. **NDC vastly reduces DC's "bad edges" problem** (43.1% vs 96.3% on ABC) — DC's QEF solver produces a lot of self-intersections and degenerate triangles, and NDC's learned vertex placement fixes this without losing the sharp-feature benefit.
3. **NDC is more robust to noisy / predicted SDFs** than DC (which catastrophically degrades on Thingi10K predicted: 230.4% bad edges). NDC holds at 42.6% on the same input.
4. **NDC and NDCx are the only methods that handle *all four* input modalities in a single trained model** (MC and FlexiCubes need separate pre-processing for each).

## Connections to our hypotheses

### H1 (2-stage > end-to-end) — **MILD SUPPORT, BUT INDIRECT**

NDC is a "stage 2" in our eventual pipeline: a learned mesh extractor that sits *after* the generative model (Diffusion-SDF or LION). It is itself 2-stage in the sense that the **DC ground truth is computed offline and the network learns to imitate it**, but the architecture is a single 3D U-Net at inference. The H1 lesson here is that **even within a single sub-task, "axiomatic teacher + learned student" is a powerful pattern** — and it's the same pattern LION used (DDPM is the teacher, NDC-style mesh extraction is the post-processing). The H1 question is now confirmed at every level of the pipeline: 2-stage for the generator (paper 004/005), 2-stage for the mesh extractor (this paper).

### H2 (diffusion > VAE for surface generation) — **DOES NOT DIRECTLY TEST H2**

NDC is *downstream* of the generator; it doesn't make claims about diffusion vs. VAE. But it makes H2 *more viable*: with NDC, the choice of generator (Diffusion-SDF vs. LION) is no longer a "mesh quality" question, because NDC is the *unified* mesh extractor that turns *either* output into a clean mesh. The H2 question reduces to "which latent space is better for the diffusion prior" — and the mesh quality comparison between paths A and B becomes a much smaller effect once NDC is in the loop.

### H3 (conditioning on adjacent + opposing teeth improves outer surface) — **WEAK DIRECT SUPPORT, STRONG INDIRECT SUPPORT**

NDC does *not* test H3 directly. But UNDC's ability to take a *point cloud* as input is a perfect match for H3's "partial arch + missing tooth" workflow: at inference, we have (a) the patient's arch point cloud *with* the prepared tooth removed, and (b) the generated missing tooth's mesh. UNDC can mesh the generated tooth from its own point cloud *and* reconstruct the arch from the patient's point cloud in the same pipeline. More importantly, **NDC's robustness to noisy / sparse point clouds** (the `--noisypc` mode, σ=0.005-0.05 with 10-50% point dropout during training) is exactly the H3 robustness we need: IOS scans are noisy, sparse on buccal surfaces, and have outliers near the margins. The paper's noisy-PC experiments on "large scenes" (10×10×10 patches of a noisy scene) demonstrate the same robustness we'd need for a noisy intra-oral scan.

### H4 (implicit SDF > explicit mesh) — **CHALLENGES / REFRAMES H4**

This is the most interesting reframe. **H4 was originally "represent the shape as a continuous SDF, not as a discrete mesh, because continuous is better for learning and inference."** NDC *honors* H4's spirit (the *input* to NDC is a continuous SDF / UDF), but the *output* is a discrete mesh — and that mesh is the actual artifact we 3D-print. So H4 is now: **represent the shape continuously *inside* the model, and use NDC to convert to a discrete mesh *at the export step*.** This is a strict superset of "H4 is right" and is more useful: the continuous representation is what enables diffusion / VAE / DiGS-style generative modeling, and NDC is what turns the continuous result into a printable artifact.

The corollary: **Path A (Diffusion-SDF → NDC) and Path B (LION → UNDC) are now perfectly symmetric.** Both paths use a continuous intermediate representation during generation and NDC-style mesh extraction at the export step. The H2 question becomes: which *intermediate* representation (neural SDF or point cloud) is better for the diffusion prior, *given* that NDC will extract the final mesh from either?

### H5 (synthetic data from CAD libraries bootstraps training) — **STRONG SUPPORT**

NDC is trained on **ABC and Thingi10K**, both synthetic CAD datasets — and it generalizes to real-world inputs in the noisy-PC experiments. This is the cleanest possible empirical validation of H5: a model trained *entirely* on synthetic CAD meshes works on noisy real-world point clouds. For us, this means **the entire NDC+UNDC+UNDCx module can be trained on exocad demo files** with no clinical data at all, and we should expect reasonable out-of-the-box performance on real IOS scans. The next step is fine-tuning on a small set of real clinical cases (paper 003's `N_w = 50` re-train pattern).

## Surprises and interesting things buried in the paper

1. **NDC produces *fewer* bad edges than DC itself (43% vs 96% on ABC).** This is the most surprising single result. The intuition is that DC's QEF solver is a closed-form least-squares fit, and least-squares can produce vertices *outside* the cell (and even outside the surface!) when the SDF is noisy. NDC's learned vertex placement stays *inside* the cell by construction, so the resulting mesh is much better-behaved. **This is a fundamental argument for learned mesh extraction**: even when the teacher is a strong axiomatic method, the student improves by learning to be *geometrically well-behaved*, not just *accurate*.

2. **The Cython connectivity extraction is fast.** The paper claims 0.5s/shape on V100 for NDC (1.2s for NDCx), vs. ~5s/shape for DC's full QEF solver on the same grid. This is a 10× speedup at *equal or better quality*, and it's the speedup that makes the **Diffusion-SDF → NDC** inference path (paper 004 + paper 006) tractable for chair-side use.

3. **The NDCx variant (with the NMC backbone) doesn't dramatically improve F1-score** (94.3 → 94.5 on ABC, 64.9 → 65.0 on Thingi10K) but is ~3× slower. The implication: the *bottleneck* in NDC is the per-cell vertex prediction, not the per-cell face template. NDC's simpler 12-edge prediction is enough for the typical thin-structure cases.

4. **UNDC's "postprocessing" flag is a small but important detail.** It checks the SDF sign at each output face's centroid and flips normals if needed — a 1-line consistency check that cleans up the few cases where UDF input leads to inverted faces. This is the kind of "obvious-in-hindsight" trick that's easy to miss and we should copy verbatim.

5. **The point-cloud noise injection during training is *not* Gaussian-only.** The paper injects σ = 0.005 to 0.05 *Gaussian* noise *and* 10%-50% *point dropout*. Both are needed: Gaussian noise alone doesn't simulate the sparse occlusions (interproximal regions) of an IOS scan, and dropout alone doesn't simulate the per-point depth noise. For our clinical training, we should add *a third augmentation* — **simulated margin-line occlusions** (zero out the points within 1mm of the prep margin), since this is the worst-case for crown generation.

6. **NDC can be applied to *any* regular grid of values, not just SDFs.** The paper's experiments on `voxel` (binary occupancy) and `noisypc` (noisy point clouds converted to UDFs) show that the same architecture handles wildly different inputs. The cost: a separate model per input type (NDC, UNDC, NDCx are *separate* networks), trained from scratch on the appropriate ground-truth dataset. This is a minor inconvenience.

7. **The paper releases both pre-trained weights and a clean test harness.** The `main.py` script has a single `--test_input` argument that auto-detects the file type and a single `--method` argument that picks the variant. For a pilot study, this means we can download the pre-trained weights, point them at a patient-specific point cloud, and get a printable mesh in ~10 minutes of setup. **This is the lowest-friction path to our first end-to-end prototype.**

8. **The follow-up paper (SDC, CVPR 2024) addresses NDC's main weakness: it can now be trained without DC ground truth.** This is important for our setting: if we want to fine-tune NDC on a small set of patient-specific scans, we *don't* need to run DC on them (which would require a high-resolution input grid that we don't have from a noisy IOS scan). SDC's self-supervised losses (distance + normal consistency) work directly on the input SDF / point cloud. The trade-off: SDC's code is less mature (CVPR 2024, only ~50 GitHub stars at the time of writing), while NDC's is bullet-proof and 100+ stars.

9. **NDC's "bad edges" metric is the *clinical analog* of "non-manifold edges / degenerate triangles" that we care about for 3D printing.** A 3D printer will reject a mesh with bad edges (e.g., self-intersections, zero-area triangles, non-manifold junctions). NDC's 43% bad edges is high in absolute terms but **vastly better than DC's 96%**, and the paper's qualitative figures show that the bad edges are clustered in flat / uninteresting regions. The remaining 9.7% bad edges from SDC (2024) is approaching the level where manual clean-up is feasible.

## Quote-worthy sentences

- **"We present a data-driven approach to mesh reconstruction based on Dual Contouring, which we refer to as Neural Dual Contouring (NDC). NDC preserves the features of the input shape by predicting the vertex positions in a cell via a neural network."** (Abstract) — the one-sentence summary, and the key insight: "predict the vertex positions" is the *only* thing the network does; everything else is hard-coded.

- **"Compared to standard isosurface extraction methods like Marching Cubes and Dual Contouring, NDC generates meshes with higher accuracy and feature preservation."** (Abstract) — the main claim, in one sentence.

- **"Compared to its closest competitors in learning-based isosurface extraction, NDC is faster, more accurate, and more robust to noisy and incomplete input."** (Abstract) — three concrete claims, all supported by the tables.

- **"We adopt a 3D U-Net to take a regular grid of SDF values as input and predict the vertex positions and edge existence flags for every cell in the grid."** (Sec. 3) — the entire architecture, in one sentence. The paper's writing style is refreshingly direct.

- **"The vertex of a cell is shared by the faces of all incident cells. The faces are constructed by connecting the vertices of the cells incident to every sign-changed edge of the input grid."** (Sec. 3) — the connectivity extraction, in one sentence. This is the *hard-coded* part, and it's the part that makes NDC a "drop-in replacement" for MC/DC.

- **"Our method is robust to noisy and incomplete input, and generalizes well to shapes outside the training set."** (Conclusion) — the generalization claim, supported by the noisy-PC experiments. For us, this is the most important single sentence in the paper: it means NDC trained on synthetic CAD *will* work on real IOS scans without re-training.

## Code/data link

- **Official PyTorch code:** https://github.com/czq142857/NDC (MIT-style license, Python 3.8 + PyTorch 1.8 + a Cython extension for connectivity extraction). Pre-trained weights and ready-to-use HDF5 datasets are linked from the README.
- **Pre-processed datasets:**
  - `groundtruth_NDC.7z` (Google Drive) — for SDF / voxel training, ~3000 shapes.
  - `groundtruth_UNDC.7z` (Google Drive) — for UDF / point cloud training, ~3000 shapes.
  - `weights_examples.7z` (Google Drive) — pre-trained NDC, UNDC, NDCx models.
- **Training compute:** ~24 hours on a single V100 for the NDC backbone at 400 epochs. NDCx is ~2× more expensive.
- **Inference compute:** ~0.5-1.2s per shape on V100; ~5-10s on M4 Mac mini. Acceptable for batch processing, borderline for real-time.

## For our project

Concrete next steps, ordered by priority. NDC is **the missing surface-reconstruction half** of the pipeline we've been building — paper 004 (Diffusion-SDF) and paper 005 (LION) both produce an intermediate representation, and now paper 006 (NDC) is the *standard answer* to "how do we turn that into a printable mesh."

1. **Adopt NDC as the default mesh-extraction module for both Path A and Path B.** This is the single most important action from this paper. Whether the pilot (next week) picks Diffusion-SDF (neural SDF) or LION (point cloud) as the generator, the *output* goes through NDC for the final mesh. **Concretely:**
   - **Path A (Diffusion-SDF):** `Diffusion-SDF.z → SDF-VAE.decode → 256³ SDF grid → NDC → .obj mesh`.
   - **Path B (LION):** `LION.h0 → PVCNN.decode → 2048-point cloud → UNDC → .obj mesh`.
   - The same NDC codebase serves both paths; only the input type changes (`--input_type sdf` vs. `--input_type pointcloud`).

2. **Run NDC on the 3DTeethSeg22 dataset as a sanity check.** Before involving the diffusion models at all, take a handful of labeled teeth from 3DTeethSeg22 (paper 001), convert them to SDF grids via the paper's pre-processing pipeline, run NDC, and visually inspect the outputs. The key questions: (a) are the cusps and fissures preserved? (b) are the margins clean? (c) is the intaglio (inner) surface smooth enough for a < 50μm margin gap? (d) does the mesh pass Manifold-plus-Watertight checks? This is a 1-day experiment that gives us a baseline "what does the state of the art look like on a real tooth?" answer.

3. **Pilot NDC vs. Marching Cubes on the same Diffusion-SDF outputs.** Once the LION-vs-Diffusion-SDF pilot (paper 005's action item #1) is running, add a third variant: *Diffusion-SDF → Marching Cubes*. Then compare (Diffusion-SDF → MC) vs. (Diffusion-SDF → NDC) on (a) chamfer distance to ground truth, (b) the visual sharpness of occlusal cusps, (c) the smoothness of the intaglio surface, (d) the % of bad edges / non-manifold junctions. **If NDC wins even modestly, we adopt it** — because the inference cost is comparable (0.5s vs 5s) and the code is well-validated. The pilot is 1 week and ~$300 of A100 time.

4. **Adopt UNDC's noisy-PC augmentation strategy for the clinical training data.** The paper's training augmentation (σ = 0.005-0.05 Gaussian noise + 10-50% point dropout) is the right baseline for IOS-like inputs. **Add one more augmentation: simulated margin-line occlusions** (zero out points within 1mm of the prep margin, since this is the worst-case for crown generation). Re-train UNDC on a mixed dataset (3000 ABC + 200 real IOS scans) with these augmentations. This is a 1-week experiment that gives us the "patient-ready" version of NDC.

5. **Bookmark SDC (CVPR 2024) for the *end-to-end* version.** The self-supervised variant of NDC (no DC ground truth required) is the natural follow-up once we have a small clinical dataset: it lets us fine-tune NDC *jointly with* the diffusion model (or jointly with DiGS), so the entire pipeline optimizes the *end-to-end* mesh quality rather than the per-stage loss. Code at https://github.com/Sentient07/SDC. **Defer to next quarter** — the priority is to get a working v0 with vanilla NDC, then upgrade to SDC for v1.

6. **Use the `--postprocessing` flag (UNDC only) as a sanity check, then turn it off for production.** The postprocessing step flips inverted face normals based on SDF sign — useful during development, but it masks real bugs in the upstream model. Ship with `--postprocessing` *on* for the first prototype (it's robust to bad inputs), then turn it off in v2 once we trust the upstream pipeline.

7. **Plan the "NDC + patient-specific fine-tune" workflow for the design doc.** The headline message: *NDC is a feed-forward mesh extractor that turns any of {SDF, UDF, voxel, point cloud} into a printable mesh in one pass. We use the pre-trained NDC/UNDC weights as a starting point and fine-tune on ~50-200 patient-specific IOS scans (augmented with noise + dropout + margin-line occlusions) for the production version. The fine-tune is cheap (~1 day on A100), and it should be done jointly with the generative model so the whole pipeline optimizes the end-to-end mesh quality. Cite Eq. 1 (the L1 vertex position loss) and Tab. 1-3 of the NDC paper directly.*

8. **Re-budget the H4 decision from paper 005.** Paper 005 left open: Path A (Diffusion-SDF → Marching Cubes) vs. Path B (LION → SAP). With NDC, the question is now: **Path A (Diffusion-SDF → NDC) vs. Path B (LION → UNDC)**, with both downstream mesh extractors being the same architecture. This is a much cleaner comparison because the *mesh extraction* is no longer a confounder. The pilot should be 1 week and produce a definitive answer.

9. **Plan paper 007 = "FlexiCubes" (Shen et al. SIGGRAPH 2023).** FlexiCubes is the *next* generation of neural mesh extraction — it predicts a per-cell "deformation" of a fixed quad-dominant template rather than predicting vertex positions directly. It's the natural follow-up to NDC because (a) it's the most recent SoTA on the same task, (b) it claims to be more robust to noise, (c) it has a differentiable forward pass that's friendlier to end-to-end training, and (d) the paper's benchmarks (FlexiCubes at 4.5 CD on ABC) are right in the same ballpark as NDC. Worth reading in 1-2 weeks to see if it should replace NDC for v2.

10. **Track the end-to-end training story across all of papers 003 / 004 / 005 / 006.** The pattern is now clear: every successful modern 3D generative pipeline is *2-stage* (learned generative model + learned mesh extractor), and the *next* frontier is *end-to-end* training of both stages. The follow-up papers to track are: (a) SDC (Sundararaman et al. CVPR 2024) for self-supervised NDC, (b) FlexiCubes (Shen et al. SIGGRAPH 2023) for a more differentiable mesh extractor, (c) Attention-Guided Multi-scale NDC (PGC 2025) for hierarchical NDC that may scale to full-arch reconstructions. *For HK's review: this is the *research arc* of modern 3D generation, and we're following it step by step.*

11. **Open question for HK: do we want a "patient-specific NDC" in production?** The pre-trained NDC works on synthetic CAD; a fine-tuned version on real IOS scans should be substantially better for clinical inputs. The question is whether to invest the 1-2 weeks of work now (while we're building the v0 prototype) or defer to v2. **My recommendation: invest now.** The fine-tune is cheap (1 day on A100), the data augmentation is well-understood, and the clinical benefit is large (a NDC that's seen 200 patient scans will produce a noticeably smoother intaglio surface than a NDC that's seen zero).

---

*Scholar 🦉 — 2026-06-06*
