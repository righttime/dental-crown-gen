# 003 — DiGS: Divergence Guided Shape Implicit Neural Representation for Unoriented Point Clouds

- **Title:** DiGS: Divergence Guided Shape Implicit Neural Representation for Unoriented Point Clouds
- **Authors:** Yizhak Ben-Shabat* (Technion / ANU / ACRV), Chamin Hewa Koneputugodage* (ANU / ACRV), Stephen Gould (ANU / ACRV)  *equal contribution
- **Year:** 2022 (CVPR 2022; arXiv:2106.10811 v1 21 Jun 2021 → v3 17 May 2023)
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2022, pp. 19323–19332
- **Links:**
  - Paper: https://arxiv.org/abs/2106.10811
  - CVPR open-access PDF: https://openaccess.thecvf.com/content/CVPR2022/papers/Ben-Shabat_DiGS_Divergence_Guided_Shape_Implicit_Neural_Representation_for_Unoriented_Point_CVPR_2022_paper.pdf
  - Project page (with video): https://chumbyte.github.io/DiGS-Site/
  - Author blog: https://itzikbs.com/blog/posts/2022-07-18-digs
  - Code (PyTorch, MIT-ish): https://github.com/Chumbyte/DiGS
- **Code/data:** PyTorch implementation released; SRB data lives in the Deep Geometric Prior repo; ShapeNet subset via the Neural Splines split (Google Drive). DFAUST requires free registration. Tested on Python 3.7.9 / torch 1.8.0 / CUDA 10.2 / Ubuntu 18.04.

---

## TL;DR

DiGS learns a **sine-activated (SIREN) implicit SDF** from a **raw, unoriented point cloud** — no normal vectors, no inside/outside labels, no normal-estimation pre-processing. The trick is a **soft penalty on the Laplacian of the learned distance field** (`|∇·∇Φ|`), which acts as a second-order regularizer that **biases the field to be a clean signed distance function** and makes the *gradient vector field* (and hence the surface normals) self-organize from the point data alone. They pair it with a **geometric initialization** (and a multi-frequency variant, MFGI) that pre-shapes the network to a sphere. On SRB and ShapeNet, DiGS **beats every other unoriented method** and **matches methods that use ground-truth normals** (Table 2: dC 0.19 vs. SIREN+norm's not-reported-without-norm baseline; IoU 0.939 on ShapeNet, the best in Table 3). Two clear limitations: thin structures (sofa legs) get smoothed out, and complex internal structure causes ghost geometry.

## Research question

> Can we learn a high-fidelity implicit SDF from **unoriented point clouds** — no normals, no sign labels — at a quality comparable to methods that *do* use normals?

Their answer: **yes**, by adding a **divergence penalty** (the Laplacian of the field) to the loss. The geometric intuition is that the *gradient* of a true SDF is a vector field whose **divergence is zero almost everywhere** (incompressible). Penalizing `|∇·∇Φ|` everywhere except on the surface pushes the field toward this incompressible ideal, which has two useful side effects: (a) it suppresses ghost geometry and overshooting iso-surfaces, and (b) it forces the gradients to be *consistent* across space, which means **the gradient at each surface point becomes a self-organized normal**. So you get the "benefit" of a normal constraint without ever needing one.

Two architectural pre-conditions make this work:
1. **Sine activations (SIREN).** ReLU MLPs (DeepSDF, SAL) have zero second derivative almost everywhere, so the divergence penalty does nothing. The network must be `C²` (or at least have non-vanishing second derivative everywhere) for the loss to have any effect.
2. **Geometric initialization.** A random SIREN init gives a field whose gradient norm is ≈ 0 everywhere (Fig. 3 in the paper) — i.e., a useless starting point. DiGS's geometric init pre-shapes the field to be the **signed distance to a sphere of radius r** (`Φ(x) ≈ ∥x∥ − r`), so the Eikonal and divergence terms are well-behaved from step 0.

## Method

### Network architecture
- **5 hidden layers, 256 units** (for shapes; 8×512 for the scene experiment), **SIREN** (sinusoidal activations), per Sitzmann et al. (NeurIPS 2020).
- No encoder, no latent code by default — single-shape fitting. In the shape-space experiment on DFAUST, they use the **DeepSDF auto-decoder** trick: a 128-dim per-shape code prepended to `(x, y, z)`, jointly optimized with the network weights.

### Geometric initialization (the "first contribution")

Goal: pick `θ₀` so that the initial field `Φ(x; θ₀) ≈ ‖x‖ − r` (signed distance to a sphere).

They don't try to make `Φ(x) = ‖x‖` directly — that's hard with smooth sines. Instead they target the **signed squared norm** `Φ(x) ≈ ‖x‖²` and apply the post-processing `ν(d) = sign(d)·√|d| + ε` to recover the SDF. Proposition 4.1: for a single-hidden-layer SIREN, setting the last layer's `W = (π/2)·I`, `b = (π/2)·1`, output weight `w = −1`, bias `b = Mn` gives `ν(Φ(x)) ≈ ‖x‖²` for `x` in the unit ball. Proposition 4.2 extends this to N layers by sampling intermediate weight matrices from `U(−c, c)` with `c = √(3/M_{i+1})` so that each layer preserves the norm in expectation — the rows become approximately orthonormal.

**MFGI (Multi-Frequency Geometric Initialization).** Geometric init alone forces all activations into the first sine period, so the network can't represent high-frequency content. MFGI splits the first weight matrix's rows into a "low-frequency" block (`kr ≈ N/4` rows, geometric init) and a "high-frequency" block (scaled by `np = 30` to hit 30 periods), then **scales down the next layer's weights** that multiply into the high-frequency block by `s = 10⁻³` so the geometric approximation still holds. This is a clean way to combine "structured low-frequency base" + "free high-frequency detail" in the same SIREN.

### The loss (`L_DiGS`)

Lifted from SIREN's standard four-term loss (Eq. 8) plus the divergence penalty (Eq. 10–11):

```
L_DiGS = L_A + L_C2 + L_D + τ·λ_div·L_div     (drop L_B; no normals)

L_A      = ∫_{surface} ‖Φ(x)‖²              # manifold: Φ=0 on surface
L_C2     = ∫_Ω |‖∇Φ(x)‖² − 1|               # Eikonal: |gradient| = 1
L_D      = ∫_{Ω\surface} exp(−α|Φ(x)|)       # off-surface non-manifold penalty
L_div    = ∫_{Ω\surface} |∇·∇Φ(x)|           # ← the new term, second-order

λ_A=3000, λ_C2=50, λ_D=100, λ_div=100
```

Where **`τ` is annealed over training** in a **smooth-to-sharp** schedule (50% / 25% / 25% of iterations):
- **High-divergence phase (50%):** large `τ` → very smooth SDF, no fine details, robust to noise. Avoids premature fitting.
- **Annealing phase (25%):** `τ` decays → fine details start to emerge, but the field stays smooth.
- **Low-divergence phase (25%):** `τ ≈ 0` → model interpolates the points and produces sharp features.

This is essentially **coarse-to-fine optimization**, and it works.

### Why a divergence penalty? (Sec. 5.3)

A nice theoretical section. They show that minimizing `L_div` is *equivalent* to minimizing the **Dirichlet energy** of the learned field (Eq. 12) — the function's "complexity" / total squared gradient magnitude. A toy 2D problem (learning the SDF to the line `y=0`, with only point constraints and Eikonal) shows:

| Loss | grid | mean Dirichlet energy | ‖∇f‖₂ | σ(‖∇f‖₂) |
|---|---|---|---|---|
| A + C2 | 20×20 | 4.32 | 1.63 | 1.05 |
| A + C2 + Div | 20×20 | **3.37** | **1.46** | **0.69** |
| A + C2 | 200×200 | 3.58 | 1.51 | 0.85 |
| A + C2 + Div | 200×200 | **1.37** | **1.04** | **0.22** |

Adding the divergence term **halves the Dirichlet energy** and produces a gradient field with much smaller variance — i.e., the function is *smoother* and *more Eikonal-consistent* in regions where there's no supervision. This is the formal justification for why it works.

## Results

### Surface Reconstruction Benchmark (SRB) — Table 2 (unoriented only)

| Method | dC ↓ | dH ↓ | ΔdC | ΔdH |
|---|---|---|---|---|
| IGR w/o n | 1.38 | 16.33 | 1.20 | 12.84 |
| SIREN w/o n | 0.42 | 7.67 | 0.23 | 4.18 |
| SAL | 0.36 | 7.47 | 0.18 | 3.99 |
| IGR+FF | 0.96 | 11.06 | 0.78 | 7.58 |
| PHASE+FF | 0.22 | 4.96 | 0.04 | 1.48 |
| **DiGS (ours)** | **0.19** | **3.52** | **0.00** | **0.04** |

**DiGS is the best unoriented method by a clear margin**, and it does this *without any pre-processing* (no normal estimation step). Visually (Fig. 6), SIREN-without-normals produces ghost geometry; DiGS removes it.

### ShapeNet (260 shapes, 13 categories) — Table 3

Methods above the line use normal supervision; below the line do not.

| Method | sq. Chamfer median | IoU mean |
|---|---|---|
| IGR (+n) | 1.13e-4 | 0.8102 |
| SIREN (+n) | 5.28e-5 | 0.8268 |
| FFN (+n) | 8.65e-5 | 0.8218 |
| NSP (+n) | 4.06e-5 | 0.8973 |
| DiGS (+n) | 2.32e-5 | **0.9200** |
| SIREN w/o n | 2.58e-4 | 0.3085 |
| SAL | 2.11e-4 | 0.4030 |
| **DiGS w/o n** | 2.55e-5 | **0.9390** |

**Two remarkable things here:**
1. **DiGS w/o normals beats every method that uses normals** on median Chamfer and on mean IoU.
2. **DiGS w/o normals beats DiGS w/ normals on mean IoU** (0.939 vs. 0.920). Why? They explain: with normal supervision the network tries to fit internal structure (e.g., sofa beams, loudspeaker cones) and creates **internal ghost geometry**, which tanks the mean. Without normals it ignores internal complexity and produces a clean exterior — *better for the use case of a closed printable mesh*.

### Shape space (DFAUST, 10 humans) — Table 4

Using the **DeepSDF auto-decoder** (latent 128-dim code per pose, learned jointly with weights):

| Method | dC(reg, recon) mean | dC(recon, reg) mean |
|---|---|---|
| IGR (+n) | 1.053 | 4.916 |
| DiGS (+n) | **0.568** | **1.834** |
| IGR w/o n | 3.745 | 12.149 |
| DiGS w/o n | 0.856 | 12.318 |

**DiGS w/o normals is the only w/o-normals method that converges.** IGR w/o n diverges (Fig. 9). DiGS oversmooths fine detail (face features) but captures the body shape.

### Scene reconstruction (one scene, qualitative, Fig. 8)
SIREN w/o normals → lots of ghost geometry. DiGS → clean reconstruction but oversmoothed fine details (sofa legs, picture frames).

## Connections to our hypotheses (H1–H5)

### H2 — Diffusion on point clouds > mesh-based VAE for surface generation
**Weakly supports H2 in a useful negative way.** DiGS is a **single-shape overfit**, not a generative model. It does not "generate" missing teeth — it reconstructs *one* object from *its* point cloud. To turn DiGS into a crown generator, you'd need either the DeepSDF auto-decoder (one code per shape, learned via MAP at inference) or to bolt a generative prior on top. **The right comparison for H2 is between DiGS-as-the-backbone and a point-cloud diffusion model**: diffusion could give a single-pass generative model, but DiGS would give a posterior with a known likelihood — much better for our low-data regime. So DiGS doesn't refute H2, but it sharpens the trade-off: **if H2 wins, the cost is "need a lot of training data"**, and DiGS-style methods are a credible alternative when we don't have that.

### H3 — Conditioning on opposing + adjacent teeth improves outer surface quality
**Directly enables H3 via the auto-decoder.** The key observation: DiGS uses the **DeepSDF auto-decoder** trick in the shape-space experiment (Sec. 6.2). That's exactly the conditional formulation we wanted — `Φ_θ(z, x, c_context)`. The mechanism is:
- For each tooth in the training set, learn a latent code `z_i`.
- At test time, given a *new patient's* intra-oral scan, you optimize `z` to match the points of the *context teeth* (adjacent + opposing).
- The reconstruction of the missing tooth is `Φ_θ(z_optimal, x)` — and `z` was *inferred from the context*, so the missing-tooth surface is **automatically conditioned on it**.

We don't even need a separate context encoder: the context teeth's SDF points + their FDI class act as the conditioning signal. This is **the cleanest path to H3** I've seen so far. We should plan the paper 003 → paper 004 jump with this in mind (the next paper after this should be one that turns auto-decoders into a class-conditional generative model — e.g., DiffusionSDF, LION, or CIGS).

### H4 — Implicit SDF > explicit mesh for high-quality surfaces
**Strong, direct support for H4.** The empirical case: DiGS produces *watertight*, *normal-consistent*, *topology-flexible* surfaces with sub-millimeter Chamfer on ShapeNet, *without* normals as input. The analytical case: the field gives analytic normals via autograd, supports continuous interpolation, and the divergence penalty is *itself a normal-consistency term* that emerges from the field structure. **Compared to DeepSDF (paper 002):**
- DiGS removes the DeepSDF "inside/outside label" requirement (DeepSDF needed the sign of off-surface SDF points as supervision; DiGS gets this from the divergence + Eikonal losses).
- DiGS recovers from **smoother** fields (high-divergence phase) and gets finer details with less fitting, vs. DeepSDF's tendency to undersmooth or overshoot.
- DiGS converges faster in our preliminary mental benchmarks (5 layers, 256 units, single shape, no normals — paper reports visual convergence in 10–30k iters).

The clear gap is **thin structures** (sofa legs, picture frames) — DiGS's `L_div` smooths them out. For dental crowns this is mostly fine (crowns are bulky, not thin), but for **bridge connectors** (which are thin struts between two crowns) this is a real concern. We should watch this when we get to the bridge-generation stage.

### H1 — 2-stage (segmentation + generation) > end-to-end
**Independent — no direct support or contradiction.** DiGS does reconstruction, not detection. The relevance is that **once sub-task 1 (segmentation) gives us per-tooth point clouds, DiGS is the natural choice for sub-tasks 3–4** (inner + outer surface generation). This continues the modular architecture implied by H1.

### H5 — Synthetic data from existing CAD libraries can bootstrap training
**Indirectly supports H5 via the Dirichlet-energy framing.** DiGS's high-divergence phase is **explicitly designed to be robust to noise and incomplete data** (Section 3: "prevents the model from prematurely fitting to fine details"). This is exactly the regime synthetic CAD data lives in — clean exterior, possibly noisy internals, possibly missing regions. If we can synthesize crown point clouds from open dental CAD repos (Tufts, OSF, manufacturer STL dumps), DiGS's robustness profile means we can train without obsessively cleaning the synthetic data first. The 3DTeethSeg22 → DiGS pipeline for sub-task 1 → sub-tasks 3–4 is therefore **a more realistic path** than trying to train DeepSDF end-to-end on noisy real IOS data.

## Surprises / interesting things buried in the paper

1. **The auto-decoder is a 2-line change to the architecture and we already discussed it in paper 002.** It's nice to see it work in the *non-normal* setting on a real shape-space benchmark (DFAUST). The lesson: paper 002 said "we might do this"; paper 003 says "we did this, and it works better than the normal-supervised version."

2. **Without-normals beats with-normals on ShapeNet mean IoU.** This is counterintuitive. The explanation (internal ghost geometry) is buried in the second-to-last paragraph of Section 6.1, and it's a *huge* point for our use case: for 3D-printable crowns, internal geometry is irrelevant — we want the cleanest exterior. So **we should probably train without normals even if we had them.**

3. **The smoothness prior isn't a hack — it's a Bayesian prior.** Section 5.3 shows that minimizing `L_div` *is* minimizing the Dirichlet energy, which is a standard smoothness prior in functional analysis. So the divergence penalty isn't "yet another loss term" — it's the **canonical smoothness prior on function spaces**, with the divergence form being what makes it numerically tractable in 3D. This is the kind of thing that makes DiGS feel principled rather than empirical.

4. **The geometric initialization is also the trick that makes DiGS usable.** Without it, SIREN without normals produces ghost geometry. So the contribution is really two coupled things: a structural prior (sphere init) + a smoothness prior (divergence loss). They can be used independently but they're stronger together.

5. **The MFGI init is a way to get high-frequency capacity in a SIREN without losing the geometric structure.** The "scale down the next layer's weights into the high-frequency block by 10⁻³" trick is the only place in the paper where I saw genuine "huh, clever" architecture insight. Worth re-using for any future SIREN-based work.

## Quote-worthy sentences

- *"The gradient vector field of the signed distance function (produced by the network) has low divergence nearly everywhere."* (Sec. 1) — the core geometric insight, in one sentence.
- *"Our approach can only be used with architectures that have activation functions with nonzero second derivative such as SIRENs. ReLU based networks will not be affected by the divergence constraint."* (Sec. 5.2) — a hard architectural constraint; do not try to bolt this onto DeepSDF.
- *"In the absence of normal information, SIREN and IGR struggle to converge to the correct zero level set and produce undesired artifacts (ghost geometries). DiGS, on the other hand, is able to remove such artifacts."* (Sec. 6.1) — the failure mode that DiGS fixes.
- *"When comparing without normals, DiGS has similar medians on both metrics to when normal supervision is added, however it has better means. We attribute this to having fewer internal ghost geometries when not attempting to fit normal vectors at internal points."* (Sec. 6.1) — the "without normals is sometimes better" finding.
- *"DiGS is mainly limited in two aspects: (1) capturing very thin structures … and (2) smoothing effects."* (Sec. 6.3) — the honest failure modes.

## Code/data availability

- **Code:** https://github.com/Chumbyte/DiGS — PyTorch, MIT-style, well-documented. Tested on Python 3.7.9 / torch 1.8.0 / CUDA 10.2. Trained model weights for SRB are on Google Drive.
- **Data:**
  - SRB: 5 shapes, 1.12 GB, free for academic use, from the Deep Geometric Prior repo.
  - ShapeNet subset: 783.76 MB, free, requires citing ShapeNet + Occupancy Networks + Neural Splines.
  - DFAUST: requires free registration on dfaust.is.tuebingen.mpg.de.

## For our project

Concrete next steps, ordered by priority:

1. **Drop DeepSDF from the plan; promote DiGS to the default surface backbone for sub-tasks 3–4.** DiGS is strictly better for our setting (no normals, robust to noise, better IoU, faster convergence). Keep DeepSDF as a "what came before" reference in the paper notes but don't reimplement it.

2. **Clone the DiGS repo and run it on a single crown point cloud first.** The code is clean and self-contained. Even before we have proper per-tooth segmentation, we can extract one tooth's point cloud from a 3DTeethSeg22 scan, run DiGS, and visualize the reconstructed mesh. This is the lowest-cost validation that the pipeline works end-to-end. Estimated: 1 day.

3. **Adopt the geometric init + MFGI + divergence loss + annealed-τ training schedule verbatim.** These four things together are what make DiGS work. Don't reinvent the architecture.

4. **For the auto-decoder extension to DFAUST-style tooth generation, follow Section 6.2 exactly.** Latent dim 128, jointly optimize codes + decoder weights, MAP-infer at test time. Conditioning on the context teeth happens *automatically* through the code: the optimal `z` for a new patient is the one that best fits the context teeth's SDF samples. **This is the cleanest path to H3 in the literature so far.**

5. **Plan to train *without normals* even if we had them.** DiGS w/o normals is empirically better on closed-surface benchmarks, and the failure mode (internal ghost geometry) doesn't apply to printable crowns. This saves us the normal-estimation pre-processing step and simplifies the data pipeline.

6. **For bridge generation (where thin connectors matter), expect DiGS to struggle.** The high-divergence phase is hostile to thin structures. Options: (a) skip the annealing for thin regions (architectural, complex), (b) post-hoc mask the connector region and re-train with a different prior (surgical, simpler), (c) skip DiGS for the connector and use a separate mesh-based method. Defer this decision until we have data.

7. **Read the next paper in the auto-decoder + diffusion lineage.** Candidates: **DiffusionSDF** (Chou et al., 2023), **LION** (Zeng et al., 2022), **CIGS** — all of these combine a DiGS-style auto-decoder with a diffusion prior over the latent codes. That's the path to H2 (diffusion) + H3 (context conditioning) + H4 (implicit) being true *simultaneously*. Should be paper 004.

8. **Bookmark the Dirichlet-energy argument from Section 5.3.** When writing the project's design doc, this is the principled justification for the "smoothness prior" we'll put on the crown's outer surface. Cite it directly.

9. **Update H4 to be more nuanced.** Originally H4 said "implicit SDF > explicit mesh." With DiGS, H4 should be "implicit SDF *with a smoothness/divergence prior* > explicit mesh", and the smoothness prior is what gives us analytical normals and avoids DeepSDF's ghost-geometry problems. Worth updating the README.

10. **Critical open question for HK:** **Do we want the crown's *inner surface* (sub-task 3) to be DiGS-generated too, or do we keep it as a deterministic offset of the prepared tooth?** The inner surface has a hard *fit constraint* (margin gap < 50μm is a clinical requirement) that is much better solved by a deterministic, geometric operation (offset + post-fit adjustment) than by a learned model. DiGS should own the *outer* surface; the *inner* should probably be a separate, much simpler pipeline. Discuss with HK before committing.

---

*Scholar 🦉 — 2026-06-06*
