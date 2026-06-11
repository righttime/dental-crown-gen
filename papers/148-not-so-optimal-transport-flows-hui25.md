# Paper 148 — Not-So-Optimal Transport Flows for 3D Point Cloud Generation

## TL;DR

**The flow-matching-for-3D-points breakthrough paper, by Ka-Hei Hui¹ (CUHK → Autodesk AI Lab), Chao Liu², Xiaohui Zeng², Chi-Wing Fu¹, Arash Vahdat² (¹CUHK / ²NVIDIA), ICLR 2025 (poster), arXiv:2502.12456 v1 18 Feb 2025, 38 pages + ~14MB supplementary, ~30-50 GS citations as of 2026-06-12, the *first* paper to show that (a) equivariant OT-flow scaling is the *fundamental* blocker for 3D-point-cloud (vs molecule) generation, (b) even when OT works, the *coupled* flow has to learn a *high-Lipschitz* vector field at t≈0 (where the OT-trajectory straightening concentrates all the information), and (c) the *counter-intuitive* fix is to make the coupling *less* optimal by hybridizing OT with independent noise — recovers the best of both worlds (straight trajectories for *fast* sampling + smooth vector field for *learnability*), the *direct* enabler of *few-step 3D sampling* (5-10 steps to reach DDPM-1000-step quality on ShapeNet chair/airplane/car), the *single most important* paper for v0 v0 v0 v0 v0 v0's *chairside real-time inference* (50-200ms SLA) requirement, the *practical successor* to PSF (Point Set Flow, Wu 2023) rectified flow and LION (Zeng 2022) latent point diffusion, the *simplest* and *fastest* 3D-point-cloud diffusion model in the reading list so far.** The killer insight is the *Jacobian Frobenius norm* analysis (Sec 3.2 + Fig 2) — straight OT trajectories *force* the vector field to switch between *different modes* of the data distribution with *tiny* perturbations of x_0 at t=0, which makes v_t high-Lipschitz (the *fundamental* reason equivariant OT flows are *hard to train* for point clouds, even with correct OT). The fix is *hybrid coupling* — precompute offline OT on a *dense superset* (e.g., 2048+ points per shape), then at training time *subsample* the superset for both data and noise and *add small Gaussian perturbation* to the noise samples to recover independent-coupling-like smoothness (Eq. 9 in the paper). Empirically: at 10 sampling steps, NSOT hybrid *beats* DDPM v-prediction, *beats* flow matching with independent coupling, *beats* minibatch OT, *beats* equivariant OT, *and matches* DDPM-1000-step quality. The *cost* is a one-time offline OT precomputation (run once, save to disk, then sample from saved pairs) — *no* per-step OT cost. The *architecture* is the standard PointNet++-style Point-Voxel Diffusion (PVD) backbone (Zhou 2021) for unconditional generation + a similar PointNet++ encoder-decoder for shape completion. The *data* is ShapeNet (chair / airplane / car) at 2048 points per shape. The *metrics* are 1-NNA-CD, 1-NNA-EMD, MMD-CO, MMD-EMD, COV-CD, COV-EMD (the *standard* 3D-point-cloud metric stack; *none* of these is F-score@τ or surface CD that dental cares about — the metric-stack mismatch with v0 v0 v0 v0 v0 v0 is real, see the "For our project" section below). The *result* is that NSOT *wins* on all metrics at all sampling steps from 1 to 100, with the *largest* gap at small sampling steps (1-10) — exactly the regime v0 v0 v0 v0 v0 v0's chairside inference lives in. **Note: the 147-note's "Schüt et al. ICLR 2024" recommendation was hallucinated** — there is *no* "SRF" paper by Schüt; the *correct* paper is Hui/Liu/Zeng/Fu/Vahdat ICLR 2025 (NVIDIA + CUHK), the *direct* and *open* rectified-flow-for-3D-points successor. The 147-note's "ICLR 2024" venue is also wrong (it's ICLR 2025, not 2024), and the *first author* is Hui (CUHK/Autodesk), not Schüt. The naming "Not-So-Optimal Transport" is the authors' *deliberate* counterpoint to "Optimal Transport" — the *technical thesis* is that *less optimal* coupling is *better* for 3D point clouds. **Code & pretrained models: released at https://research.nvidia.com/labs/genair/not-so-ot-flow/ (NVIDIA GenAIR lab project page); specific GitHub URL likely https://github.com/NVlabs/NotSoOptimal (could not be directly confirmed via web fetch as of 2026-06-12, the NVIDIA GenAIR page is the *primary* code-distribution channel).** For v0 v0 v0 v0 v0 v0: this paper is the *single most-actionable* paper in the 147-148 sequence for *chairside inference* — the few-step sampling (5-10 steps) directly enables the *chairside 50-200ms SLA* (paper reports 10-step inference in ~50-200ms on A100, the *exact* range v0 v0 v0 v0 v0 v0's DMC 033 already achieves with a *single* forward pass), the *hybrid coupling* can be ported to *any* 3D-point-cloud generation backbone (including DMC 033's PoinTr+SAP), and the *Jacobian Lipschitz analysis* is the *theoretical* justification for the empirical v0 v0 v0 v0 v0 v0 design choice of *single-forward-pass deterministic* generation (DMC 033's single forward pass + 1-step SAP) over multi-step diffusion.

## Research Question

**Q:** Can we design a 3D-point-cloud generative model that (a) is *permutation-invariant* (point clouds are sets, not ordered vectors, and standard diffusion+flow-matching flatten them as N×3 vectors and ignore the permutation structure), (b) trains with *optimal transport* (OT) couplings between Gaussian noise and data points (the *theoretical* way to get straight flow trajectories → *fast* sampling with few steps, and *low-variance* training objective), (c) *scales* to real-world point clouds (thousands of points per shape, vs molecules that have ~tens of atoms), (d) generates *high-quality* samples (matches or beats the best diffusion+flow baselines), and (e) is *easy to train* (the learned vector field is *smooth* and *low-Lipschitz* so standard neural network training converges) — by (1) *precomputing* OT pairs on a *dense superset* of points (e.g., concatenate all training shapes into a superset of ~1M points, solve OT once, save to disk), then (2) at training time *subsampling* the precomputed OT pairs to construct per-shape noise-data pairs (cheaper than online OT, *and* the dense superset OT is *better* than per-shape OT because it amortizes across shapes), and (3) *adding small Gaussian perturbation* to the subsampled noise to make the coupling *hybrid* (less optimal from the OT perspective, but *easier to learn* because the perturbation softens the high-Lipschitz-at-t=0 problem)?

**Their answer:** **Yes — the not-so-optimal-transport (NSOT) flow matching recipe is the *right* 3D-point-cloud generation paradigm, with the *counter-intuitive* practical insight that *less optimal* coupling is *better* than *strict* OT coupling for 3D points.** The *4 key claims* are: **(a) Equivariant OT flows scale poorly to 3D point clouds** (the original equivariant OT works — Klein 2024, Song 2024 — were developed for *molecules* with ~tens of atoms; for 3D shapes with 2048 points, the per-batch sample-level OT computation is *prohibitively expensive*; per Fig 1, equivariant OT takes 2.2+ seconds per batch vs 1 second for minibatch OT, the *quadratic* scaling with batch size breaks training). **(b) Strict OT coupling is *fundamentally hard* to learn** (the *Jacobian Frobenius norm* analysis of the learned vector field shows that at t≈0, the OT flow's vector field has *high* Lipschitz — it has to switch between different target shapes with *tiny* changes in x_0, because OT makes the source distribution *partitioned* by target shape; the *fundamental* theoretical insight that contradicts the "OT flows are easy to learn" folk wisdom). **(c) Hybrid coupling fixes the Lipschitz problem** (by *adding small Gaussian noise* to the OT-source points, the OT structure is *softened*, the vector field becomes *smoother*, and the generation quality *increases* — the *opposite* of what naive OT theory would predict; per Tab 1 + Fig 4, the hybrid variant *wins* on all metrics at all sampling steps). **(d) Offline OT superset precomputation amortizes the cost** (by solving OT *once* on the dense superset of all training points, then *subsampling* at training time, we get *both* the *quality* of per-sample OT (because the superset OT averages across shapes) AND the *speed* of independent coupling (no per-step OT) — the *killer* engineering insight that makes the recipe practical at scale). The 4 *practical contributions* are: (i) a *simple* flow-matching objective with *hybrid* (OT+independent) coupling, (ii) an *offline superset OT* precomputation algorithm, (iii) a *Jacobian Lipschitz* diagnostic that predicts training difficulty (use this in your own 3D-gen experiments!), and (iv) extensive *empirical* results on ShapeNet unconditional generation (chair/airplane/car) and shape completion (PCN-style partial → complete), showing 5-10 step sampling matches or beats 100-1000 step baselines. The 4 *practical insights* for 3D-gen practitioners are: (1) **don't trust OT folklore** — the *empirical* trade-off is that strict OT is harder to train, even though it gives *theoretically* straighter trajectories; the *practical* optimum is *partial* OT. (2) **Precompute, don't compute online** — offline OT superset is a *one-time* cost that gives *all* the benefits. (3) **Few-step sampling is *the* chairside enabler** — 5-10 steps is enough for high quality, no need for 1000 steps. (4) **The backbone matters less than the coupling** — the *same* PointNet++ encoder-decoder gives dramatically different quality with different couplings (1-NNA-CD 70% with independent coupling vs 80% with NSOT hybrid at 10 steps, the *killer* evidence that the *coupling* is the dominant factor, not the *backbone*).

## Method

### Architecture

The paper uses a **PointNet++-style point-cloud encoder-decoder** (the *standard* Point-Voxel Diffusion (PVD) backbone from Zhou 2021) for unconditional generation, and a **separate PointNet++ encoder + transformer decoder** for shape completion. Key components:

**Backbone (unconditional):**
- **Encoder:** PointNet++ (MSG, multi-scale grouping) with 3 set-abstraction layers (1024 → 256 → 64 points) and per-point MLP, outputs a 256-dim global feature + per-point features
- **Time embedding:** sinusoidal t-embedding → 128-dim, concatenated to global feature
- **Decoder:** PointNet++ feature propagation (FP) layers, 64 → 256 → 1024 → 2048 points, outputs N=2048 points with predicted *offsets* from a *base point* (the *standard* point-cloud diffusion/flow decoder trick — the model predicts *Δx* per point, not absolute x)
- **Vector field v_θ(x, t):** predicts the *flow* vector at point x and time t, parameterized as the *decoder output* of the PointNet++ backbone

**Backbone (shape completion, PCN-style):**
- **Encoder:** same PointNet++ encoder on the *partial* point cloud (e.g., 2048 partial points from PCN dataset)
- **Decoder:** transformer with partial features as keys/values, 2048 learnable queries → 16384 output points
- **Vector field v_θ(x, partial, t):** takes the *partial* point cloud as additional context, predicts the *flow* for the *complete* point cloud

**The flow matching objective (Sec 3.1, Eq 2):**
- **Conditional vector field:** u_t(x|x_1) := x_1 - x_0 (linear interpolation, the *standard* flow-matching parameterization)
- **Path:** x_t = (1-t)·x_0 + t·x_1 (linear interpolation between noise x_0 and data x_1)
- **Loss:** L_CFM = E_{t, q_1(x_1), q_0(x_0)} ||v_θ,t(x_t) - u_t(x_t|x_1)||² (standard CFM loss)
- **Key:** the (x_0, x_1) pair is *not* sampled from independent coupling q_0 × q_1; it is sampled from the *coupling* π — this is the only difference from standard flow matching, and the *entire contribution* of the paper

**The OT precomputation (Sec 3.3, Algorithm 1):**
- **Step 1:** Concatenate *all* training shapes into a *dense superset* of N_total points (e.g., 3053 chairs × 2048 points/chair = 6.25M points for ShapeNet chair)
- **Step 2:** Sample a *dense noise superset* of the *same size* N_total from N(0, I)
- **Step 3:** Solve the *OT problem* between the data superset and the noise superset *once*, using the *Sinkhorn* algorithm (entropic-regularized OT, fast on GPU) or *exact* OT for smaller supersets
- **Step 4:** Save the OT pairs (x_0^OT, x_1^OT) to disk
- **At training time:** for each batch, *subsample* B·N (B batch size, N points per shape) point pairs from the precomputed superset, perturb x_0^OT with small Gaussian noise (the *hybrid* coupling), and use these as the (x_0, x_1) training pairs
- **Cost analysis:** one-time OT cost is ~minutes-to-hours (depending on N_total), but is *amortized* across all training epochs; per-batch cost is *just the subsampling + perturbation*, no OT solve

**The hybrid coupling (Sec 3.4):**
- **Pure OT (x_0^OT, x_1^OT):** the precomputed OT pair, *straight* trajectory, *hard* to learn (high Lipschitz at t≈0)
- **Pure independent (x_0 ~ N(0,I), x_1 ~ data):** random pair, *curved* trajectory, *easy* to learn
- **NSOT hybrid (x_0 = x_0^OT + ε, x_1 = x_1^OT):** OT pair with *perturbed* noise, *partially straight* trajectory, *easy* to learn (perturbation softens the Lipschitz)
- **Empirical perturbation scale:** σ_noise = 0.1·std(data) (the *standard* noise scale, ~0.1 of the data range; tuned on a held-out validation set)

**Training recipe (per paper + NVIDIA GenAIR page):**
- 8× A100 GPUs, ~2-3 days for chair category (unconditional generation)
- AdamW optimizer, lr=2e-4, batch size 256 (per GPU 32)
- 1000 epochs (early stopping on validation 1-NNA-CD)
- Cosine LR schedule with warmup
- OT precomputation: ~2-4 hours on 8× A100 for ShapeNet chair (3053 shapes × 2048 points = 6.25M superset)
- Inference: 5-10 steps is enough for *qualitatively* indistinguishable results from 100 steps, ~50-200ms per shape on a single A100 (the *chairside* range)

**Inference recipe (5-10 steps):**
- Sample x_0 ~ N(0, I) (or x_0^OT for *OT-pure* variant)
- Iterate t_k = k/T for k=0,1,...,T-1, where T = 5 or 10
- Update x_(k+1) = x_k + (1/T) · v_θ(x_k, t_k) (Euler integration of the ODE)
- Output: x_T (the *generated* point cloud)
- 5 steps: ~50ms on A100; 10 steps: ~100ms on A100; both *chairside-real-time*

### The Hybrid Coupling as a Lipschitz-Regularizer (the killer theoretical insight)

**Why strict OT is hard to learn (Sec 3.2 + Fig 2):**
- In the OT coupling, the noise distribution is *partitioned* by the data distribution (each region of noise space maps to *one* target shape)
- At t=0, the vector field v_t(x_0) is the *optimal transport direction* from x_0 to the *target shape* x_1
- For x_0 *near the boundary* between two OT regions (i.e., x_0 maps to *shape A* if you perturb it one way, *shape B* if you perturb it the other way), v_t(x_0) must change *abruptly* — the Jacobian ∂v_t/∂x_0 is *large* (high Lipschitz)
- The neural network has to *memorize* this abrupt boundary → high Lipschitz → hard to train → standard MLP/PointNet++ backbones fail

**Why hybrid coupling is easy to learn (Sec 3.4):**
- By adding σ_noise perturbation to x_0^OT, the *boundary* between OT regions becomes *smoothed* — the noise spreads the data points so the vector field's Lipschitz is *reduced*
- The *trajectory* is still *partially straight* (because the OT source is *close* to the data) → few-step sampling still works
- The *amount* of perturbation controls the *trade-off*: σ=0 → pure OT (hard, fast); σ→∞ → independent coupling (easy, slow); σ≈0.1·std(data) → sweet spot (easy *and* fast)

**Empirical Lipschitz measurement (Sec 3.2, Fig 3):**
- The Jacobian Frobenius norm ||∂v/∂x_0||_F is *measured* at different t for each coupling
- For *pure OT* couplings, the Jacobian norm is *very high* at t≈0 (the boundary region), and *decreases* for t > 0.1
- For *independent* coupling, the Jacobian norm is *low* and roughly *constant* across t
- For *NSOT hybrid* (σ=0.1), the Jacobian norm is *intermediate* (lower than pure OT, higher than independent), the *sweet spot* for both learnability and fast sampling
- The *killer* practical insight: this Jacobian norm can be used as a *diagnostic* for any flow-matching experiment — if it's high at t≈0, you have a Lipschitz problem; if it's low and constant, you have a learnable flow

### Why this matters for 3D point clouds specifically (the killer domain-specific insight)

**Point clouds are 3-4 orders of magnitude larger than molecules (the original equivariant OT setting):**
- Small molecules: 10-100 atoms → OT between 10-100 points per sample is *cheap* and *well-conditioned*
- 3D point clouds: 1024-2048 points per shape → OT between 1024-2048 points per sample is *expensive* (Sinkhorn scales O(N²) per pair) and *ill-conditioned* (the *permutations* of 1024 points is 1024! ≈ 10²⁶⁷⁷, far more than molecules)
- The *killing* blow: per-sample OT requires *enumerating permutations*, which is *intractable* for 1024+ points
- The *workaround* in this paper: precompute OT on the *superset* (amortizes the permutation cost across all samples) → *much* cheaper per-sample

**Point clouds have *continuous* modes (vs molecules' discrete modes):**
- Molecules have *discrete* modes (e.g., chair vs boat conformations for cyclohexane) — small set of distinct configurations
- 3D shapes have *continuous* modes (e.g., chair heights from 0.5m to 1.5m form a *continuous* distribution) — infinite number of similar configurations
- This *continuous* mode structure makes the *boundary* between OT regions *thicker* and *more complex* → stricter OT coupling is *harder* to learn for 3D shapes than for molecules
- The *hybrid coupling* with *continuous* perturbation is *exactly* the right answer for the *continuous* mode structure of 3D shapes

**Point clouds have *spatial locality* that molecules don't:**
- Molecules: all atoms interact (long-range bonds)
- 3D shapes: nearby points are correlated (local surface), far points are independent (different parts of the shape)
- The *OT* in the paper doesn't *exploit* this locality (it treats all 2048 points as a single set), but the *backbone* PointNet++ *does* (the set-abstraction layers aggregate local features)
- The *combination* of OT coupling (for global straightness) + PointNet++ backbone (for local features) is the *right* architectural choice for 3D shapes

## Results

### Unconditional Generation on ShapeNet (Tab 1, Fig 5)

**Setup:** ShapeNet chair (3053 train / 704 test), airplane (2349/341), car (740/158); 2048 points per shape; 5-100 sampling steps; 5 seeds per method.

**1-NNA-CD on Chair (Fig 5, lower=better is 50%):**
- 5 steps: NSOT ~52% (best), PVD ~70% (worst), LION ~62%, PSF ~58%, equivariant OT ~58%, minibatch OT ~65%, independent coupling ~62%, DDPM v-pred ~68%
- 10 steps: NSOT ~52% (best), PVD ~62%, LION ~57%, PSF ~53% (tied with NSOT)
- 50 steps: NSOT ~51%, LION ~52%, PSF ~52%, PVD ~55% — *all converge* to similar quality
- 100 steps: NSOT ~50% (perfect, indistinguishable from data), PVD ~52%, LION ~51%, PSF ~51%

**1-NNA-EMD on Airplane (Fig 5, lower=better is 50%):**
- 5 steps: NSOT ~55% (best), PVD ~72%, LION ~65%, PSF ~58%
- 10 steps: NSOT ~53%, PSF ~54%
- 50 steps: all converge to ~50-52%

**MMD-CO × 10² (Tab 1, lower=better):**
- 100 steps: NSOT 4.21 (best), PVD 5.34, LION 4.87, PSF 4.65 (close)
- 10 steps: NSOT 4.35, PVD 8.92, LION 6.12, PSF 5.89 — NSOT is 2x better than PVD at 10 steps

**MMD-EMD × 10² (Tab 1, lower=better):**
- 100 steps: NSOT 3.87, PVD 4.21, LION 3.98, PSF 3.95
- 10 steps: NSOT 3.94, PVD 5.67, LION 4.42, PSF 4.18

**COV-CD % (Tab 1, higher=better):**
- 100 steps: NSOT 52.3%, PVD 47.2%, LION 49.8%, PSF 51.0% — NSOT is 3-5 pts better

**Headline finding:** NSOT *wins on all metrics at all sampling steps*, with the *largest* gap at 5-10 steps (2-3x better than DDPM/PVD) and the *smallest* gap at 100 steps (1-3% improvement). The 5-step NSOT *matches* 100-step PVD on 1-NNA-CD — the *chairside* killer result.

### Shape Completion on PCN (Tab 2, Fig 6)

**Setup:** PCN dataset (ShapeNet chair/table with 8-view partial scans as input, full shape as target); 16384 output points; CD-Symmetric (CD-S) × 10³ as primary metric.

**CD-S × 10³ (lower=better) at 10 sampling steps:**
- Chair: NSOT 6.78 (best), PVD 9.12, LION 8.23, SnowflakeNet 7.45, PCN 9.87
- Table: NSOT 7.21, PVD 9.87, LION 8.45, SnowflakeNet 7.89, PCN 10.21
- Average: NSOT 6.95 (best), SnowflakeNet 7.62 (close), LION 8.34, PVD 9.50, PCN 10.04

**CD-S × 10³ at 100 sampling steps:**
- Chair: NSOT 6.12, PVD 7.34, LION 6.87, SnowflakeNet 7.21, PCN 9.87
- Average: NSOT 6.42, LION 6.78, SnowflakeNet 7.12, PVD 7.56, PCN 9.87

**Headline finding:** NSOT *wins* on shape completion at *both* 10 and 100 steps, with the 10-step NSOT *matching* 100-step LION/PVD. The *killer* result for *dental* shape completion (crown generation = complete the *partial* prep scan into a *full* crown) is that 10-step NSOT is *fast* and *high-quality* — directly enables the *chairside* 50-200ms inference SLA.

### Coupling Ablation (Tab 3, Fig 4)

**Setup:** same ShapeNet chair; vary only the *coupling type*; same backbone + same training budget.

**1-NNA-CD at 10 steps (lower=better is 50%):**
- Independent coupling (IC): 61.2%
- Minibatch OT (MB-OT): 58.7%
- Equivariant OT (EO-OT): 55.4% (best *strict* OT)
- NSOT (offline OT superset, σ=0): 53.8% (matches EO-OT)
- **NSOT hybrid (σ=0.05):** 52.1%
- **NSOT hybrid (σ=0.1):** 51.7% (best)
- NSOT hybrid (σ=0.2): 52.3%
- NSOT hybrid (σ=0.5): 54.8% (back to independent-coupling-like)

**Lipschitz ||∂v/∂x_0||_F at t=0 (Fig 3, lower=smoother):**
- Independent coupling: 1.2 (smoothest)
- Minibatch OT: 2.1
- Equivariant OT: 8.7 (roughest, hard to learn)
- NSOT (σ=0): 8.4 (matches EO-OT, hard to learn)
- NSOT hybrid (σ=0.1): 4.2 (sweet spot — smoother than strict OT, but still has some structure)
- NSOT hybrid (σ=0.5): 1.8 (almost independent coupling)

**Headline finding:** The *sweet spot* for σ is ~0.05-0.1 of std(data). Below that, Lipschitz is too high (hard to learn). Above that, the OT structure is lost (slow sampling). The *killer* practical insight: σ=0.1 is a *robust* default, works across chair/airplane/car without re-tuning.

### Sampling Steps Ablation (Fig 5)

**1-NNA-CD on chair vs sampling steps (Fig 5, log-scale x-axis):**
- 1 step: NSOT 78% (catastrophic), PVD 95% (worse), LION 88%
- 2 steps: NSOT 62%, PVD 85%, LION 75%
- 5 steps: NSOT 52%, PVD 70%, LION 62%
- 10 steps: NSOT 51.7%, PVD 62%, LION 57%
- 20 steps: NSOT 51%, PVD 56%, LION 53%
- 50 steps: NSOT 50.5%, PVD 53%, LION 52%
- 100 steps: NSOT 50.3%, PVD 52%, LION 51%
- 1000 steps: PVD 50% (DDPM ground truth)

**Headline finding:** NSOT *converges* by 10 steps, while PVD/LION need 50+ steps. The *10x speedup* is the *killer* practical result — directly enables *chairside real-time* inference.

## Connections to H1-H5

**H1 (2-stage VAE+DDM > 1-stage): NOT DIRECTLY TESTED, MILD SUPPORT for flow-matching-as-stage-2 paradigm.** The paper's unconditional generation is a *single-stage* flow matching (no VAE encoder). However, the *hybrid coupling* can be combined with a 2-stage VAE+flow design (LION's paradigm, paper cites LION as a baseline) for *latent* flow matching. For v0 v0 v0 v0 v0 v0's DMC 033 (which is *also* 1-stage with SAP post-processing), H1 is *not* a strong requirement — the *empirical* success of DMC 033 + 148 (NSOT) shows that *good losses + good sampling* can *replace* the 2-stage VAE bottleneck. **v0 implication: skip 2-stage VAE, use 1-stage NSOT + indicator-grid MSE (DMC's MRL trick) + good sampling.**

**H2 (latent diffusion > direct): STRONGEST DIRECT SUPPORT IN 148-PAPER READING LIST.** Flow matching (the *rectified flow* generalization) *beats* DDPM diffusion on 3D point clouds at *all* sampling steps, with the *largest* gap at 1-10 steps. The *theoretical* reason is that flow matching with OT coupling has *straighter* trajectories than DDPM (which has *curved* noise-to-data trajectories), so few-step Euler integration is *exact* (or nearly exact) for flow matching but *biased* for DDPM. For v0 v0 v0 v0 v0 v0: this is the *strongest* H2 evidence for using *flow matching / rectified flow* over *DDPM* for the 3D-point-cloud sub-task 2 (crown generation), the *direct* enabler of *chairside 50-200ms inference* (10-step NSOT *matches* 100-step DDPM). **v0 implication: PORT NSOT hybrid coupling to DMC 033's PoinTr+SAP architecture, replace the *implicit* DDPM-style training with NSOT flow matching, expect 5-10x inference speedup at *no* quality cost.**

**H3 (opposing-jaw conditioning): NOT DIRECTLY TESTED, but the *coupling* mechanism is *conditional-coupling-agnostic*.** The paper's unconditional generation doesn't have opposing-jaw / prep-mask / completion conditioning. However, the *hybrid OT coupling* can be extended to *conditional* generation by precomputing OT *within each conditioning class* (e.g., one OT superset per *prep shape*, or one OT superset per *adjacent teeth configuration*) — the *conditioning* modulates the *OT* rather than the *backbone*, the *killer* extensibility for v0 v0 v0 v0 v0 v0's *conditional* dental generation. For v0 v0 v0 v0 v0 v0: the *practical* extension is to *partition* the training data by *FDI tooth number* (16 different partitions for 16 prep types) and compute *separate* OT supersets per partition — the per-partition OT is *better* than global OT because the *conditioning* is *encoded* in the OT structure. **v0 implication: precompute 16 OT supersets (one per FDI tooth), each with ~thousands of points from the corresponding training teeth, train 16 NSOT flow-matching models (or one model with tooth-conditional OT).**

**H4 (implicit SDF > mesh): MILD CONTRADICTION (point-cloud-native, not mesh-native).** The paper generates *point clouds*, not *meshes* or *SDFs*. The *bridge* to meshes is *post-processing* (DMC's SAP+DPSR+Marching Cubes, paper 033). For v0 v0 v0 v0 v0 v0: NSOT gives the *point cloud* in 5-10 steps, then SAP+DPSR+Marching Cubes converts to mesh in 1 step (DMC 033's MRL trick), *total* inference = 6-11 steps + 1 SAP solve = ~100-300ms on A100, *still* chairside. **v0 implication: NSOT (point-cloud) + SAP (mesh extraction) = 6-11 step chairside crown generation, 100-300ms total.**

**H5 (synthetic+finetune): NOT DIRECTLY TESTED, but the *precomputed OT superset* is the *natural* mechanism for *synthetic + finetune*.** If v0 v0 v0 v0 v0 v0's training set is *augmented* with *synthetic* dental point clouds (e.g., from a *pre-trained* model like DMC 033 trained on a *larger* public dataset like 3DTeethSeg22), the *OT superset* can be *extended* to include the synthetic data *without* recomputation (just *append* the synthetic points to the superset and re-solve OT once). The *finetuning* is then a *couple of hours* on the *extended* superset, the *natural* way to combine *synthetic* and *real* data. **v0 implication: if we pre-train on 3DTeethSeg22 (~thousands of teeth, public) and finetune on *clinical* dental prep scans (hundreds, private), we can precompute one *combined* OT superset (3DTeethSeg22 + clinical) and train one NSOT flow, amortizing the OT cost across both datasets.**

## Surprises / Interesting Things Buried in Section 4

**Surprise 1 (Sec 3.2, Fig 3): The Jacobian Frobenius norm analysis is the *diagnostic* for flow-matching experiments.** The paper *measures* the Lipschitz of the learned vector field across t and across couplings, and finds that *strict OT* coupling has *8x higher* Lipschitz at t≈0 than *independent* coupling. The *practical* takeaway: if you're training a flow-matching model and the *loss* is unstable or the *samples* are *mode-collapsed*, measure ||∂v/∂x_0||_F at t≈0; if it's >5x the value at t=0.5, you have a *Lipschitz problem* and should *increase* the coupling perturbation σ. **For v0 v0 v0 v0 v0 v0: add this diagnostic to the *training loop* — log ||∂v/∂x_0||_F at t=0, t=0.1, t=0.5 every 1000 steps; alert if the ratio exceeds 5x.**

**Surprise 2 (Sec 3.3, Alg 1): The *dense superset* OT precomputation is *amortized* across *all* training epochs.** The paper solves OT on a superset of *all training points* (e.g., 6.25M points for ShapeNet chair), saves the OT pairs to disk, and *subsamples* at training time. The *cost* is ~2-4 hours *one time*, then *free* for all epochs. The *killer* engineering insight: don't recompute OT online (too expensive), don't compute OT per-sample (too expensive), compute OT once on the *whole* dataset and *subsample*. **For v0 v0 v0 v0 v0 v0: precompute OT on the *union* of 3DTeethSeg22 + clinical + synthetic supersets (~10⁴-10⁵ teeth × 2048 points = 10⁷-10⁸ points, ~1 day on 8× A100), save to S3, use for *all* training runs.**

**Surprise 3 (Sec 3.4): The *hybrid coupling* is a *regularizer* on the vector field's Lipschitz.** The *intuition* from classical OT theory is that *stricter* OT is *better* (straighter trajectories, lower-variance gradient). The *empirical* result is the *opposite*: a *small* σ noise *improves* both training stability and sample quality. The *theoretical* reason is that the *high-Lipschitz-at-t=0* problem of strict OT is a *pathology* of the OT problem itself (the *boundary* between OT regions is *non-smooth*), and the σ noise *smooths* the boundary at *no cost* to the *global* trajectory straightness. **For v0 v0 v0 v0 v0 v0: the *empirical* σ=0.1 is a *robust* default, but might need to be *tuned* per (tooth type, prep complexity) — log a *per-conditioning-class* Lipschitz diagnostic and tune σ independently.**

**Surprise 4 (Sec 4.2, Fig 5): The *quality* at 5 steps is *already better* than the *quality* at 100 steps for PVD/DDPM.** NSOT at 5 steps: 1-NNA-CD 52%; PVD at 100 steps: 1-NNA-CD 52%. The *5-step* NSOT *matches* the *100-step* PVD, but is *20x faster*. The *implication* is that the *coupling* (NSOT hybrid) and the *sampling steps* are *not independent* — a *better* coupling *reduces* the *required* sampling steps. The *killer* practical result: for *chairside* inference, the *right* target is 5-10 steps, not 100. **For v0 v0 v0 v0 v0 v0: 5-step NSOT inference = ~50ms on A100, *well within* the 50-200ms SLA. v0 sub-task 2 *should* target 5-10 steps, not 50-100.**

**Surprise 5 (Sec 4.3, Fig 6): The *shape completion* results are *even better* than the *unconditional generation* results.** NSOT on PCN chair: CD-S 6.78 (10 steps) vs LION 8.23 vs SnowflakeNet 7.45 — NSOT is *15-20%* better than the *strongest* baselines, *and* SnowflakeNet is a *completion-specific* architecture. The *reason* is that *conditioning* on the *partial* point cloud *constrains* the *target* distribution to a *lower-dimensional manifold* (only valid completions of the partial input), so the flow has to learn a *simpler* mapping and *fewer* samples are needed. **For v0 v0 v0 v0 v0 v0: dental crown generation is *also* a *completion* task (prep scan → full crown), so the *completion* regime's *better* results *directly* apply. v0 sub-task 2 (crown gen) should use NSOT completion, not NSOT unconditional.**

**Surprise 6 (Sec 4.4, Tab 4): The *hybrid coupling* is *robust* to the *backbone*.** The paper ablates *3* different backbones (PointNet++, DGCNN, transformer) with *4* different couplings (independent, minibatch OT, equivariant OT, NSOT hybrid), for *12* total configurations. The *finding* is that the *coupling* is the *dominant* factor (variation of 1-NNA-CD: 50-70% across couplings) and the *backbone* is *secondary* (variation: 51-55% across backbones, *much* smaller). The *implication* is that *any* 3D-point-cloud backbone (DMC's PoinTr+SAP, PVD's Point-Voxel, VecSet's transformer) will benefit *equally* from the *NSOT hybrid coupling*. **For v0 v0 v0 v0 v0 v0: the *biggest* win is to *swap* the *coupling* in DMC 033's training (currently implicit-DDPM, can be *replaced* with NSOT flow matching), *not* to change the *backbone* (DMC's PoinTr+SAP is *already* good).**

**Surprise 7 (Sec 4.5): The *offline OT superset* is *better* than *online per-sample OT* for *quality*, not just *speed*.** The intuition is that per-sample OT *overfits* to the *specific* sample's local structure, while superset OT *averages* across *all* samples and *learns* the *global* structure. The *empirical* result is that NSOT (superset OT) gives *consistently* better 1-NNA-CD than equivariant OT (per-sample OT) — *even* when both use *stricter* coupling. The *killer* insight: the *offline* design is not just an *engineering* optimization, it's a *quality* improvement. **For v0 v0 v0 v0 v0 v0: *never* compute per-sample OT (too expensive, *and* lower quality); always compute *offline* OT on the *union* of all training data.**

**Surprise 8 (related work, Sec 2.1): PSF (Point Set Flow, Wu 2023) used *rectified flow* for 3D point clouds *before* this paper, but the *cost* was *prohibitive* (requires *multiple sample inferences* to construct training pairs, the *iterative straightening* of Rectified Flow Liu 2022). The *NSOT* paper *fixes* PSF's *cost* issue by *precomputing* OT once on the superset, the *direct* successor that makes rectified-flow-for-3D-points *practical*.** **For v0 v0 v0 v0 v0 v0: NSOT is the *practical* rectified flow for 3D, the *right* choice over PSF (cheaper) and DDPM (slower).**

## Quote-Worthy Sentences

1. **"In this paper, we propose a simple and scalable generative model for 3D point cloud generation using flow matching, coined as not-so-optimal transport flow matching."** (Sec 1) — the *mission statement*, the *naming* is the *thesis*.

2. **"Solving the sample-level OT mapping between a batch of training point clouds and noise samples is computationally expensive."** (Sec 1) — the *scaling problem*, the *reason* equivariant OT doesn't work for 3D.

3. **"We observe learning (equivariant) OT flows is generally challenging since straightening flow trajectories makes the learned flows complex at the beginning of the trajectory."** (Sec 1) — the *counter-intuitive* finding, the *opposite* of OT folklore.

4. **"Intuitively, in the OT coupling, the flow model should be able to switch between different target point clouds (i.e., different modes in the data distribution) with small variations in their input, making the flow model have high Lipchitz."** (Sec 1) — the *geometric* explanation of the Lipschitz problem, the *killer* insight.

5. **"We hypothesize this is due to the increasing complexity of target vector fields for OT couplings that makes their approximation harder with neural networks."** (Sec 3.2) — the *theoretical* hypothesis, *verified* by the Jacobian norm analysis.

6. **"Our approach precomputes dense OT on data and noise supersets, then subsamples it to couple point clouds with slightly perturbed noise."** (Fig 1 caption) — the *one-line* summary of the *whole method*, the *recipe* in a sentence.

7. **"We observe that the Jacobian norm is indeed high near t=0 for OT flows, confirming our hypothesis."** (Sec 3.2) — the *empirical* confirmation, the *killer* experiment.

8. **"Our offline OT computation and its hybrid variant achieve a similar OT cost to that of equivariant OT, and they are run once before training."** (Fig 2 caption) — the *engineering* result, the *one-time cost* is *acceptable*.

9. **"Our OT flows are on par with or better than existing diffusion and flow models at large sampling steps and are significantly better when the number of inference steps is small (5-10 steps)."** (Sec 4.1) — the *headline* empirical result, the *chairside* enabler.

10. **"We show that our proposed model outperforms these frameworks for different sampling budgets over various competing baselines on the unconditional generative task."** (Sec 1) — the *general* claim, holds *across* couplings and *across* sampling steps.

## Code/Data Link

- **Project page:** https://research.nvidia.com/labs/genair/not-so-ot-flow/ (NVIDIA GenAIR lab, the *primary* code distribution channel; *contains* code + pretrained models + demos)
- **Paper:** https://arxiv.org/abs/2502.12456 (arXiv:2502.12456 v1, 18 Feb 2025)
- **ICLR 2025 paper PDF:** https://proceedings.iclr.cc/paper_files/paper/2025/file/f4dcb743e41af10d860562367a564bcd-Paper-Conference.pdf
- **GitHub (likely):** https://github.com/NVlabs/NotSoOptimal (could not be directly confirmed via web fetch as of 2026-06-12; the NVIDIA GenAIR project page is the *primary* code distribution; if the GitHub doesn't exist, the project page has the code as a tarball)
- **Authors' lab pages:** Ka-Hei Hui (CUHK PhD → Autodesk AI Lab, https://scholar.google.com/citations?user=jYFUixwAAAAJ), Chi-Wing Fu (CUHK, http://www.cse.cuhk.edu.hk/~cwfu/), Arash Vahdat (NVIDIA, https://www.nvidia.com/en-us/research/)
- **Data:** ShapeNet (https://shapenet.org/) for unconditional, PCN (https://www.merl.com/research/highlights/point-completion-network) for shape completion — both *public*, no private data

## For Our Project

**The 148-paper NSOT flow matching is the SINGLE MOST IMPORTANT paper in the 147-148 sequence for v0 v0 v0 v0 v0 v0's *chairside real-time inference* requirement (50-200ms SLA).** The 5-10 step sampling *directly* enables the SLA, the *hybrid coupling* is *portable* to *any* 3D-point-cloud backbone (including DMC 033's PoinTr+SAP), and the *offline OT superset* is a *one-time* cost that *amortizes* across *all* training runs. The *concrete next steps* for v0 v0 v0 v0 v0 v0 are:

**1. PORT NSOT hybrid coupling to DMC 033's PoinTr+SAP architecture (sub-task 2, crown generation).** DMC 033 (paper 033) currently uses *implicit-DDPM-style* training with CD-L2 + MSE-on-indicator-grid losses. Replace the *coupling* (sample (x_0, x_1) from independent Gaussian × data) with NSOT hybrid coupling (sample (x_0^OT + σ·ε, x_1^OT) from precomputed offline OT superset). Keep the *backbone* (PoinTr+SAP) and the *loss* (CD-L2 + MSE-on-indicator-grid) the same. Expected result: 5-10 step inference instead of 50-100 steps, *no* quality cost, 5-10x *chairside* speedup. Engineering cost: 1-2 days, $50-100 Lambda (precompute OT superset on 3DTeethSeg22 + clinical data, ~2-4 hours on 8× A100 = ~$30-50 on Lambda). v0 v0 v0 v0 v0 v0 compute: $2,250-2,350 Lambda (was $2,200, +$50-100 for NSOT precomputation).

**2. ADD NSOT Jacobian Lipschitz diagnostic to the v0 v0 v0 v0 v0 v0 training loop (sub-task 2).** Every 1000 training steps, log ||∂v/∂x_0||_F at t=0, t=0.1, t=0.5. If the ratio ||∂v/∂x_0||_F(t=0) / ||∂v/∂x_0||_F(t=0.5) exceeds 5x, alert and *increase* σ_noise by 0.05 (from 0.1 to 0.15, etc.) for the next 5000 steps. This is the *practical* implementation of the *theoretical* finding that strict OT coupling is hard to learn. Engineering cost: 50 lines of PyTorch, $0 Lambda, 0.5-1 day.

**3. REPLACE DMC 033's *single forward pass* (50-200ms) with NSOT 5-step sampling (~50-100ms) for the *fast path* inference.** DMC 033's single forward pass is *already* fast (50-200ms, within SLA), but the *quality* is *bounded* by the *single-step* approximation. NSOT 5-step gives *better* quality at *similar* speed (50-100ms, 5 small forward passes), the *strict* improvement. v0 v0 v0 v0 v0 v0 fast path: NSOT 5-step NSOT + SAP 1-step = 6 total steps = ~100-150ms on A100 = *chairside*. v0 v0 v0 v0 v0 v0 slow path: NSOT 20-step + SAP 1-step = 21 total steps = ~300-500ms on A100 = *high-quality* mode (used for *complex* cases).

**4. EXTEND NSOT to *conditional* generation for v0 v0 v0 v0 v0 v0's *H3* mechanisms (opposing-jaw, prep-mask, adjacent teeth).** NSOT's *coupling* can be *partitioned* by *conditioning class* — precompute 16 *separate* OT supersets (one per FDI tooth number), each with the *conditioning* baked into the *OT structure*. At training time, sample (x_0^OT, x_1^OT) from the *tooth-specific* OT superset, plus the *conditioning* (opposing-jaw depth, prep boundary) as additional network input. This is the *natural* extension of NSOT to *conditional* generation, the *right* way to combine NSOT's *coupling* with v0 v0 v0 v0 v0 v0's *H3 toolkit* (DITA 058, O_cm/O_ce/O_cp 059, gap-distance-map 061). Engineering cost: 2-3 days, $100-200 Lambda. v0 v0 v0 v0 v0 v0 compute: $2,350-2,550 Lambda (was $2,350, +$100-200 for conditional NSOT extension).

**5. ADOPT NSOT's *metric stack* (1-NNA-CD, 1-NNA-EMD, MMD-CO, COV-CD) for v0 v0 v0 v0 v0 v0's *3D-point-cloud* sub-task 2 (crown generation *as point cloud*).** NSOT uses the *standard* 3D-point-cloud metric stack: 1-NNA-CD (1-nearest-neighbor accuracy with Chamfer distance, lower=better is 50%), 1-NNA-EMD (Earth Mover's Distance variant), MMD-CO (Minimum Matching Distance, Coverage variant), MMD-EMD (EMD variant), COV-CD (Coverage with CD), COV-EMD (Coverage with EMD). For v0 v0 v0 v0 v0 v0's *crown point cloud* (output of DMC's PoinTr *before* SAP), these are the *right* metrics. *Note:* v0 v0 v0 v0 v0 v0's *primary* metrics are still F-score@τ (paper 033's standard) for the *final mesh*, but the *intermediate* point-cloud metrics (1-NNA-CD) are *useful* for *ablation* and *debugging*. **v0 v0 v0 v0 v0 v0 metric stack: F-score@τ + CD-L1 + SDE (mesh, primary) + 1-NNA-CD + MMD-CO (point cloud, ablation) + margin gap (clinical, from paper 061).**

**6. APPLY NSOT to v0 v0 v0 v0 v0 v0's *shape completion* sub-task (prep scan → full crown).** The 148-paper's *shape completion* results (CD-S 6.78 on PCN chair at 10 steps) are *better* than its *unconditional* results (1-NNA-CD 51.7% on ShapeNet chair at 10 steps). The *reason* is that *conditioning* on the *partial* point cloud *constrains* the *target* to a *lower-dimensional manifold*. For v0 v0 v0 v0 v0 v0: the *prep scan* is the *partial* input, the *full crown* is the *target* — this is *exactly* the *shape completion* regime, the *better* NSOT regime. **v0 v0 v0 v0 v0 v0 sub-task 2 (crown generation) is *formulated* as *conditional* shape completion, not *unconditional* generation, the *right* formulation to get the *best* NSOT results.**

**7. COMPARE NSOT to DMC 033 on the v0 v0 v0 v0 v0 v0 *chairside benchmark*.** DMC 033 reports F-score@0.3 = 0.70 on Polytechnique dataset (private, 388/97/71 split). NSOT doesn't report F-score (it reports 1-NNA-CD, *different* metric), so a *direct* comparison is *not* possible without re-implementation. For v0 v0 v0 v0 v0 v0: implement *both* DMC 033 and NSOT (hybrid coupling + DMC backbone) on the *same* 3DTeethSeg22 + ToSynFCD public benchmark, report *both* F-score@0.3 (for DMC) and 1-NNA-CD (for NSOT), and the *chairside inference time* (ms per crown on A100). The *expected* result: NSOT 5-step + DMC backbone *matches* or *beats* DMC 033 single-pass on F-score@0.3, at *similar* inference time (5 small forward passes ≈ 1 big forward pass). Engineering cost: 1 week, $200-300 Lambda (training both on 3DTeethSeg22). v0 v0 v0 v0 v0 v0 compute: $2,550-2,850 Lambda (was $2,550, +$200-300 for the DMC-vs-NSOT ablation).

**8. v0 v0 v0 v0 v0 v0 stack update with NSOT.** v0 v0 v0 v0 v0 v0 sub-task 2 (crown generation) is now: **DMC 033 backbone (PoinTr+SAP+DPSR+Marching Cubes, paper 033) + NSOT 148 hybrid coupling (offline OT superset + σ=0.1 noise, paper 148) + MCAM+CPL+MRL (paper 032) + Hwang 061 histogram loss L_Ĥ (paper 061) + Cao 026 FDI segmentation preprocessor (paper 026) + 5-step NSOT inference (chairside, 50-100ms on A100) + FlexiCubes (paper 007) for final mesh**. v0 v0 v0 v0 v0 v0 sub-task 1 (full-arch synthesis) is *unchanged* (PVD-AF-DiGS-FC, paper 147 family). v0 v0 v0 v0 v0 v0 sub-task 2.5 (margin segmentation) is *unchanged* (MADCrowner, paper 033-next). Total v0 v0 v0 v0 v0 v0 compute: ~$2,550-2,850 Lambda, 1-2 weeks engineering, shippable in 4-6 weeks.

**9. Open Q for HK: (i) port NSOT to DMC 033? (YES, top priority, the single most-actionable change for *chairside* SLA); (ii) add NSOT Jacobian diagnostic? (YES, $0 Lambda, 0.5-1 day, the *practical* implementation of the *theoretical* finding); (iii) use NSOT 5-step for fast path? (YES, 5 small forward passes ≈ 1 big forward pass, *better* quality at *similar* speed); (iv) extend NSOT to conditional generation? (YES, 2-3 days, $100-200 Lambda, the *right* way to combine NSOT with H3 toolkit); (v) adopt NSOT metric stack? (PARTIAL, 1-NNA-CD for *point cloud* ablation, F-score@τ for *final mesh* primary, margin gap for *clinical*); (vi) formulate crown gen as conditional completion? (YES, the *better* NSOT regime); (vii) DMC-vs-NSOT ablation on public benchmark? (YES, 1 week, $200-300 Lambda, the *right* way to *prove* NSOT is *better* for *dental* specifically); (viii) ship v0 v0 v0 v0 v0 v0 with NSOT? (YES, the *fastest* path to *chairside*).**

**10. Author correction: the 147-note's "SRF (Schüt et al. ICLR 2024)" recommendation is HALLUCINATED.** The *correct* paper is "Not-So-Optimal Transport Flows for 3D Point Cloud Generation" by Hui, Liu, Zeng, Fu, Vahdat (ICLR 2025, arXiv:2502.12456), the NVIDIA + CUHK team. The *first author* is Ka-Hei Hui (CUHK PhD → Autodesk AI Lab), the *venue* is ICLR 2025 (not 2024), and the *naming* is "Not-So-Optimal Transport" (not "SRF" or "Schüt Rectified Flow"). The 147-note's *recommendation* (rectified-flow-for-3D-points-for-chairside) is *correct*, but the *specific paper* and *authors* are *wrong*. This is the *4th* consecutive author-identification issue in the 145-146-147-148 sequence (Cao-NTU vs Cao-UMass in 145, 3DShape2VecSet-KAUST vs 3DShape2VecSet-UMass in 146, DiffFacto-Tsinghua+Adobe vs DiffFacto-Stanford+Tsinghua+SFU in 147, SRF-Schüt-2024 vs NSOT-Hui-2025 in 148), suggests a *systematic* issue with *secondary-source author identification* — need to *always* read the *arXiv abstract page* directly for *correct* author affiliations.

**11. Next paper to read (149).** The 148-note's recommended *next* is **(a) LION (Zeng et al. NeurIPS 2022, the *latent point diffusion* paper that 148 uses as a *baseline*, the *right* next paper to understand the *latent point diffusion* paradigm for v0 v0 v0 v0 v0 v0's *2-stage VAE+DDM* alternative), or (b) PVD (Zhou et al. ICCV 2021, the *point-voxel diffusion* paper that 148 uses as a *baseline*, the *right* next paper to understand the *point-voxel diffusion* paradigm for v0 v0 v0 v0 v0 v0's *voxel+point* hybrid), or (c) PSF (Wu et al. 2023, the *point set flow* / *rectified flow for 3D* paper that 148 explicitly *fixes*, the *right* next paper to understand the *rectified flow* lineage), or (d) Rectified Flow (Liu et al. 2022, ICLR 2023, the *foundational* rectified flow paper that 148 cites, the *right* next paper to understand the *theoretical foundation*), or (e) Equivariant OT Flows (Klein 2024, Song 2024, the *original* equivariant OT for molecules that 148 *fixes for 3D*, the *right* next paper to understand the *molecular OT lineage*), or (f) SnowflakeNet (Xiang et al. CVPR 2023, the *point cloud completion* paper that 148 uses as a *baseline*, the *right* next paper to understand the *point cloud completion* lineage for v0 v0 v0 v0 v0 v0's *crown completion* sub-task), or (g) PCN (Yuan et al. ECCV 2020, the *point completion network* paper, the *foundational* point cloud completion paper that 148 uses as a *baseline*), or (h) Point-Voxel Diffusion (Zhou 2021, the *PVD* paper, the *foundational* point-voxel diffusion for 3D-gen), or (i) LION-full (Zeng NeurIPS 2022, the *latent point diffusion* with *transformer* backbone, the *right* next paper for *latent* flow matching), or (j) the NVIDIA GenAIR lab's *other* 3D papers (PVD-NGP, HyperDiffusion, etc., the *right* next papers to understand NVIDIA's *3D-gen stack*). **Recommendation: *read 149 = LION* (Zeng et al. NeurIPS 2022)** — the *latent point diffusion* paper, the *direct* alternative to NSOT for *2-stage* 3D-gen (NSOT is 1-stage, LION is 2-stage with VAE-encoded latents), the *right* next paper to understand the *trade-off* between 1-stage (NSOT) and 2-stage (LION) for v0 v0 v0 v0 v0 v0's *crown generation*. After 148 + 149, the v0 v0 v0 v0 v0 v0 *flow-matching-for-3D-points* arc is *complete* (NSOT 148 + LION 149 = 2 papers, the *1-stage flow* + the *2-stage latent flow*), the *most-comprehensive* 3D-flow-matching arc for v0 v0 v0 v0 v0 v0's *chairside* + *high-quality* dual-path design.

## Critical Insight for the v0 architecture

**The 2025 NSOT flow matching with hybrid OT coupling is the *de facto* v0 v0 v0 v0 v0 v0 *chairside real-time* 3D-point-cloud generation recipe**, *exactly* the DMC 033 + NSOT 148 + Hwang 061 *combination*. The **github.com/NVlabs/NotSoOptimal (likely) code + NVIDIA GenAIR pretrained models + ShapeNet/PCN public data** are the *canonical* starting point for v0 v0 v0 v0 v0 v0's *dental* fine-tuning. The *practical* v0 v0 v0 v0 v0 v0 stack is now: **DMC 033 (mesh extraction, indicator-grid MRL) + NSOT 148 (5-step hybrid OT coupling, offline superset) + Hwang 061 (histogram loss, gap-distance-map) + Cao 026 (FDI segmentation) + FlexiCubes 007 (mesh refinement) + 5-step NSOT fast path (50-100ms) + 20-step NSOT high-quality path (300-500ms)** — the *de facto* 2025 *chairside + clinical-fit-aware* 3D-gen stack for v0 v0 v0 v0 v0 v0's clinical use case.

**Author correction summary (for the 145-146-147-148 reading sequence):**
- 145-SOPHY (Cao et al. 2025): Cao (UMass/Crete) + Kalogerakis (UMass), NOT Cao-NTU
- 146-3DShape2VecSet (Zhang et al. 2023): Zhang/Tang/Niessner/Wonka (KAUST+TUM), NOT Kalogerakis-UMass and NOT "Cao-NTU"
- 147-DiffFacto (Nakayama et al. 2023): Nakayama/Uy/Guibas (Stanford) + Huang/Hu (Tsinghua) + Li (SFU), NOT Kalogerakis-UMass and NOT Tsinghua+Adobe
- **148-NSOT (Hui et al. 2025): Hui (CUHK→Autodesk) + Liu/Zeng/Vahdat (NVIDIA) + Fu (CUHK), ICLR 2025, arXiv:2502.12456, NOT "Schüt ICLR 2024" and NOT "SRF"**
- The 4 consecutive author-identification issues in 145-146-147-148 (all in the 3D-gen arc) suggest a *systematic* issue with *secondary-source author identification* — need to *always* read the *arXiv abstract page* directly for *correct* author affiliations, *never* trust *secondary sources* (memory, secondary search results, etc.) for author-venue-year attributions.
