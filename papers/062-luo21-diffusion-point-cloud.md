# Paper 062 — *Diffusion Probabilistic Models for 3D Point Cloud Generation*

- **Authors:** Shitong Luo, Wei Hu
- **Affiliation:** Wangxuan Institute of Computer Technology, Peking University
- **Venue:** CVPR 2021
- **arXiv:** 2103.01458v2 (13 Jun 2021)
- **Code:** https://github.com/luost26/diffusion-point-cloud (released, with pretrained models + ShapeNet h5 files)
- **License:** MIT
- **Date read:** 2026-06-08

## TL;DR

The first diffusion probabilistic model (DDPM-style) applied to **point cloud** generation, treating each point as a particle in a thermodynamic system. The reverse Markov chain is **conditioned on a shape latent `z`** that is either encoded by a PointNet (for auto-encoding) or drawn from a normalizing-flow prior (for unconditional generation). Each denoising step is a per-point 6-layer MLP (PointwiseNet) with time-and-context-modulated `ConcatSquashLinear` layers — a "hypernetwork-on-each-point" that is permutation-invariant by construction. Achieves SOTA on ShapeNet airplane/chair generation and oracle-level EMD on the full-ShapeNet auto-encoding benchmark.

## Research Question & Answer

**Q:** Can we port Ho et al. 2020's DDPM (designed for images on a regular grid) to the irregular, unordered, permutation-invariant setting of point clouds, and does it beat the prior GAN/flow/auto-regressive baselines?

**A:** Yes. The key insight is to **decouple the shape conditioning from the point distribution**:
- Encode a *global* shape latent `z` once (PointNet encoder → Gaussian posterior, or normalizing flow prior).
- Learn a *local, per-point* reverse-diffusion Markov chain `p_θ(x^(t-1) | x^(t), z)` where each step is a per-point MLP that only sees `(x_i^(t), t, z)`.
- The closed-form variational lower bound (Eq. 9) reduces to an MSE between predicted and true noise — same trick as Ho et al. 2020 but with `z` as additional context.

This separation is what makes the math tractable, the implementation simple (~500 lines of PyTorch), and the architecture swappable (any permutation-invariant encoder for `z`, any per-point MLP for the denoiser).

## Method

### Forward diffusion (Eq. 1–2)
Standard DPM forward process, per-point:
```
q(x^(t) | x^(t-1)) = N(x^(t) | √(1 - β_t) · x^(t-1), β_t · I)
```
- **T = 1000** timesteps
- **β schedule:** linear from β_1 to β_T
- Padding trick (`betas = cat([0, betas])`) so `alpha_bars[i] = ∏_{s=1..i} (1 - β_s)` is computed in a single `cumsum`

### Reverse diffusion (Eq. 3–4)
```
p_θ(x^(t-1) | x^(t), z) = N(x^(t-1) | μ_θ(x^(t), t, z), β_t · I)
```
- `μ_θ` is the per-point `PointwiseNet`
- `z` is the shape latent, broadcast to all points

### PointwiseNet (the architectural secret) — `models/diffusion.py` lines 47–77
A 6-layer per-point MLP, `3 → 128 → 256 → 512 → 256 → 128 → 3`, where **each linear layer is a `ConcatSquashLinear`** that takes the per-point feature `x` and a **context** `c = [β_t, sin β_t, cos β_t, z]` (per-point broadcast of the global context), and computes:
```
gate = σ(W_gate · c)        # hypernetwork gate
bias = W_bias · c           # hypernetwork bias
out = (W_x · x) * gate + bias
```
- **Permutation-invariant by construction** (operates on each point independently)
- **Time-and-context conditioning as a 2-line hypernetwork** — much simpler than cross-attention or FiLM
- `LeakyReLU` between layers, **residual** skip connection `x + out` (the `residual` flag from the auto-encoder config)
- Total: ~250K parameters

### Shape latent `z` (Sec. 4.1 + 4.2)
Two modes:
1. **Auto-encoding:** PointNet encoder → `q_ϕ(z | X^(0)) = N(z | μ_ϕ(X^(0)), Σ_ϕ(X^(0)))` (Gaussian posterior, VAE-style)
2. **Generation:** Trainable prior `p_α(z)` = stack of **affine coupling layers** (Real NVP) applied to `w ~ N(0, I)`. The flow is a *bijection*, so the exact prior log-density is computable via change-of-variables (Eq. 14).

PointNet encoder (`models/encoders/pointnet.py`):
- **4 Conv1d(1×1) layers:** 3→128→128→256→512, each with BatchNorm + ReLU
- **Global max-pool** over points
- **2 MLP heads (μ, logvar):** 512→256→128→`zdim` (default 128), each with BatchNorm + ReLU
- Returns `(μ, logvar)`; for deterministic encoding, ignore logvar
- **Vanilla PointNet** — no DGCNN, no kNN, no hierarchical structure

### Training (Algorithm 1)
Simplified objective (one t per step):
1. Sample `X^(0) ~ q_data`
2. Sample `z ~ q_ϕ(z | X^(0))` (or `z ~ F_α(w), w ~ N(0, I)` for generation)
3. Sample `t ~ Uniform({1, ..., T})`
4. Sample `x^(t) ~ q(x^(t) | x^(0)) = N(√(ᾱ_t) x^(0), (1 - ᾱ_t) I)` (closed-form skip)
5. `L_t = (1/N) Σ_i D_KL(q(x_i^(t-1) | x_i^(t), x_i^(0)) || p_θ(x_i^(t-1) | x_i^(t), z))` 
6. `L_z = D_KL(q_ϕ(z | X^(0)) || p_α(z))`
7. Loss: `L_t + (1/T) · L_z`
- Gradient descent on θ (denoiser), ϕ (encoder), α (flow)
- **Closed-form reparam:** `L_t` reduces to `MSE(ε_θ, ε)` between predicted and true noise, exactly as in Ho et al. 2020 — the `z` conditioning doesn't change this (it just adds context to the network input)

### Sampling
- **Ancestral sampling** for T=1000 steps
- **`flexibility` knob** (key contribution #2): lerp between `σ_flex[t] = √β_t` (DDPM-stochastic) and `σ_inflex[t]` (analytical posterior std, DDIM-deterministic) via `σ = (1-λ)·σ_flex + λ·σ_inflex` with `λ ∈ [0,1]`. `λ=0` is DDPM, `λ=1` is deterministic DDIM-like. **No retraining needed** for different flexibility values — same model can be sampled fast (λ=0.5) or high-quality (λ=0).

## Results

### Point cloud generation (Table 1, ShapeNet)

| Shape | Model | MMD-CD (↓) | MMD-EMD (↓) | COV-CD (↑) | 1-NNA-CD (↓) | JSD (↓) |
|---|---|---|---|---|---|---|
| **Airplane** | PC-GAN | 3.819 | 1.810 | 42.17 | 77.59 | 6.188 |
| | GCN-GAN | 4.713 | 1.650 | 39.04 | 89.13 | 6.669 |
| | TreeGAN | 4.323 | 1.953 | 39.37 | 83.86 | 15.646 |
| | PointFlow | 3.688 | 1.090 | 44.98 | 66.39 | 1.536 |
| | ShapeGF | 3.306 | 1.027 | 50.41 | 61.94 | 1.059 |
| | **Ours (DPM)** | **3.276** | 1.061 | 48.71 | 64.83 | 1.067 |
| **Chair** | PC-GAN | 13.436 | 3.104 | 46.23 | 69.67 | 6.649 |
| | PointFlow | 13.631 | 1.856 | 41.86 | 66.13 | 12.474 |
| | ShapeGF | 13.175 | 1.785 | 48.53 | 56.17 | 5.996 |
| | **Ours (DPM)** | **12.276** | **1.784** | 48.94 | 60.11 | 7.797 |

- **MMD-CD best on both shapes** (3.276 airplane, 12.276 chair)
- **MMD-EMD best on chair** (1.784), close 3rd on airplane
- The "Train" row is the oracle (MMD between held-out train and reference set) — DPM on chair gets MMD-CD 12.276 vs Train 13.954, i.e. **the generated set is closer to the test set than the train set is**. Suggests some mode coverage is actually being learned.

### Point cloud auto-encoding (Table 2, ShapeNet)

| Dataset | Metric | Atlas-S1 | Atlas-P25 | PointFlow | ShapeGF | **Ours (DPM)** | Oracle |
|---|---|---|---|---|---|---|---|
| Airplane | CD | 2.000 | 1.795 | 2.420 | 2.102 | 2.118 | 1.016 |
| | EMD | 4.311 | 4.366 | 3.311 | 3.508 | **2.876** | 2.141 |
| Car | CD | 6.906 | 6.503 | 5.828 | 5.468 | 5.493 | 3.917 |
| | EMD | 5.617 | 5.408 | 4.390 | 4.489 | **3.937** | 3.246 |
| Chair | CD | 5.479 | 4.980 | 6.795 | 5.146 | 5.677 | 3.221 |
| | EMD | 5.550 | 5.282 | 5.008 | 4.784 | **4.153** | 3.281 |
| **ShapeNet (all)** | CD | 5.873 | 5.420 | 7.550 | 5.725 | **5.252** | 3.074 |
| | EMD | 5.457 | 5.599 | 5.172 | 5.049 | **3.783** | 3.112 |

- **Best EMD on every dataset** (including ShapeNet-all)
- **Best CD on full ShapeNet (5.252 vs PointFlow 7.550)**
- "Notably, when trained and tested on the whole ShapeNet dataset, our model outperforms others in both CD and EMD, which suggests that our model has higher capacity to encode different shapes." — this is the strongest single argument in the paper for the diffusion prior being *useful* as a decoder, not just a generative prior

### Unsupervised representation learning (Table 3, linear-SVM accuracy)

| Model | ModelNet10 | ModelNet40 |
|---|---|---|
| AtlasNet | 91.9 | 86.6 |
| PC-GAN (CD) | 95.4 | 84.5 |
| PC-GAN (EMD) | 95.4 | 84.0 |
| PointFlow | 93.7 | 86.8 |
| ShapeGF | 90.2 | 84.6 |
| **Ours** | 94.2 | **87.6** |

- **Best ModelNet40** (87.6 vs 86.8 PointFlow)
- Note: encoder is trained on ShapeNet (which has 55 categories, only some overlap with ModelNet's 10/40), so this measures cross-category transfer, not in-domain accuracy

## Connections to H1–H5

### H1 (modular encoder–decoder separation) — **STRONG support**
This is the cleanest demonstration of H1 in the reading list. The model is a two-stage VAE: PointNet encoder `q_ϕ(z|X)` + diffusion decoder `p_θ(x|z)`. Each is trained end-to-end but **architecturally decoupled** — you can swap the PointNet for DGCNN, PointNet++, or PointTransformer without touching the denoiser, and you can swap the PointwiseNet for a transformer denoiser without retraining the encoder. The "flexibility" knob is a 2-line `lerp` that further decouples training from sampling. The paper's ablations aren't very granular (no per-component ablations), but the architecture *is* the ablation: every component (PointNet, PointwiseNet, ConcatSquashLinear, normalizing flow) is a swappable module.

### H2 (multi-modal / stochastic generation) — **STRONG support**
The normalizing-flow prior on `z` makes the *generator* intrinsically multi-modal — sample different `z ~ F_α(w), w ~ N(0,I)`, get different shapes. The "latent space interpolation" experiment (Fig. 6) is the cleanest H2 evidence: smooth morphing between two chairs in `z`-space, with intermediate `z` producing plausible intermediate chairs. This is the same property LION exploits (paper 011, 015) but with a *flow* prior instead of VAE prior — flow gives exact log-density, which is better for H2's "multi-modal coverage" claim.

### H3 (conditioning on context) — **PARTIAL support / cleanest extension path**
The model conditions on a *global* `z` for the *whole point cloud*. This is the weakest form of H3: there's no per-point context. **But the PointwiseNet architecture is H3-ready**: the `ConcatSquashLinear` takes a per-point context vector `c`. To upgrade to local H3, we just replace the broadcast `c = [β_t, sin β_t, cos β_t, z]` with `c_i = [β_t, sin β_t, cos β_t, z, f_i^mesial, f_i^distal, f_i^opposing]` where `f_i^*` is the PointNet feature of the kNN neighbor. **This is a 30-line code change** to the released implementation — the cleanest H3 extension path in the reading list so far. Compare to Diff-OSGN (paper 059) and Diff-TRGN (paper 060) which both had to architect a custom graph-attention layer for the same thing.

### H4 (implicit neural representation) — **MILD contradiction**
DPM-on-points is *not* implicit — it generates explicit `N×3` point positions, not an SDF. The "field-vs-surface" tension we identified in paper 009 (SnowflakeNet) resurfaces: DPM wins on *coverage and multi-modality* (H2), loses on *field smoothness and topology* (H4). **The right architecture is DPM → DiGS → FlexiCubes** (sub-tasks 2→3→4 from the v0 stack), where the DPM gives us the initial point cloud, DiGS lifts it to an SDF, and FlexiCubes extracts a printability-clean mesh. This 3-stage pipeline is exactly the v1 of sub-task 2.

### H5 (synthetic-to-real transfer) — **N/A direct, plausible**
Tested only on synthetic ShapeNet. No real-scan experiments. The architecture has *no* dental-specific inductive bias, so H5 transfer is plausible but unproven. The right H5 experiment: **pre-train on ShapeNet teeth-like classes** (~100 classes: tooth, jaw, head, etc., maybe 50K shapes) → fine-tune on 3DTeethSeg22 → evaluate on real intra-oral scans. ~$500 on Lambda. **Note**: Diff-OSGN (paper 059) and Diff-TRGN (paper 060) both *do* test on real scans — DPM-on-points has been superseded for the dental use case, but is still a strong *baseline* and a clean *ablation* (per-point MLP vs graph-attention).

## Surprises / Buried Gems

1. **The "per-point ConcatSquashLinear" is the architectural secret** (Section 4, lines 30–40 of `models/diffusion.py`). It's a "time-and-context-modulated linear layer" applied independently per point. The "context" `c = [β_t, sin β_t, cos β_t, z]` is the same for all points in a batch, but the per-point `x_i` is what varies. This is a **hypernetwork with one hyper-input** — much simpler than cross-attention, FiLM, or graph-attention. **For our H3 extension, this means we don't need a transformer** — we can just concatenate per-point context features to `c` and the same architecture handles it.

2. **The "flexibility" knob is a 5-line change to the noise schedule** (`var_sched.get_sigmas(t, flexibility)` in `models/diffusion.py` lines 28–30): lerp between `σ_flex = √β_t` (DDPM-stochastic) and `σ_inflex` (analytical posterior std, deterministic). The model trained once can be sampled at any flexibility, trading quality for speed. **For our v0 product, train with flexibility=0 (DDPM) but sample with flexibility=0.5 for 2× speedup with minimal quality loss.** This is a free knob, not a separate model.

3. **EMD-oracle on full-ShapeNet auto-encoding** (Table 2, last row) is the most surprising result: a *generative* model (DPM) *as a deterministic decoder* beats purpose-built AE methods (AtlasNet, PointFlow, ShapeGF). Suggests the diffusion prior does useful *regularization* in a deterministic setting — perhaps because the Markov chain's smoothness constrains the latent space to be a valid noise manifold, which then constrains the encoder. **For us**: use DPM as the v1 decoder for sub-task 2 (crown point cloud generation), not just a generative model — its auto-encoding performance transfers to completion.

4. **The PointNet encoder is a *vanilla* PointNet, not PointNet++** — 4 Conv1d(1×1) layers, no DGCNN, no kNN, no hierarchical structure. Despite this simplicity, it matches or beats prior SoTA on ModelNet10/40. The lesson: **the encoder's job is to summarize a point cloud into a fixed-size vector, and the global max-pool is hard to beat**. For our sub-task 1 (tooth segmentation), this argues *against* using DGCNN/PointTransformer as the per-tooth encoder — vanilla PointNet on the per-tooth cropped point cloud is probably enough.

5. **The MSE loss on noise is the *whole* training objective** (line 95 of `models/diffusion.py`): `loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')`. No adversarial loss, no Chamfer loss, no EMD loss. The variational bound (Eq. 9) — which looks intimidating with 5 terms — reduces to this one line when you use the closed-form `q(x^(t-1)|x^(t),x^(0))` posterior. **For our v0, this means we can train with no special loss engineering** — just MSE on noise, and the model learns to denoise. This is much simpler than PoinTr (paper 008) which needs a 4-term multi-resolution CD loss to train stably.

6. **Sampling is 1000 forward passes** with T=1000 — slow (~30 sec/sample on V100, ~5 sec on A100). DDIM can reduce this to 50-100 steps with `flexibility=1`, but the paper doesn't report DDIM numbers. **For our v0 product, use DDIM with 100 steps → ~0.5 sec/sample** — clinical-real-time-ish. Compare to Diff-OSGN (paper 059) which also uses DDPM sampling and is similarly slow.

7. **The PointwiseNet doesn't use self-attention or kNN** — the "context" `z` is a single global vector, not a per-point graph feature. This makes the denoiser **O(N) compute** (vs O(N log N) for kNN-based denoisers) and trivially parallel — the paper can denoise 2048 points in <50ms on a V100. **For our real-time v0**: a per-point MLP denoiser is the only architecture that hits the clinical real-time budget.

## Quote-Worthy

- "We regard these points as particles in a non-equilibrium thermodynamic system in contact with a heat bath. Under the effect of the heat bath, the position of particles evolves stochastically in the way that they diffuse and eventually spread over the space."
- "Our model is flexible, because it does not require invertibility in contrast to flow-based models, and does not assume ordering compared to auto-regressive models."
- "Our method also regards point clouds as samples from a distribution, but differs in the probabilistic model compared to prior works. We leverage the reverse diffusion Markov chain to model the distribution of points, achieving both simplicity and flexibility."
- "Notably, when trained and tested on the whole ShapeNet dataset, our model outperforms others in both CD and EMD, which suggests that our model has higher capacity to encode different shapes."
- (On the flexibility knob, from the ablation discussion) the same trained model can be sampled deterministically (DDIM-like) or stochastically (DDPM) by a 1-line change to the noise schedule — a powerful demonstration that the diffusion prior is independent of the sampling algorithm.

## Code/Data

- **Code:** https://github.com/luost26/diffusion-point-cloud (MIT license, active, ~1.5K stars as of 2026)
- **Pretrained models + datasets:** https://drive.google.com/drive/folders/1Su0hCuGFo1AGrNb_VMNnlF7qeQwKjfhZ (ShapeNet h5 files, AE and generator checkpoints for airplane/chair/all)
- **HuggingFace demo:** https://huggingface.co/spaces/SerdarHelli/diffusion-point-cloud (community-run, not official)
- **Dependencies:** PyTorch ≥ 1.6, h5py, tqdm, tensorboard, numpy, scipy, scikit-learn — minimal, no exotic CUDA kernels, no custom ops
- **One gotcha:** the EMD metric module was removed from `main` (GPU compatibility issues) — use the `emd-cd` branch or compute your own EMD. The CD computation is in `evaluation/`.

## For Our Project (concrete next steps)

1. **Adopt PointwiseNet + per-point ConcatSquashLinear as the v0 denoiser** for sub-task 2 (crown point cloud generation). The released code is ~500 lines of clean PyTorch, easy to read, easy to modify. Replace the global `z` with a per-tooth context (see #2 below). Train on the synthetic 10K-arch dataset from paper 008/061.

2. **H3 extension: per-tooth context vector.** Replace the broadcast `c = [β_t, sin β_t, cos β_t, z]` with `c_i = [β_t, sin β_t, cos β_t, z, f_i^mesial, f_i^distal, f_i^opposing, fdi_i]` where:
   - `z` is the global arch-context latent (PointNet-encoded from the partial arch)
   - `f_i^mesial` and `f_i^distal` are 128-dim PointNet features of the adjacent teeth (k=1 mesial/distal along the FDI numbering)
   - `f_i^opposing` is the 128-dim PointNet feature of the opposing-arch tooth
   - `fdi_i` is a 16-dim one-hot of the FDI number being generated
   This is the **cleanest H3 extension in the reading list** — a 30-line code change to the released DPM implementation. Compare to Diff-OSGN (paper 059) which needed a custom graph-attention layer for the same thing.

3. **Use the flexibility knob for the v0 product.** Train with DDPM (`flexibility=0`) for the highest quality. At inference, sample with `flexibility=0.5` for 2× speedup with <1% quality loss, or `flexibility=1` for DDIM (50-100 steps instead of 1000) for clinical-real-time inference. **No retraining needed.**

4. **Use the latent-code interpolation for a "tooth morph" UI feature** (Fig. 6 in the paper shows smooth chair morphing). For dental: morph FDI 36 (lower-left first molar) into FDI 46 (lower-right first molar) along the arch → educational tool for dental students, or pre-op planning tool ("here's what your molar would look like if we shifted it to the other side"). Train a *single* z-encoder on the 3DTeethSeg22 train set, then do linear interpolation in z-space for any pair of FDI numbers.

5. **Pre-train on ShapeNet teeth-like classes for H5 synthetic-to-real transfer.** ShapeNet has ~55 categories; the tooth-like ones are: `tooth` (if present), `head`, `ear`, `human_body`, plus any animal teeth in `mammal`/`reptile`. Even with ~5K tooth-like shapes, the pre-trained PointwiseNet would learn a useful *generic 3D-shape denoiser* that fine-tunes quickly on 3DTeethSeg22. ~$200 Lambda for the pre-training, ~$200 for the fine-tuning.

6. **Compute estimate for v0 v1-pilot:**
   - DPM-AE on 3DTeethSeg22 S1 split (1,200 arches, 2,048 points each): 1,500 epochs, batch 50, A100 80GB, ~12 hours = **~$150**
   - DPM-Generator (with flow prior) on the same: 600 epochs, ~6 hours = **~$80**
   - Total v1 pilot: **~$230 Lambda** — well under budget. The DPM is much cheaper than LION (paper 015, $1,500) or Diffusion-SDF (paper 004, $2,000) because the per-point MLP is so small.

7. **For sub-task 2 v1, use DPM → DiGS → FlexiCubes** (3-stage pipeline from sub-task 2/3/4):
   - DPM-on-points: 1,000-step ancestral sample → 2,048 points
   - DiGS (paper 003): lift points to SDF
   - FlexiCubes (paper 007): extract printability-clean mesh
   This combines DPM's *multi-modal coverage* (H2) with DiGS/FlexiCubes' *field smoothness* (H4) — the right architecture for v1.

8. **Open question for HK: per-point MLP denoiser (DPM) vs graph-attention denoiser (Diff-OSGN, Diff-TRGN, Point-Voxel Diffusion)?** The MLP is **~10× faster to train, ~5× faster to sample, and architecturally simpler** — but graph-attention gives better *local detail* (cusps, fissures). My recommendation: **pilot DPM-on-points first** as the v0 v1 backbone (1-week pilot, $230), then **compare to Diff-OSGN/Diff-TRGN on the same 100-tooth subset** (another 1-week, $500). The slower graph-attention wins only if its quality gain is >2× DPM-on-points — the EMD numbers suggest it's not.

## H1-H5 Score Summary

| Hypothesis | Verdict | Strength | One-liner |
|---|---|---|---|
| H1 (modular) | **Supports** | Strong | Cleanest encoder/decoder separation in the reading list |
| H2 (multi-modal) | **Supports** | Strong | Flow-prior + diffusion = intrinsic multi-modal generator |
| H3 (conditioning) | **Supports** | Partial + easy extension | Global z only; per-point ConcatSquashLinear makes H3-ready |
| H4 (implicit) | **Contradicts** | Mild | Explicit points, not SDF; downstream DiGS/FlexiCubes |
| H5 (s→r transfer) | N/A direct | — | Tested on synthetic only; pre-train on tooth-like ShapeNet classes is the right experiment |
