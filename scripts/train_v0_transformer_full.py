#!/usr/bin/env python3
"""v0 Phase 2: Full Transformer (DMC-style, 4096 pts, full attention).

Upgraded from train_v0_transformer.py (which was 1024 pts, 2 layers — too small).

Differences vs lightweight version:
  - 4096 points (was 1024) — full resolution for spatial fidelity
  - 4 layers, 8 heads, 512d (was 2/4/256) — real DMC-style capacity
  - Centroid L2 loss added — prevents position collapse
  - AdamW + weight decay — better regularization

Designed for M4 Max 128GB (MacBook). Will OOM on Mac mini 16GB.
Expected: val CD 0.025-0.035 (target — beat v0 simple's 0.0389).

Performance notes (revised 2026-06-06):
  - DataLoader uses workers + prefetch (was num_workers=0, GPU-starved)
  - KNN indices are cached to disk per sample (was O(N^2) cdist+topk on MPS every step)
  - Train loop avoids per-batch .item() syncs (was breaking MPS pipelining)
"""
import os
import json
import argparse
import time
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

# CUDA perf knobs (4090). These are no-ops on MPS.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True      # ~2x matmul speedup
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True            # autotune conv kernels

# On macOS, MPS's high-watermark (default 0.9) caps GPU memory at ~90% of
# system RAM. On a 128 GB M-series that gives ~115 GB max, but the
# allocator cache pool grows to that ceiling over a full-data epoch run
# and OOMs. Cap at 0.5 (~64 GB max on 128 GB systems) — the allocator
# still has plenty of room and will release cache as it approaches the
# cap instead of going to swap. We also drop the cache per step below.
# Pin both HIGH and LOW (PyTorch 2.12 has a non-0.0 default for LOW that
# exceeds HIGH and trips an "invalid low watermark ratio" assertion).
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.5"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"


# ---------------------------------------------------------------------------
# KNN index helper (used by Dataset for offline caching).
# Prefer scipy KDTree (much faster than O(N^2) cdist on CPU); fall back to torch.
# ---------------------------------------------------------------------------
try:
    from scipy.spatial import cKDTree as _KDTree  # type: ignore

    def _compute_knn_idx(pc, k):
        # pc: (N, 3) float32 numpy. Returns (N, k) int32 excluding self.
        _, idx = _KDTree(pc).query(pc, k=k + 1)
        return idx[:, 1:].astype(np.int32)
except Exception:  # pragma: no cover
    def _compute_knn_idx(pc, k):
        p = torch.from_numpy(pc)
        d = torch.cdist(p.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
        return d.topk(k + 1, largest=False).indices[:, 1:].numpy().astype(np.int32)


def chamfer_torch(a, b):
    """a: (B, N, 3), b: (B, M, 3) -> (B,) CD"""
    d = torch.cdist(a, b)
    return d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1)


def centroid_l2(a, b):
    """L2 distance between point cloud centroids. Forces position-aware output."""
    return (a.mean(dim=1) - b.mean(dim=1)).norm(dim=-1)


class V0Dataset(Dataset):
    def __init__(self, files, n_points=4096, k=16, knn_cache_dir=None):
        self.files = files
        self.n_points = n_points
        self.k = k
        # Resolve to absolute path. DataLoader workers may run with a different
        # CWD (especially on macOS), and a relative path there resolves to a
        # non-existent directory — that triggered a FileNotFoundError in os.replace.
        self.knn_cache_dir = (
            os.path.abspath(knn_cache_dir) if knn_cache_dir is not None else None
        )
        if self.knn_cache_dir is not None:
            os.makedirs(self.knn_cache_dir, exist_ok=True)
        self.fdi_to_idx = {fdi: i for i, fdi in enumerate(
            list(range(11, 18)) + list(range(21, 28)) + list(range(31, 38)) + list(range(41, 48))
        )}

    def _ensure_cache_dir(self):
        # Idempotent in worker processes: each worker may have its own CWD and
        # may not see the dir created by the parent.
        if self.knn_cache_dir is not None:
            os.makedirs(self.knn_cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.files)

    def _cache_path(self, i):
        if self.knn_cache_dir is None:
            return None
        base = os.path.splitext(os.path.basename(self.files[i]))[0]
        return os.path.join(self.knn_cache_dir, f"{base}_n{self.n_points}_k{self.k}.npy")

    def __getitem__(self, i):
        d = np.load(self.files[i])
        partial = d['partial_pc'].astype(np.float32)
        target = d['target_pc'].astype(np.float32)
        # Sub-sample to n_points if needed (use deterministic seed per index)
        rng = np.random.default_rng(i)
        if partial.shape[0] != self.n_points:
            idx = rng.choice(partial.shape[0], self.n_points, replace=False)
            partial = partial[idx]
        if target.shape[0] != self.n_points:
            idx = rng.choice(target.shape[0], self.n_points, replace=False)
            target = target[idx]

        # KNN indices: cache-on-first-use. Both sub-sampling and KNN are deterministic
        # in (i, n_points, k), so a per-file cache is valid across epochs.
        cache_path = self._cache_path(i)
        if cache_path is not None and os.path.exists(cache_path):
            knn_idx = np.load(cache_path)
        else:
            knn_idx = _compute_knn_idx(partial, self.k)
            if cache_path is not None:
                # Make sure the directory exists in this process (we may be a worker).
                self._ensure_cache_dir()
                # np.save silently appends ".npy" if the path doesn't already
                # end in it — keep the .npy suffix here or we'll be checking the
                # existence of a path that was actually written one suffix deeper.
                tmp = cache_path + f".tmp.{os.getpid()}.npy"
                np.save(tmp, knn_idx)
                # Defensive: only replace if the temp actually got written.
                if os.path.exists(tmp):
                    os.replace(tmp, cache_path)

        fdi = int(d['fdi'])
        fdi_idx = self.fdi_to_idx[fdi]
        return {
            'partial': torch.from_numpy(partial),
            'target': torch.from_numpy(target),
            'knn_idx': torch.from_numpy(knn_idx).long(),
            'fdi_idx': fdi_idx,
            'fdi': fdi,
        }


class PointNetEncoder(nn.Module):
    """Per-point MLP + max-pool -> global feature."""
    def __init__(self, out_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.GELU(),
            nn.Conv1d(64, 128, 1), nn.GELU(),
            nn.Conv1d(128, 256, 1), nn.GELU(),
            nn.Conv1d(256, out_dim, 1), nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        # x: (B, N, 3)
        x = x.transpose(1, 2)  # (B, 3, N)
        x = self.mlp(x)  # (B, out_dim, N)
        return x.max(dim=2).values  # (B, out_dim)


class KNNGrouping(nn.Module):
    """Per-point k-NN features: for each point, concat [neighbor, relative position].

    If `knn_idx` is supplied (precomputed in the Dataset / DataLoader workers),
    skip the O(N^2) cdist+topk on MPS — that was the dominant cost on Apple Silicon.
    """
    def __init__(self, k=16, out_dim=64):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(6, 64, 1), nn.GELU(),
            nn.Conv2d(64, out_dim, 1), nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, x, knn_idx=None):
        # x: (B, N, 3); knn_idx (optional): (B, N, k) long
        B, N, _ = x.shape
        if knn_idx is None:
            d = torch.cdist(x, x)  # (B, N, N)
            knn_idx = d.topk(self.k + 1, largest=False).indices[..., 1:]  # exclude self
        neighbors = x.unsqueeze(2).expand(B, N, self.k, 3).gather(
            2, knn_idx.unsqueeze(-1).expand(B, N, self.k, 3)
        )
        rel = neighbors - x.unsqueeze(2)
        feat = torch.cat([neighbors, rel], dim=-1)  # (B, N, k, 6)
        feat = feat.permute(0, 3, 1, 2)  # (B, 6, N, k)
        feat = self.mlp(feat)  # (B, out_dim, N, k)
        feat = feat.max(dim=-1).values  # (B, out_dim, N)
        return feat.transpose(1, 2)  # (B, N, out_dim)


class TransformerDecoder(nn.Module):
    """Transformer decoder with learnable queries.

    Uses gradient checkpointing to reduce activation memory ~4x. Time cost ~30%.
    Applied only in training mode; eval reuses the standard forward.
    """
    def __init__(self, dim=512, n_layers=4, n_heads=8, n_points=4096):
        super().__init__()
        self.n_points = n_points
        self.queries = nn.Parameter(torch.randn(n_points, dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=0.1, batch_first=True, activation='gelu',
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

    def forward(self, context, query_proj):
        if self.training:
            return checkpoint(
                self.decoder, query_proj, context,
                use_reentrant=False,
            )
        return self.decoder(tgt=query_proj, memory=context)


class CrownGenTransformerFull(nn.Module):
    """DMC-style transformer for point cloud completion.

    Architecture:
      1. PointNet encoder -> global feature (512d)
      2. KNN grouping (k=16) -> per-point features (64d) preserving local structure
      3. Concat [knn | global | fdi_emb] -> (B, N, 640d)
      4. Linear projection to decoder dim (512d)
      5. Transformer decoder with learnable queries (4 layers, 8 heads)
      6. Per-point output head -> (B, N, 3)
    """
    def __init__(self, n_fdi=28, n_points=4096, k=16,
                 encoder_dim=512, knn_dim=64, fdi_dim=64,
                 dec_dim=512, n_layers=4, n_heads=8):
        super().__init__()
        self.n_points = n_points
        self.encoder = PointNetEncoder(out_dim=encoder_dim)
        self.knn = KNNGrouping(k=k, out_dim=knn_dim)
        self.fdi_emb = nn.Embedding(n_fdi, fdi_dim)
        self.ctx_proj = nn.Linear(encoder_dim + knn_dim + fdi_dim, dec_dim)
        self.q_proj = nn.Linear(dec_dim, dec_dim)
        self.decoder = TransformerDecoder(dim=dec_dim, n_layers=n_layers, n_heads=n_heads, n_points=n_points)
        self.head = nn.Linear(dec_dim, 3)

    def forward(self, partial, fdi_idx, knn_idx=None):
        B, N, _ = partial.shape
        global_feat = self.encoder(partial)  # (B, 512)
        knn_feat = self.knn(partial, knn_idx=knn_idx)  # (B, N, 64)
        fdi_feat = self.fdi_emb(fdi_idx)  # (B, 64)
        ctx = torch.cat([
            knn_feat,
            global_feat.unsqueeze(1).expand(B, N, -1),
            fdi_feat.unsqueeze(1).expand(B, N, -1),
        ], dim=-1)  # (B, N, 640)
        ctx = self.ctx_proj(ctx)  # (B, N, 512)
        q = self.q_proj(self.decoder.queries.unsqueeze(0).expand(B, -1, -1))  # (B, 4096, 512)
        out = self.decoder(ctx, q)  # (B, 4096, 512)
        return self.head(out)  # (B, 4096, 3)


def evaluate(model, loader, device):
    model.eval()
    # Accumulate on-device to avoid per-batch GPU->CPU syncs.
    cd_sum = torch.zeros((), device=device)
    cd_sq_sum = torch.zeros((), device=device)
    cent_sum = torch.zeros((), device=device)
    n = 0
    n_skipped_total = 0
    with torch.no_grad():
        for batch in loader:
            partial = batch['partial'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            fdi_idx = batch['fdi_idx'].to(device, non_blocking=True)
            knn_idx = batch['knn_idx'].to(device, non_blocking=True)
            pred = model(partial, fdi_idx, knn_idx)
            cd = chamfer_torch(pred, target)
            cent = centroid_l2(pred, target)
            # Skip NaN/inf samples (rare forward-path blowups). The val set
            # is deterministic (shuffle=False), so the same samples re-fire
            # every epoch — without skipping, val_cd stays NaN forever and
            # best.pt never updates. Log the skip count for diagnostics.
            finite_mask = torch.isfinite(cd) & torch.isfinite(cent)
            n_skipped = (~finite_mask).sum().item()
            n_skipped_total += n_skipped
            if n_skipped:
                cd = cd[finite_mask]
                cent = cent[finite_mask]
            if cd.numel() == 0:
                continue
            cd_sum += cd.sum()
            cd_sq_sum += (cd * cd).sum()
            cent_sum += cent.sum()
            n += cd.numel()
    mean = (cd_sum / max(n, 1)).item()
    var = max((cd_sq_sum / max(n, 1)).item() - mean * mean, 0.0)
    std = var ** 0.5
    cent_mean = (cent_sum / max(n, 1)).item()
    return mean, std, cent_mean, n_skipped_total


def main(split_path, n_epochs=20, batch_size=1, lr=5e-4, device='mps',
         n_points=1024, out_dir='models/v0_transformer_full',
         max_train=None, max_val=None,
         num_workers=0, knn_cache_dir='./data/v0_knn_cache'):
    with open(split_path) as f:
        s = json.load(f)
    train_files = s['train_files']
    val_files = s['val_files']
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    os.makedirs(out_dir, exist_ok=True)

    train_ds = V0Dataset(train_files, n_points=n_points, knn_cache_dir=knn_cache_dir)
    val_ds = V0Dataset(val_files, n_points=n_points, knn_cache_dir=knn_cache_dir)
    if max_train:
        train_ds.files = train_ds.files[:max_train]
        print(f"[DEBUG] subsampled train to {max_train}")
    if max_val:
        val_ds.files = val_ds.files[:max_val]
        print(f"[DEBUG] subsampled val to {max_val}")

    # M-series unified memory: pin_memory has no benefit (and warns). Skip it.
    # Trade-off summary:
    #   - 'spawn' (set globally below) keeps workers from inheriting the
    #     parent's MPS allocator high-watermark cache pool (~80GB).
    #   - persistent_workers=True keeps workers alive across epochs so we
    #     don't pay the 5-10s spawn cost per batch fetch (which was the
    #     "1 epoch never finishes" bug when combined with num_workers=6).
    #   - num_workers=2 (down from 6): halves allocator state across workers
    #     and is enough to keep the GPU fed given prefetch_factor=4.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=max(1, num_workers // 2) if num_workers > 0 else 0,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )

    model = CrownGenTransformerFull(n_fdi=28, n_points=n_points).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params, device: {device}")
    print(f"DataLoader: num_workers={num_workers}, knn_cache_dir={knn_cache_dir}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float('inf')

    # CUDA bf16 autocast. MPS doesn't support bf16/fp16 reliably, so this is
    # gated on CUDA. Activates Tensor Cores on 4090 (Ampere) for ~2x speedup
    # with no quality loss for our 4-layer transformer.
    use_amp = (device == 'cuda' and torch.cuda.is_available())
    amp_dtype = torch.bfloat16 if use_amp else None

    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        # Accumulate train CD on-device; avoid per-batch .item() (breaks MPS pipelining).
        train_cd_sum = torch.zeros((), device=device)
        n_batches = 0
        for batch in train_loader:
            partial = batch['partial'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            fdi_idx = batch['fdi_idx'].to(device, non_blocking=True)
            knn_idx = batch['knn_idx'].to(device, non_blocking=True)
            opt.zero_grad()
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    pred = model(partial, fdi_idx, knn_idx)
                    cd = chamfer_torch(pred, target)
                    cent = centroid_l2(pred, target)
                    loss = (cd + 0.5 * cent).mean()
                loss.backward()
            else:
                pred = model(partial, fdi_idx, knn_idx)
                cd = chamfer_torch(pred, target)
                cent = centroid_l2(pred, target)
                loss = (cd + 0.5 * cent).mean()
                loss.backward()
            opt.step()
            train_cd_sum += cd.detach().mean()
            n_batches += 1
            # Drop MPS allocator cache every step. Without this, the cache
            # pool inflates to 90% of system RAM over a full-data epoch and
            # OOMs. ~5-10% throughput cost, but the alternative is OOM.
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

        train_cd = (train_cd_sum / max(n_batches, 1)).item()
        train_time = time.time() - t0
        # Safety net: drop MPS allocator cache pool before evaluation. Without
        # this, fragmentation in a long training run can balloon the allocator
        # high-water mark (we observed 110 GB on M-series after several epochs).
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        val_cd, val_std, val_cent, n_skipped = evaluate(model, val_loader, device)
        skip_msg = f"  (skipped {n_skipped} NaN/inf)" if n_skipped else ""
        print(f"Epoch {epoch+1:3d}/{n_epochs} | train CD: {train_cd*1000:.2f} | "
              f"val CD: {val_cd*1000:.2f} ± {val_std*1000:.2f} "
              f"val centErr: {val_cent:.3f} (×1e-3) | {train_time:.1f}s/epoch{skip_msg}", flush=True)

        if val_cd < best_val:
            best_val = val_cd
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
            print(f"  ✓ New best: {best_val*1000:.2f} (×1e-3)", flush=True)

    torch.save(model.state_dict(), os.path.join(out_dir, 'final.pt'))
    print(f"\n=== Done. Best val CD: {best_val*1000:.2f} (×1e-3) ===")
    print(f"Model saved to {out_dir}/")


if __name__ == "__main__":
    # Use 'spawn' for DataLoader workers: under macOS's default 'fork' method,
    # workers inherit the parent's MPS allocator high-watermark cache pool
    # (potentially 80+ GB of reserved memory). 'spawn' gives each worker a
    # clean Python interpreter with no inherited MPS state.
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # already set; ignore.

    p = argparse.ArgumentParser()
    p.add_argument('--split', default='./data/v0_split.json')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=1,
                   help='bs=1 default is M-series safe. On 24GB+ CUDA GPUs use '
                        '--batch-size 8 for much faster training.')
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--device', default='mps', choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--out-dir', default='./models/v0_transformer_full')
    p.add_argument('--max-train', type=int, default=None)
    p.add_argument('--max-val', type=int, default=None)
    p.add_argument('--n-points', type=int, default=1024,
                   help='n=1024 default is M-series safe. On 24GB+ CUDA GPUs '
                        'use --n-points 4096 for full DMC resolution.')
    p.add_argument('--num-workers', type=int, default=0,
                   help='DataLoader workers (0 disables prefetch). Default 0 for memory '
                        'stability on MPS. On CUDA use 8+ for full prefetch.')
    p.add_argument('--knn-cache-dir', default='./data/v0_knn_cache',
                   help='Where to cache per-sample KNN indices. Set to "" to disable caching.')
    args = p.parse_args()
    knn_cache = args.knn_cache_dir or None
    main(args.split, n_epochs=args.epochs, batch_size=args.batch_size,
         lr=args.lr, device=args.device, out_dir=args.out_dir,
         max_train=args.max_train, max_val=args.max_val,
         n_points=args.n_points,
         num_workers=args.num_workers, knn_cache_dir=knn_cache)
