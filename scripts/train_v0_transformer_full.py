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
"""
import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def chamfer_torch(a, b):
    """a: (B, N, 3), b: (B, M, 3) -> (B,) CD"""
    d = torch.cdist(a, b)
    return d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1)


def centroid_l2(a, b):
    """L2 distance between point cloud centroids. Forces position-aware output."""
    return (a.mean(dim=1) - b.mean(dim=1)).norm(dim=-1)


class V0Dataset(Dataset):
    def __init__(self, files, n_points=4096):
        self.files = files
        self.n_points = n_points
        self.fdi_to_idx = {fdi: i for i, fdi in enumerate(
            list(range(11, 18)) + list(range(21, 28)) + list(range(31, 38)) + list(range(41, 48))
        )}

    def __len__(self):
        return len(self.files)

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
        fdi = int(d['fdi'])
        fdi_idx = self.fdi_to_idx[fdi]
        return {
            'partial': torch.from_numpy(partial),
            'target': torch.from_numpy(target),
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
    """Per-point k-NN features: for each point, concat [neighbor, relative position]."""
    def __init__(self, k=16, out_dim=64):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(6, 64, 1), nn.GELU(),
            nn.Conv2d(64, out_dim, 1), nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        # x: (B, N, 3)
        B, N, _ = x.shape
        d = torch.cdist(x, x)  # (B, N, N)
        idx = d.topk(self.k + 1, largest=False).indices[..., 1:]  # (B, N, k) — exclude self
        neighbors = x.unsqueeze(2).expand(B, N, self.k, 3).gather(
            2, idx.unsqueeze(-1).expand(B, N, self.k, 3)
        )
        rel = neighbors - x.unsqueeze(2)
        feat = torch.cat([neighbors, rel], dim=-1)  # (B, N, k, 6)
        feat = feat.permute(0, 3, 1, 2)  # (B, 6, N, k)
        feat = self.mlp(feat)  # (B, out_dim, N, k)
        feat = feat.max(dim=-1).values  # (B, out_dim, N)
        return feat.transpose(1, 2)  # (B, N, out_dim)


class TransformerDecoder(nn.Module):
    """Transformer decoder with learnable queries."""
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

    def forward(self, partial, fdi_idx):
        B, N, _ = partial.shape
        global_feat = self.encoder(partial)  # (B, 512)
        knn_feat = self.knn(partial)  # (B, N, 64)
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
    cds = []
    cents = []
    with torch.no_grad():
        for batch in loader:
            partial = batch['partial'].to(device)
            target = batch['target'].to(device)
            fdi_idx = batch['fdi_idx'].to(device)
            pred = model(partial, fdi_idx)
            cd = chamfer_torch(pred, target)
            cent = centroid_l2(pred, target)
            cds.extend(cd.cpu().tolist())
            cents.extend(cent.cpu().tolist())
    return np.mean(cds), np.std(cds), np.mean(cents)


def main(split_path, n_epochs=20, batch_size=8, lr=5e-4, device='mps',
         n_points=4096, out_dir='models/v0_transformer_full',
         max_train=None, max_val=None):
    with open(split_path) as f:
        s = json.load(f)
    train_files = s['train_files']
    val_files = s['val_files']
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    os.makedirs(out_dir, exist_ok=True)

    train_ds = V0Dataset(train_files, n_points=n_points)
    val_ds = V0Dataset(val_files, n_points=n_points)
    if max_train:
        train_ds.files = train_ds.files[:max_train]
        print(f"[DEBUG] subsampled train to {max_train}")
    if max_val:
        val_ds.files = val_ds.files[:max_val]
        print(f"[DEBUG] subsampled val to {max_val}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CrownGenTransformerFull(n_fdi=28, n_points=n_points).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params, device: {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float('inf')

    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        train_cds = []
        for batch in train_loader:
            partial = batch['partial'].to(device)
            target = batch['target'].to(device)
            fdi_idx = batch['fdi_idx'].to(device)
            pred = model(partial, fdi_idx)
            cd = chamfer_torch(pred, target)
            cent = centroid_l2(pred, target)
            loss = (cd + 0.5 * cent).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_cds.append(cd.mean().item())

        train_cd = np.mean(train_cds)
        train_time = time.time() - t0
        val_cd, val_std, val_cent = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1:3d}/{n_epochs} | train CD: {train_cd*1000:.2f} | "
              f"val CD: {val_cd*1000:.2f} ± {val_std*1000:.2f} "
              f"val centErr: {val_cent:.3f} (×1e-3) | {train_time:.1f}s/epoch", flush=True)

        if val_cd < best_val:
            best_val = val_cd
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
            print(f"  ✓ New best: {best_val*1000:.2f} (×1e-3)", flush=True)

    torch.save(model.state_dict(), os.path.join(out_dir, 'final.pt'))
    print(f"\n=== Done. Best val CD: {best_val*1000:.2f} (×1e-3) ===")
    print(f"Model saved to {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='./data/v0_split.json')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--device', default='mps', choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--out-dir', default='./models/v0_transformer_full')
    p.add_argument('--max-train', type=int, default=None)
    p.add_argument('--max-val', type=int, default=None)
    p.add_argument('--n-points', type=int, default=4096)
    args = p.parse_args()
    main(args.split, n_epochs=args.epochs, batch_size=args.batch_size,
         lr=args.lr, device=args.device, out_dir=args.out_dir,
         max_train=args.max_train, max_val=args.max_val,
         n_points=args.n_points)
