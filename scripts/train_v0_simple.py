#!/usr/bin/env python3
"""v0 Tier 2: simple trainable model.

Architecture:
  - PointNet encoder on partial_pc -> 1024-dim global feature
  - Concat with FDI one-hot (28-dim) -> 1052-dim
  - MLP decoder: 1052 -> 1024 -> 1024 -> 4096*3
  - Reshape to (B, 4096, 3) target prediction

Loss: symmetric Chamfer Distance to GT target_pc.
Training: Adam, MPS-accelerated.
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


class V0Dataset(Dataset):
    def __init__(self, files, n_points=4096):
        self.files = files
        self.n_points = n_points
        # FDI one-hot: 28 classes (11-17, 21-27, 31-37, 41-47)
        self.fdi_to_idx = {fdi: i for i, fdi in enumerate(
            list(range(11, 18)) + list(range(21, 28)) + list(range(31, 38)) + list(range(41, 48))
        )}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        partial = d['partial_pc'].astype(np.float32)  # (4096, 3)
        target = d['target_pc'].astype(np.float32)    # (4096, 3)
        fdi = int(d['fdi'])
        fdi_idx = self.fdi_to_idx[fdi]
        return {
            'partial': torch.from_numpy(partial),
            'target': torch.from_numpy(target),
            'fdi_idx': fdi_idx,
            'fdi': fdi,
        }


class PointNetEncoder(nn.Module):
    """Simple PointNet encoder: per-point MLP -> max-pool -> global feature."""
    def __init__(self, out_dim=1024):
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
        x = x.max(dim=2).values  # (B, out_dim)
        return x


class CrownGenModel(nn.Module):
    def __init__(self, n_fdi=28, n_points=4096, feat_dim=1024, hidden=1024):
        super().__init__()
        self.n_points = n_points
        self.encoder = PointNetEncoder(out_dim=feat_dim)
        self.fdi_emb = nn.Embedding(n_fdi, 64)
        # Decoder: feat_dim + 64 -> hidden -> ... -> n_points * 3
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim + 64, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_points * 3),
        )

    def forward(self, partial, fdi_idx):
        feat = self.encoder(partial)  # (B, feat_dim)
        fdi_feat = self.fdi_emb(fdi_idx)  # (B, 64)
        z = torch.cat([feat, fdi_feat], dim=1)  # (B, feat_dim+64)
        out = self.decoder(z)  # (B, n_points * 3)
        return out.view(-1, self.n_points, 3)


def evaluate(model, loader, device):
    model.eval()
    cds = []
    with torch.no_grad():
        for batch in loader:
            partial = batch['partial'].to(device)
            target = batch['target'].to(device)
            fdi_idx = batch['fdi_idx'].to(device)
            pred = model(partial, fdi_idx)
            cd = chamfer_torch(pred, target)
            cds.extend(cd.cpu().tolist())
    return np.mean(cds), np.std(cds)


def main(split_path, n_epochs=20, batch_size=32, lr=1e-3, device='mps', n_points=4096, out_dir='models/v0_simple'):
    with open(split_path) as f:
        s = json.load(f)
    train_files = s['train_files']
    val_files = s['val_files']
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    os.makedirs(out_dir, exist_ok=True)

    train_ds = V0Dataset(train_files, n_points=n_points)
    val_ds = V0Dataset(val_files, n_points=n_points)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = CrownGenModel(n_fdi=28, n_points=n_points).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params, device: {device}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
            loss = chamfer_torch(pred, target).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_cds.append(loss.item())
        train_cd = np.mean(train_cds)
        train_time = time.time() - t0

        # Validate
        val_cd, val_std = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{n_epochs} | train CD: {train_cd*1000:.2f} | "
              f"val CD: {val_cd*1000:.2f} ± {val_std*1000:.2f} "
              f"(×1e-3) | {train_time:.1f}s/epoch")

        # Save best
        if val_cd < best_val:
            best_val = val_cd
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
            print(f"  ✓ New best: {best_val*1000:.2f} (×1e-3)")

    # Final save
    torch.save(model.state_dict(), os.path.join(out_dir, 'final.pt'))
    print(f"\n=== Done. Best val CD: {best_val*1000:.2f} (×1e-3) ===")
    print(f"Model saved to {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='/Users/alf/Projects/AlfResearch/dental-crown-gen/data/v0_split.json')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='mps', choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--out-dir', default='/Users/alf/Projects/AlfResearch/dental-crown-gen/models/v0_simple')
    args = p.parse_args()
    main(args.split, n_epochs=args.epochs, batch_size=args.batch_size,
         lr=args.lr, device=args.device, out_dir=args.out_dir)
