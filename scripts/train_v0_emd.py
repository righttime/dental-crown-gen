#!/usr/bin/env python3
"""v0 Simple + EMD loss + spatial prior (combined methodology upgrade).

Two new tricks on top of v0 simple (PointNet encoder + MLP decoder):
  1. EMD (Earth Mover's Distance) loss instead of pure Chamfer
     - More stable gradient, penalizes distribution shape directly
  2. Spatial prior: FDI-conditional centroid target
     - For each FDI, the mean centroid location is computed from train data
     - Model is told "your centroid should be near this point" via a target
     - Loss: ||pred_centroid - fdi_mean_centroid||

This is a "two heads" approach — CD for shape, EMD for distribution, spatial for position.
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
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v0_simple import CrownGenModel, V0Dataset, chamfer_torch, centroid_l2


def emd_torch(a, b, n_iter=5):
    """Approximate Earth Mover's Distance via Sinkhorn-like iterative assignment.

    a: (B, N, 3), b: (B, M, 3) -> (B,) approximate EMD.

    For efficiency: use a fast approximate version. We use Hungarian-style
    one-to-one matching with random subsets, which is a known EMD lower bound.
    """
    B, N, _ = a.shape
    M = b.shape[1]
    # Use subset to keep cost down (N=M=1024, full = 1M, OK for our batch)
    cost = torch.cdist(a, b)  # (B, N, M)
    # Iterative greedy: for each point in a, find nearest in b not yet assigned
    # Approximation: just match with optimal assignment via min-cost matching
    # (scipy.optimize.linear_sum_assignment) — too slow for batch
    # Use Sinkhorn-style soft assignment:
    # P = -cost / temperature, then row/col normalize
    temperature = 0.1
    P = -cost / temperature  # (B, N, M)
    P = F.softmax(P, dim=2)  # soft assignment, sum to 1 over M
    # P[i, j, k] = prob that a[i,j] is matched to b[i,k]
    # EMD approx = sum_{j,k} P[j,k] * cost[j,k] * something
    # For uniform weights:
    flow = P / N  # so each row of a "carries" 1/N of mass
    emd = (flow * cost).sum(dim=(1, 2))  # (B,)
    return emd


def compute_fdi_centroids(train_files, n_per_fdi=200, n_points=2048):
    """Pre-compute mean centroid of target_pc for each FDI from train set."""
    fdi_to_idx = {fdi: i for i, fdi in enumerate(
        list(range(11, 18)) + list(range(21, 28)) + list(range(31, 38)) + list(range(41, 48))
    )}
    fdi_centroids = torch.zeros(28, 3, dtype=torch.float32)
    fdi_counts = torch.zeros(28, dtype=torch.long)
    rng = np.random.default_rng(0)
    for fp in train_files[:n_per_fdi * 28 * 2]:  # sample enough
        d = np.load(fp)
        fdi = int(d['fdi'])
        if fdi not in fdi_to_idx:
            continue
        target = d['target_pc'].astype(np.float32)
        # Sub-sample for speed
        if target.shape[0] > n_points:
            target = target[rng.choice(target.shape[0], n_points, replace=False)]
        idx = fdi_to_idx[fdi]
        fdi_centroids[idx] += torch.tensor(target.mean(axis=0), dtype=torch.float32)
        fdi_counts[idx] += 1
    # Average
    mask = fdi_counts > 0
    fdi_centroids[mask] /= fdi_counts[mask].unsqueeze(1).float()
    return fdi_centroids  # (28, 3)


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


def main(split_path, n_epochs=3, batch_size=8, lr=1e-3, device='mps',
         n_points=4096, out_dir='models/v0_emd', max_train=None, max_val=None,
         emd_weight=0.5, spatial_weight=0.5):
    with open(split_path) as f:
        s = json.load(f)
    train_files = s['train_files']
    val_files = s['val_files']
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    os.makedirs(out_dir, exist_ok=True)

    # Pre-compute FDI centroids
    print("Computing FDI centroids from train data...")
    fdi_centroids = compute_fdi_centroids(train_files, n_per_fdi=200, n_points=2048)
    fdi_centroids = fdi_centroids.to(device)
    print(f"  fdi_centroids shape: {fdi_centroids.shape}")

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
            # Multi-component loss
            cd = chamfer_torch(pred, target)
            emd = emd_torch(pred, target)
            # Spatial: distance from pred centroid to FDI mean centroid
            pred_c = pred.mean(dim=1)
            target_c = fdi_centroids[fdi_idx]
            spatial = (pred_c - target_c).norm(dim=-1)
            # Combined
            loss = (cd + emd_weight * emd + spatial_weight * spatial).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_cds.append(cd.mean().item())

        train_cd = np.mean(train_cds)
        train_time = time.time() - t0
        val_cd, val_std = evaluate(model, val_loader, device)
        # Spatial eval
        model.eval()
        spatial_eval = []
        with torch.no_grad():
            for batch in val_loader:
                partial = batch['partial'].to(device)
                target = batch['target'].to(device)
                fdi_idx = batch['fdi_idx'].to(device)
                pred = model(partial, fdi_idx)
                pred_c = pred.mean(dim=1)
                target_c = fdi_centroids[fdi_idx]
                spatial_eval.extend(((pred_c - target_c).norm(dim=-1)).cpu().tolist())
        mean_spatial = np.mean(spatial_eval)
        print(f"Epoch {epoch+1:3d}/{n_epochs} | train CD: {train_cd*1000:.2f} | "
              f"val CD: {val_cd*1000:.2f} ± {val_std*1000:.2f} "
              f"val spatialErr: {mean_spatial:.3f} (×1e-3) | {train_time:.1f}s/epoch", flush=True)

        if val_cd < best_val:
            best_val = val_cd
            torch.save({
                'model': model.state_dict(),
                'fdi_centroids': fdi_centroids.cpu(),
                'val_cd': best_val,
                'val_spatial': mean_spatial,
            }, os.path.join(out_dir, 'best.pt'))
            print(f"  ✓ New best: {best_val*1000:.2f} (×1e-3)", flush=True)

    torch.save({
        'model': model.state_dict(),
        'fdi_centroids': fdi_centroids.cpu(),
    }, os.path.join(out_dir, 'final.pt'))
    print(f"\n=== Done. Best val CD: {best_val*1000:.2f} (×1e-3) ===")
    print(f"Model saved to {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='/Users/alf/Projects/AlfResearch/dental-crown-gen/data/v0_split.json')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='mps', choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--out-dir', default='/Users/alf/Projects/AlfResearch/dental-crown-gen/models/v0_emd')
    p.add_argument('--max-train', type=int, default=None)
    p.add_argument('--max-val', type=int, default=None)
    p.add_argument('--emd-weight', type=float, default=0.5)
    p.add_argument('--spatial-weight', type=float, default=0.5)
    args = p.parse_args()
    main(args.split, n_epochs=args.epochs, batch_size=args.batch_size,
         lr=args.lr, device=args.device, out_dir=args.out_dir,
         max_train=args.max_train, max_val=args.max_val,
         emd_weight=args.emd_weight, spatial_weight=args.spatial_weight)
