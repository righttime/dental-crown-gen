#!/usr/bin/env python3
"""v0 baseline: per-FDI 1-NN retrieval (no training).

For each val sample:
  - Find training sample with same FDI label that has lowest Chamfer to val partial
  - Use its target_pc as prediction
  - Compute Chamfer to val target_pc

Also report per-FDI mean target as a weaker baseline.

Optimized: pre-load all train partial_pcs into memory (~3 GB).
"""
import os
import json
import glob
import argparse
import numpy as np
import torch
from collections import defaultdict


def chamfer_torch(a, b):
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    d = torch.cdist(a, b)
    return (d.min(dim=1).values.mean() + d.min(dim=0).values.mean()).item()


def parse_fdi(path):
    base = os.path.basename(path)
    for p in base.split("_"):
        if p.startswith("fdi"):
            return int(p[3:])
    raise ValueError(f"No FDI in {base}")


def main(split_path, K=1, max_val=None, max_train_per_fdi=200, device='mps'):
    with open(split_path) as f:
        s = json.load(f)
    train_files = s['train_files']
    val_files = s['val_files']
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    if max_val:
        val_files = val_files[:max_val]

    # Group train by FDI, subsample for speed
    train_by_fdi = defaultdict(list)
    for fp in train_files:
        fdi = parse_fdi(fp)
        train_by_fdi[fdi].append(fp)
    for fdi in train_by_fdi:
        train_by_fdi[fdi] = train_by_fdi[fdi][:max_train_per_fdi]
    print(f"FDIs in train: {sorted(train_by_fdi.keys())}")
    total_train_used = sum(len(v) for v in train_by_fdi.values())
    print(f"Total train samples used: {total_train_used}")

    # Pre-load all train partial_pcs and target_pcs
    print(f"Loading {total_train_used} train files into memory...")
    train_partials = {}  # fp -> (4096, 3) numpy
    train_targets = {}   # fp -> (4096, 3) numpy
    for fdi, files in train_by_fdi.items():
        for fp in files:
            d = np.load(fp)
            train_partials[fp] = d['partial_pc']
            train_targets[fp] = d['target_pc']
    print(f"Loaded: {len(train_partials)} partials, {len(train_targets)} targets")
    mem_partials = sum(a.nbytes for a in train_partials.values()) / 1024**3
    mem_targets = sum(a.nbytes for a in train_targets.values()) / 1024**3
    print(f"Memory: partials {mem_partials:.2f} GB, targets {mem_targets:.2f} GB")

    # === Baseline 1: per-FDI mean target ===
    print("\n=== Baseline 1: per-FDI mean target ===")
    fdi_mean = {}
    for fdi, files in train_by_fdi.items():
        ts = [train_targets[fp] for fp in files]
        fdi_mean[fdi] = np.mean(ts, axis=0).astype(np.float32)

    # === Baseline 2: 1-NN retrieval by partial-arch Chamfer ===
    print(f"\n=== Baseline 2: {K}-NN retrieval (K={K}) by partial-arch Chamfer ===")
    val_chamfers_mean = []
    val_chamfers_knn = []
    val_chamfers_per_fdi = defaultdict(lambda: [0.0, 0, 0.0])  # [sum_knn, count, sum_mean]

    # Pre-compute: stack partials per FDI as (N, 4096, 3) torch tensor for fast batched CD
    print("Stacking train partials per FDI for batched CD...")
    fdi_partials_stack = {}
    for fdi, files in train_by_fdi.items():
        stack = np.stack([train_partials[fp] for fp in files])  # (N, 4096, 3)
        fdi_partials_stack[fdi] = torch.tensor(stack, dtype=torch.float32, device=device)
    print(f"Stacked {len(fdi_partials_stack)} FDIs on {device}")

    for i, vfp in enumerate(val_files):
        vd = np.load(vfp)
        v_fdi = parse_fdi(vfp)
        v_partial = vd['partial_pc']
        v_target = vd['target_pc']

        # Baseline 1: per-FDI mean
        cd_mean = chamfer_torch(v_target, fdi_mean[v_fdi])
        val_chamfers_mean.append(cd_mean)
        val_chamfers_per_fdi[v_fdi][2] += cd_mean

        # Baseline 2: K-NN with batched CD on device
        v_partial_t = torch.tensor(v_partial, dtype=torch.float32, device=device)  # (4096, 3)
        stack = fdi_partials_stack[v_fdi]  # (N, 4096, 3) on device
        cds = []
        for n in range(stack.shape[0]):
            d = torch.cdist(v_partial_t.unsqueeze(0), stack[n].unsqueeze(0)).squeeze(0)
            cd = d.min(dim=1).values.mean() + d.min(dim=0).values.mean()
            cds.append((cd.item(), n))
        cds.sort()
        topk = cds[:K]
        topk_targets = torch.stack([torch.tensor(train_targets[train_by_fdi[v_fdi][n]], dtype=torch.float32) for _, n in topk])
        knn_pred = topk_targets.mean(dim=0).numpy()
        cd_knn = chamfer_torch(v_target, knn_pred)
        val_chamfers_knn.append(cd_knn)
        val_chamfers_per_fdi[v_fdi][0] += cd_knn
        val_chamfers_per_fdi[v_fdi][1] += 1

        if (i + 1) % 50 == 0:
            mn = np.mean(val_chamfers_mean) * 1000
            kn = np.mean(val_chamfers_knn) * 1000
            print(f"  [{i+1}/{len(val_files)}] mean CD (×1e-3) — per-FDI-mean: {mn:.2f}, {K}-NN: {kn:.2f}")

    # Report
    print(f"\n=== Final (val) ===")
    print(f"Per-FDI mean target:")
    print(f"  mean Chamfer (×1e-3): {np.mean(val_chamfers_mean)*1000:.2f} ± {np.std(val_chamfers_mean)*1000:.2f}")
    print(f"{K}-NN retrieval:")
    print(f"  mean Chamfer (×1e-3): {np.mean(val_chamfers_knn)*1000:.2f} ± {np.std(val_chamfers_knn)*1000:.2f}")
    print()
    print(f"=== Per-FDI breakdown (mean of {K}-NN) ===")
    print(f"  {'FDI':4s} {'n':4s} {'mean CD (×1e-3)':16s}")
    for fdi in sorted(val_chamfers_per_fdi.keys()):
        s, c, _ = val_chamfers_per_fdi[fdi]
        if c > 0:
            print(f"  {fdi:4d} {c:4d} {s/c*1000:14.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='/Users/alf/Projects/AlfResearch/dental-crown-gen/data/v0_split.json')
    p.add_argument('--K', type=int, default=1)
    p.add_argument('--max-val', type=int, default=None)
    p.add_argument('--max-train-per-fdi', type=int, default=200)
    p.add_argument('--device', default='mps', choices=['cpu', 'mps', 'cuda'])
    args = p.parse_args()
    main(args.split, K=args.K, max_val=args.max_val, max_train_per_fdi=args.max_train_per_fdi, device=args.device)
