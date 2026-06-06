#!/usr/bin/env python3
"""v0 best.pt predictions visualization.

Loads best.pt (val CD 61.99, epoch 2) and renders side-by-side:
  - full arch | partial arch | GT target | predicted target

Per-sample Chamfer shown in title.
"""
import os
import json
import glob
import random
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v0_simple import CrownGenModel, V0Dataset, chamfer_torch


def downsample(pc, n=2048):
    if pc.shape[0] <= n:
        return pc
    return pc[np.random.choice(pc.shape[0], n, replace=False)]


def main(model_path, split_path, out_path, n_samples=6, device='mps'):
    # Load model
    model = CrownGenModel(n_fdi=28, n_points=4096).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {model_path} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    # Load val split
    with open(split_path) as f:
        s = json.load(f)
    val_files = s['val_files']
    print(f"Val files: {len(val_files)}")

    # Pick 6 diverse samples (one per jaw type, different FDI ranges)
    targets = [
        ('upper', 11), ('upper', 16), ('upper', 26),
        ('lower', 31), ('lower', 36), ('lower', 46),
    ]
    fdi_to_idx = {fdi: i for i, fdi in enumerate(
        list(range(11, 18)) + list(range(21, 28)) + list(range(31, 38)) + list(range(41, 48))
    )}
    chosen = []
    for jaw, fdi_t in targets:
        for fp in val_files:
            d = np.load(fp)
            if str(d['jaw']) == jaw and int(d['fdi']) == fdi_t:
                chosen.append(fp)
                break

    # Run model + visualize
    fig = plt.figure(figsize=(4 * 3.2, len(chosen) * 3.0))
    for idx, fp in enumerate(chosen):
        d = np.load(fp)
        partial = d['partial_pc'].astype(np.float32)
        target = d['target_pc'].astype(np.float32)
        fdi = int(d['fdi'])
        fdi_idx = fdi_to_idx[fdi]
        jaw = str(d['jaw'])
        pid = str(d['patient_id'])[:8]

        # Predict
        with torch.no_grad():
            partial_t = torch.tensor(partial, dtype=torch.float32, device=device).unsqueeze(0)
            fdi_t = torch.tensor([fdi_idx], dtype=torch.long, device=device)
            pred = model(partial_t, fdi_t).cpu().numpy()[0]  # (4096, 3)
        # Chamfer
        cd = chamfer_torch(
            torch.tensor(pred, dtype=torch.float32).unsqueeze(0),
            torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        ).item()

        # Sub-sample for plotting
        full_pc = downsample(partial, 1024)  # use partial as full visualization
        partial_v = downsample(partial, 1024)
        target_v = downsample(target, 1024)
        pred_v = downsample(pred, 1024)

        # Bounding box from all 4
        all_pts = np.concatenate([partial_v, target_v, pred_v])
        lim = np.max(np.abs(all_pts)) * 1.05

        for j, (title, pc, color) in enumerate([
            ('PARTIAL (no target)', partial_v, 'gray'),
            ('GT target', target_v, 'crimson'),
            ('PRED target', pred_v, 'dodgerblue'),
        ]):
            ax = fig.add_subplot(len(chosen), 3, idx * 3 + j + 1, projection='3d')
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], color=color, s=1, alpha=0.85)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_box_aspect((1, 1, 1))
            if idx == 0:
                ax.set_title(title, fontsize=10, fontweight='bold' if j == 2 else 'normal')
            if j == 0:
                ax.text2D(-0.05, 0.5, f"{pid} {jaw}\nFDI {fdi}\nCD: {cd*1000:.2f}",
                          transform=ax.transAxes, fontsize=9,
                          ha='right', va='center')

    fig.suptitle('v0 best.pt predictions (val CD 61.99 × 1e-3) — GT (crimson) vs PRED (dodgerblue)',
                 fontsize=13, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='white')
    print(f"Saved {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")

    # ALSO save centroid-aligned version (so we can see if shapes match when location is removed)
    out_path_aligned = out_path.replace('.png', '_aligned.png')
    fig2 = plt.figure(figsize=(4 * 3.2, len(chosen) * 3.0))
    for idx, fp in enumerate(chosen):
        d = np.load(fp)
        target = d['target_pc'].astype(np.float32)
        fdi = int(d['fdi'])
        fdi_idx = fdi_to_idx[fdi]
        jaw = str(d['jaw'])
        pid = str(d['patient_id'])[:8]

        # Re-predict
        partial = d['partial_pc'].astype(np.float32)
        with torch.no_grad():
            partial_t = torch.tensor(partial, dtype=torch.float32, device=device).unsqueeze(0)
            fdi_t = torch.tensor([fdi_idx], dtype=torch.long, device=device)
            pred = model(partial_t, fdi_t).cpu().numpy()[0]

        # Align: shift PRED to GT centroid
        gt_c = target.mean(axis=0)
        pred_c = pred.mean(axis=0)
        pred_aligned = pred - pred_c + gt_c

        target_v = downsample(target, 1024)
        pred_v = downsample(pred_aligned, 1024)
        all_pts = np.concatenate([target_v, pred_v])
        lim = np.max(np.abs(all_pts - all_pts.mean(axis=0))) * 1.5

        for j, (title, pc, color) in enumerate([
            ('GT target', target_v, 'crimson'),
            ('PRED (centroid-aligned)', pred_v, 'dodgerblue'),
            ('OVERLAY', None, None),  # placeholder
        ]):
            ax = fig2.add_subplot(len(chosen), 3, idx * 3 + j + 1, projection='3d')
            if j == 2:
                # overlay
                ax.scatter(target_v[:, 0], target_v[:, 1], target_v[:, 2], color='crimson', s=1, alpha=0.4, label='GT')
                ax.scatter(pred_v[:, 0], pred_v[:, 1], pred_v[:, 2], color='dodgerblue', s=1, alpha=0.4, label='PRED')
                ax.legend(loc='upper right', fontsize=7)
            else:
                ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], color=color, s=1.5, alpha=0.85)
            # Recenter for display
            mid = all_pts.mean(axis=0)
            ax.set_xlim(mid[0] - lim, mid[0] + lim)
            ax.set_ylim(mid[1] - lim, mid[1] + lim)
            ax.set_zlim(mid[2] - lim, mid[2] + lim)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_box_aspect((1, 1, 1))
            if idx == 0:
                ax.set_title(title, fontsize=10, fontweight='bold' if j == 1 else 'normal')
            if j == 0:
                ax.text2D(-0.05, 0.5, f"{pid} {jaw}\nFDI {fdi}", transform=ax.transAxes, fontsize=9, ha='right', va='center')
    fig2.suptitle('Centroid-aligned: GT (crimson) vs PRED (dodgerblue) — same location, compare shape', fontsize=13, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path_aligned, dpi=110, bbox_inches='tight', facecolor='white')
    print(f"Saved aligned: {out_path_aligned} ({os.path.getsize(out_path_aligned)/1024:.1f} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/Users/alf/Projects/AlfResearch/dental-crown-gen/models/v0_simple/best.pt')
    p.add_argument('--split', default='/Users/alf/Projects/AlfResearch/dental-crown-gen/data/v0_split.json')
    p.add_argument('--out', default='/Users/alf/Projects/AlfResearch/dental-crown-gen/docs/figures/v0_predictions.png')
    p.add_argument('--n-samples', type=int, default=6)
    p.add_argument('--device', default='mps')
    args = p.parse_args()
    main(args.model, args.split, args.out, args.n_samples, args.device)
