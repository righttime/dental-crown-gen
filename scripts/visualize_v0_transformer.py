#!/usr/bin/env python3
"""v0 Transformer predictions visualization (n_points=1024)."""
import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v0_transformer import CrownGenTransformer, V0Dataset, chamfer_torch

def downsample(pc, n):
    if pc.shape[0] <= n:
        return pc
    return pc[np.random.choice(pc.shape[0], n, replace=False)]


def main(model_path, split_path, out_path, n_samples=6, device='mps', n_points=1024):
    model = CrownGenTransformer(n_fdi=28, n_points=n_points).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {model_path} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    with open(split_path) as f:
        s = json.load(f)
    val_files = s['val_files']

    targets = [
        ('upper', 11), ('upper', 16), ('upper', 26),
        ('lower', 31), ('lower', 36), ('lower', 46),
    ]
    chosen = []
    for jaw, fdi_t in targets:
        for fp in val_files:
            d = np.load(fp)
            if str(d['jaw']) == jaw and int(d['fdi']) == fdi_t:
                chosen.append(fp)
                break

    fdi_to_idx = {fdi: i for i, fdi in enumerate(
        list(range(11, 18)) + list(range(21, 28)) + list(range(31, 38)) + list(range(41, 48))
    )}

    fig = plt.figure(figsize=(4 * 3.2, len(chosen) * 3.0))
    for idx, fp in enumerate(chosen):
        d = np.load(fp)
        # Downsample to 1024 (model input/output)
        partial = d['partial_pc'].astype(np.float32)
        target_full = d['target_pc'].astype(np.float32)
        # Take first n_points via random sample
        rng = np.random.default_rng(idx)
        partial_ds = partial[rng.choice(partial.shape[0], n_points, replace=False)]
        target = target_full[rng.choice(target_full.shape[0], n_points, replace=False)]
        fdi = int(d['fdi'])
        fdi_idx = fdi_to_idx[fdi]
        jaw = str(d['jaw'])
        pid = str(d['patient_id'])[:8]

        with torch.no_grad():
            partial_t = torch.tensor(partial_ds, dtype=torch.float32, device=device).unsqueeze(0)
            fdi_t = torch.tensor([fdi_idx], dtype=torch.long, device=device)
            pred = model(partial_t, fdi_t).cpu().numpy()[0]
        # Chamfer (on 1024 points)
        cd = chamfer_torch(
            torch.tensor(pred, dtype=torch.float32).unsqueeze(0),
            torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        ).item()

        all_pts = np.concatenate([partial_ds, target, pred])
        lim = np.max(np.abs(all_pts)) * 1.05

        for j, (title, pc, color) in enumerate([
            ('PARTIAL', partial_ds, 'gray'),
            ('GT', target, 'crimson'),
            ('PRED (Transformer)', pred, 'dodgerblue'),
        ]):
            ax = fig.add_subplot(len(chosen), 3, idx * 3 + j + 1, projection='3d')
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], color=color, s=2, alpha=0.85)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_box_aspect((1, 1, 1))
            if idx == 0:
                ax.set_title(title, fontsize=10, fontweight='bold' if j == 2 else 'normal')
            if j == 0:
                ax.text2D(-0.05, 0.5, f"{pid} {jaw}\nFDI {fdi}\nCD: {cd*1000:.2f}",
                          transform=ax.transAxes, fontsize=9, ha='right', va='center')

    fig.suptitle('v0 Transformer predictions (val CD 102 × 1e-3, 1024 pts) — GT vs PRED', fontsize=13, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='white')
    print(f"Saved {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/Users/alf/Projects/AlfResearch/dental-crown-gen/models/v0_transformer/best.pt')
    p.add_argument('--split', default='/Users/alf/Projects/AlfResearch/dental-crown-gen/data/v0_split.json')
    p.add_argument('--out', default='/Users/alf/Projects/AlfResearch/dental-crown-gen/docs/figures/v0_transformer.png')
    p.add_argument('--n-samples', type=int, default=6)
    p.add_argument('--n-points', type=int, default=1024)
    p.add_argument('--device', default='mps')
    args = p.parse_args()
    main(args.model, args.split, args.out, args.n_samples, args.device, args.n_points)
