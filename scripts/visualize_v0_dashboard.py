#!/usr/bin/env python3
"""v0 dataset dashboard: FDI distribution + 8 diverse samples."""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

DATA_DIR = "/Volumes/extSSD/dental-data/v0_dataset"
OUT = "/Users/alf/Projects/AlfResearch/dental-crown-gen/docs/figures/v0_dashboard.png"
N_SAMPLES = 8
N_POINTS = 1024


def downsample(pc, n):
    if pc.shape[0] <= n:
        return pc
    return pc[np.random.choice(pc.shape[0], n, replace=False)]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"Found {len(files)} files in {DATA_DIR}")

    # ---- Pass 1: FDI distribution (sample 5k) ----
    fdi_counts = {}
    for fp in files[:5000]:
        d = np.load(fp)
        fdi = int(d['fdi'])
        fdi_counts[fdi] = fdi_counts.get(fdi, 0) + 1
    scale = len(files) / 5000
    fdi_counts_full = {k: int(v * scale) for k, v in fdi_counts.items()}
    fdis = sorted(fdi_counts_full.keys())
    counts = [fdi_counts_full[f] for f in fdis]

    # ---- Pass 2: pick 8 diverse samples ----
    targets = [
        ('upper', 11), ('upper', 14), ('upper', 16), ('upper', 21),
        ('lower', 31), ('lower', 34), ('lower', 36), ('lower', 41),
    ]
    chosen = []
    for jaw, fdi_t in targets:
        for fp in files:
            d = np.load(fp)
            if str(d['jaw']) == jaw and int(d['fdi']) == fdi_t:
                chosen.append(fp)
                break

    # ---- Render ----
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 1, height_ratios=[1, 3], hspace=0.25)

    # Top: histogram
    ax_h = fig.add_subplot(gs[0, 0])
    colors = ['#3b82f6' if 11 <= f <= 18 or 41 <= f <= 48 else '#f59e0b' for f in fdis]
    ax_h.bar(fdis, counts, color=colors, edgecolor='black', linewidth=0.5)
    ax_h.set_xticks(fdis)
    ax_h.set_xlabel('FDI tooth number')
    ax_h.set_ylabel('Approx pair count')
    ax_h.set_title(f'v0 dataset FDI distribution — {len(files):,} pairs, {len(fdis)} tooth types',
                   fontsize=12)
    ax_h.grid(axis='y', alpha=0.3)
    ax_h.legend(handles=[
        Patch(facecolor='#3b82f6', label='left side (1x, 3x)'),
        Patch(facecolor='#f59e0b', label='right side (2x, 4x)'),
    ], loc='upper right')

    # Bottom: 2 rows × 4 samples = 8 samples × 3 panels = 24 subplots
    gs2 = gs[1, 0].subgridspec(2, 12, hspace=0.15, wspace=0.1)
    for idx, fp in enumerate(chosen):
        d = np.load(fp)
        full_pc = downsample(d['full_pc'], N_POINTS)
        partial_pc = downsample(d['partial_pc'], N_POINTS)
        target_pc = downsample(d['target_pc'], N_POINTS)
        fdi = int(d['fdi'])
        jaw = str(d['jaw'])
        pid = str(d['patient_id'])[:6]

        all_pts = np.concatenate([full_pc, partial_pc, target_pc])
        lim = np.max(np.abs(all_pts)) * 1.05

        row = idx // 4
        col = idx % 4
        for j, (lbl, pc, color) in enumerate([
            ('F', full_pc, 'steelblue'),
            ('P', partial_pc, 'gray'),
            ('T', target_pc, 'crimson'),
        ]):
            ax = fig.add_subplot(gs2[row, col * 3 + j], projection='3d')
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], color=color, s=1, alpha=0.85)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_box_aspect((1, 1, 1))
            ax.set_title(f"{lbl} · FDI {fdi}", fontsize=8)
        # Add patient + jaw annotation to the left of the row
        ax.text2D(-0.05, 0.5, f"{pid}\n{jaw}",
                  transform=ax.transAxes, fontsize=8,
                  ha='right', va='center', color='gray',
                  rotation=90 if row == 0 else 270)

    fig.suptitle('v0 dataset — diverse FDI samples (F=full arch, P=partial w/o target, T=target tooth)',
                 fontsize=14, y=0.995)
    plt.savefig(OUT, dpi=100, bbox_inches='tight', facecolor='white')
    print(f"Saved {OUT}")
    print(f"Size: {os.path.getsize(OUT)/1024:.1f} KB")


if __name__ == "__main__":
    main()
