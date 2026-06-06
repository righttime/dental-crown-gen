#!/usr/bin/env python3
"""Visualize 4 random v0 dataset samples as 3D scatter plots.

For each sample, shows 3 columns:
  - full arch (with all teeth)
  - partial arch (target tooth removed) — gray
  - target tooth — red

Output: docs/figures/v0_samples.png
"""
import os
import glob
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_DIR = "/Volumes/extSSD/dental-data/v0_dataset"
OUT = "/Users/alf/Projects/AlfResearch/dental-crown-gen/docs/figures/v0_samples.png"
N_SAMPLES = 4
N_POINTS = 1024  # sub-sample for visualization


def downsample(pc, n):
    if pc.shape[0] <= n:
        return pc
    idx = np.random.choice(pc.shape[0], n, replace=False)
    return pc[idx]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"Found {len(files)} files in {DATA_DIR}")
    random.seed(0)
    samples = random.sample(files, N_SAMPLES)

    fig = plt.figure(figsize=(4 * 3.2, 4 * 3.0))
    for i, fp in enumerate(samples):
        d = np.load(fp)
        full_pc = downsample(d['full_pc'], N_POINTS)
        partial_pc = downsample(d['partial_pc'], N_POINTS)
        target_pc = downsample(d['target_pc'], N_POINTS)
        fdi = int(d['fdi'])
        jaw = str(d['jaw'])
        pid = str(d['patient_id'])

        # Compute consistent axes across the 3 subplots
        all_pts = np.concatenate([full_pc, partial_pc, target_pc])
        lim = np.max(np.abs(all_pts)) * 1.05

        for j, (title, pc, color) in enumerate([
            ('FULL (all teeth)', full_pc, 'steelblue'),
            ('PARTIAL (target removed)', partial_pc, 'gray'),
            (f'TARGET tooth (FDI {fdi})', target_pc, 'crimson'),
        ]):
            ax = fig.add_subplot(N_SAMPLES, 3, i * 3 + j + 1, projection='3d')
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=color, s=1.5, alpha=0.85)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.set_title(f"{title}\n{pid} {jaw}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_box_aspect((1, 1, 1))

    plt.tight_layout()
    plt.savefig(OUT, dpi=110, bbox_inches='tight', facecolor='white')
    print(f"Saved {OUT}")
    print(f"Size: {os.path.getsize(OUT)/1024:.1f} KB")


if __name__ == "__main__":
    main()
