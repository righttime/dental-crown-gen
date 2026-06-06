#!/usr/bin/env python3
"""v0 dataset — interactive 3D visualization with plotly.

Drop down to switch between 8 diverse samples.
Toggle visibility of full/partial/target via legend.

Output: docs/figures/v0_interactive.html
"""
import os
import glob
import numpy as np
import plotly.graph_objects as go

DATA_DIR = "/Volumes/extSSD/dental-data/v0_dataset"
OUT = "/Users/alf/Projects/AlfResearch/dental-crown-gen/docs/figures/v0_interactive.html"
N_POINTS = 1024


def downsample(pc, n):
    if pc.shape[0] <= n:
        return pc
    return pc[np.random.choice(pc.shape[0], n, replace=False)]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"Found {len(files)} files")

    # Pick 8 diverse samples (4 upper, 4 lower, different FDI types)
    targets = [
        ('upper', 11), ('upper', 14), ('upper', 16), ('upper', 21),
        ('lower', 31), ('lower', 34), ('lower', 36), ('lower', 41),
    ]
    chosen = []
    for jaw, fdi_t in targets:
        for fp in files:
            d = np.load(fp)
            if str(d['jaw']) == jaw and int(d['fdi']) == fdi_t:
                chosen.append((fp, dict(d)))
                break

    fig = go.Figure()
    sample_labels = []

    for sample_idx, (fp, d) in enumerate(chosen):
        full_pc = downsample(d['full_pc'], N_POINTS)
        partial_pc = downsample(d['partial_pc'], N_POINTS)
        target_pc = downsample(d['target_pc'], N_POINTS)
        fdi = int(d['fdi'])
        jaw = str(d['jaw'])
        pid = str(d['patient_id'])
        label = f"{pid} {jaw} FDI {fdi}"
        sample_labels.append(label)

        # All 3 traces belong to this sample, all start visible
        # The dropdown will show/hide groups of 3
        fig.add_trace(go.Scatter3d(
            x=full_pc[:, 0], y=full_pc[:, 1], z=full_pc[:, 2],
            mode='markers',
            marker=dict(size=2, color='steelblue', opacity=0.85),
            name=f'full arch',
            legendgroup=f's{sample_idx}',
            showlegend=True,
            visible=(sample_idx == 0),
            hovertemplate='full arch<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>',
        ))
        fig.add_trace(go.Scatter3d(
            x=partial_pc[:, 0], y=partial_pc[:, 1], z=partial_pc[:, 2],
            mode='markers',
            marker=dict(size=2, color='gray', opacity=0.85),
            name=f'partial (target removed)',
            legendgroup=f's{sample_idx}',
            showlegend=True,
            visible=(sample_idx == 0),
            hovertemplate='partial<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>',
        ))
        fig.add_trace(go.Scatter3d(
            x=target_pc[:, 0], y=target_pc[:, 1], z=target_pc[:, 2],
            mode='markers',
            marker=dict(size=3, color='crimson', opacity=0.9),
            name=f'target tooth (FDI {fdi})',
            legendgroup=f's{sample_idx}',
            showlegend=True,
            visible=(sample_idx == 0),
            hovertemplate=f'target FDI {fdi}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>',
        ))

    # Build dropdown: each option toggles visibility of 3 traces (per sample)
    n_traces = len(chosen) * 3
    buttons = []
    for i, label in enumerate(sample_labels):
        visible = [False] * n_traces
        for j in range(3):
            visible[i * 3 + j] = True
        buttons.append(dict(
            label=label,
            method='update',
            args=[{'visible': visible},
                  {'title': f'v0 sample — {label}'}],
        ))

    fig.update_layout(
        title=f'v0 dataset — interactive 3D point clouds ({len(chosen)} samples)',
        scene=dict(
            xaxis=dict(title='x', showticklabels=False),
            yaxis=dict(title='y', showticklabels=False),
            zaxis=dict(title='z', showticklabels=False),
            aspectmode='cube',
        ),
        updatemenus=[dict(
            buttons=buttons,
            direction='down',
            showactive=True,
            x=0.0, xanchor='left',
            y=1.12, yanchor='top',
        )],
        margin=dict(l=0, r=0, b=0, t=80),
        height=750,
        legend=dict(x=0.85, y=0.95, bgcolor='rgba(255,255,255,0.7)'),
    )

    fig.write_html(OUT, include_plotlyjs='cdn', full_html=True)
    print(f"Saved {OUT}")
    print(f"Size: {os.path.getsize(OUT)/1024:.1f} KB")


if __name__ == "__main__":
    main()
