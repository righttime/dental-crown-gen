# Dental Crown / Bridge 3D Model Generation 🦷

**Project:** Generate 3D-printable dental crown and bridge models from intra-oral scans, using deep learning.

**Started:** 2026-06-05 (HK + Alf)
**Status:** 🟡 Research phase — long-term multi-agent

---

## 🎯 Problem

Given an intra-oral 3D scan of a patient's dentition (with one or more missing teeth), generate a 3D-printable model of the missing crown(s) or bridge(s), with:
- **Inner surface** that fits the prepared tooth (margin)
- **Outer surface** that matches adjacent + opposing teeth (occlusion + contacts)
- Output as a watertight digital mesh (STL/OBJ/PLY)

## 🧩 Sub-tasks

1. **Tooth detection / segmentation** — Identify which teeth are missing from the scan
2. **Margin analysis** — Detect the prep margin (boundary between prepared tooth and gum)
3. **Inner surface design** — Generate the intaglio surface that fits on the prepared tooth
4. **Outer surface design** — Generate the occlusal + buccal surface, conditioned on opposing + adjacent teeth
5. **Mesh output** — Watertight mesh suitable for 3D printing

## 🧠 Hypotheses (initial)

- **H1** — 2-stage (segmentation + generation) outperforms end-to-end for missing-tooth detection
- **H2** — Diffusion on point clouds > mesh-based VAE for surface generation
- **H3** — Conditioning on opposing + adjacent teeth improves outer surface quality
- **H4** — Implicit SDF representation > explicit mesh for high-quality surfaces
- **H5** — Synthetic data from existing CAD libraries can bootstrap training (data is currently zero)

## 📊 Evaluation metrics (proposed)

- Segmentation: IoU, Dice coefficient
- Mesh quality: Chamfer distance, Hausdorff distance
- Clinical fit (later): margin fit error (μm)

## 🛠 Architecture (TBD)

- **Training:** Workstation (M4 Mac mini is for inference, not training 3D diffusion)
- **Inference:** TBD — possibly local with model compression
- **Data:** Currently zero — must be acquired (public dental datasets, synthetic CAD, or scraped)

## 🦉 Team

- **Alf** (main) — orchestration, memory
- **Scholar** (🦉) — long-horizon research, paper reading, weekly digest
- **Red** (🔥) — engineering, code, training pipelines
- **Mauve** (🎨) — visualization, diagrams, design

## 📁 Structure

```
dental-crown-gen/
├── README.md          (this)
├── papers/            (paper notes, summaries, links)
├── data/              (data acquisition notes, dataset references)
├── code/              (implementation, when ready)
└── docs/              (research design, weekly digests)
```

## 📜 Workflow

1. Scholar reads papers → weekly digest
2. Data acquisition plan (datasets, synthetic, scraping)
3. Baseline implementations (point cloud completion, segmentation)
4. Custom model dev (crown-specific)
5. Mesh output + clinical validation

---
*Started 2026-06-05. Maintained by Alf + Scholar + Red + Mauve.*
