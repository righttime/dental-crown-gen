# Project Status

**Last updated:** 2026-06-05 (kickoff + first paper read)

## Current state
- [x] Concept defined (intra-oral scan → printable crown/bridge mesh)
- [x] Sub-tasks identified (5 steps)
- [x] Hypotheses drafted (5)
- [x] Architecture draft (training = workstation, inference = TBD)
- [x] Literature survey (Scholar — 1 paper read, see `papers/001-…`)
- [ ] Data acquisition plan
- [ ] Baseline implementation
- [ ] Custom model dev
- [ ] Mesh output validation
- [ ] Clinical fit evaluation

## Open questions
- Data acquisition: public datasets vs synthetic vs scraping?
- Compute: cloud GPU (Lambda, RunPod) vs university cluster?
- 3D format conversion pipeline (OBJ/STL/PLY all in?)

## Scholar weekly digest

- **Week of 2026-06-05:** read paper 001 — *3DTeethSeg'22: 3D Teeth Scan Segmentation and Labeling Challenge* (Ben-Hamadou et al., MICCAI 2022 challenge). Key insight: largest public IOS dataset (1,800 scans / 900 patients / 23,999 FDI-labeled teeth) is now downloadable, and every winning method is 2-stage (centroid detection → per-tooth segmentation) — strong empirical support for **H1**. IGIP team's dental-arch-curve post-processing (Bezier fit through centroids used to correct FDI labels) is a direct analogue of the inductive bias we need for **H3** (conditioning on global arch context for outer-surface generation). Concrete action: download the dataset this week, use it as the public-data backbone for sub-task 1. Note in `papers/001-3dteethseg22.md`.
