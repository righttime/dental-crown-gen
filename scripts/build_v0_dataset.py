#!/usr/bin/env python3
"""
Build v0 training dataset from 3DTeethSeg (MedShapeNetCore).

For each patient x jaw x tooth instance:
  - Sample point cloud of full arch (z_0 context)
  - Sample point cloud of partial arch (with one tooth removed)
  - Sample point cloud of the removed tooth (target)
  - Record FDI label, patient ID, jaw

Optimized: vectorized face filtering + custom area-weighted sampling
(no trimesh) — ~5-10x faster than v1.
"""
import os
import sys
import time
import argparse
from collections import OrderedDict
import numpy as np

# ==== Configuration ====
NPZ_PATH = "/Volumes/extSSD/dental-data/medshapenetcore/medshapenetcore_3DTeethSeg.npz"
OUT_DIR = "/Volumes/extSSD/dental-data/v0_dataset"
N_ARCH_POINTS = 4096
N_TARGET_POINTS = 4096
NORMALIZE = True
MIN_TOOTH_VERTICES = 200
EXCLUDE_WISDOM = True
COMPRESS = True

# ==== NPZ deserialization ====
class MyDict(OrderedDict):
    def __missing__(self, key):
        val = self[key] = MyDict()
        return val

import __main__
__main__.MyDict = MyDict


def load_dataset(npz_path):
    print(f"Loading {npz_path}...")
    t0 = time.time()
    data = np.load(npz_path, allow_pickle=True)
    ds = data['data'].item()  # top-level IS the patient dict
    print(f"  {len(ds)} patients loaded in {time.time()-t0:.1f}s")
    return ds


def face_areas(vertices, faces):
    """Compute per-face triangle area (vectorized)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def sample_surface(vertices, faces, areas, n_points, rng):
    """Area-weighted surface sampling (no trimesh)."""
    total = areas.sum()
    if total <= 0 or len(faces) == 0:
        # Degenerate; just sample vertices
        idx = rng.integers(0, len(vertices), n_points)
        return vertices[idx]
    probs = areas / total
    face_idx = rng.choice(len(faces), size=n_points, p=probs)
    u = rng.random(n_points).astype(np.float32)
    v = rng.random(n_points).astype(np.float32)
    mask = (u + v) > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - u - v
    p0 = vertices[faces[face_idx, 0]]
    p1 = vertices[faces[face_idx, 1]]
    p2 = vertices[faces[face_idx, 2]]
    return (u[:, None] * p0 + v[:, None] * p1 + w[:, None] * p2).astype(np.float32)


def class_faces_by_instance(faces, instances):
    """Return face_instance (F,) — instance_id if all 3 verts share non-zero inst, else 0."""
    fi = instances[faces]  # (F, 3)
    all_same = (fi[:, 0] == fi[:, 1]) & (fi[:, 1] == fi[:, 2])
    return np.where(all_same & (fi[:, 0] > 0), fi[:, 0], 0).astype(np.int32)


def extract_teeth_fast(vertices, faces, labels, instances):
    """Per-jaw: classify faces by instance, then per-tooth extract vertices + faces.

    Returns: dict inst_id -> {vertices, faces_remapped, fdi}
    """
    face_inst = class_faces_by_instance(faces, instances)
    teeth = {}
    for inst in np.unique(face_inst):
        if inst == 0:
            continue
        v_mask = instances == inst
        v_indices = np.where(v_mask)[0]
        f_local_indices = np.where(face_inst == inst)[0]
        f_global = faces[f_local_indices]
        idx_map = np.full(len(vertices), -1, dtype=np.int64)
        idx_map[v_indices] = np.arange(len(v_indices))
        teeth[int(inst)] = {
            'vertices': vertices[v_indices],
            'faces': idx_map[f_global],
            'fdi': int(labels[v_indices[0]]),
        }
    return teeth


def remove_tooth_fast(vertices, faces, instances, tooth_id, face_inst):
    """Return (partial_v, partial_f) with tooth_id removed."""
    keep_mask = instances != tooth_id
    keep_indices = np.where(keep_mask)[0]
    f_keep = face_inst != tooth_id
    f_local = faces[f_keep]
    idx_map = np.full(len(vertices), -1, dtype=np.int64)
    idx_map[keep_indices] = np.arange(len(keep_indices))
    return vertices[keep_indices], idx_map[f_local]


def normalize_pc(pc, centroid=None, scale=None):
    if centroid is None:
        centroid = pc.mean(axis=0)
    pc = pc - centroid
    if scale is None:
        scale = np.max(np.abs(pc)) + 1e-8
    return (pc / scale).astype(np.float32), centroid.astype(np.float32), np.float32(scale)


def process_patient(pid, pdata, dry_run=False):
    n_pairs = 0
    n_skipped = 0
    for jaw in ['upper', 'lower']:
        if jaw not in pdata.get('mesh', {}):
            continue
        m = pdata['mesh'][jaw]
        vertices = m['vertices'].astype(np.float32)
        faces = m['faces'].astype(np.int64)
        labels = m['labels']
        instances = m['instances']

        # Pre-compute
        face_inst = class_faces_by_instance(faces, instances)
        full_areas = face_areas(vertices, faces)

        # Sample full arch ONCE
        rng = np.random.default_rng(42)
        full_pc = sample_surface(vertices, faces, full_areas, N_ARCH_POINTS, rng)
        if NORMALIZE:
            full_pc, centroid, scale = normalize_pc(full_pc)

        teeth = extract_teeth_fast(vertices, faces, labels, instances)

        for tooth_id, tooth in teeth.items():
            fdi = tooth['fdi']
            if EXCLUDE_WISDOM and fdi in [18, 28, 38, 48]:
                continue
            if tooth['vertices'].shape[0] < MIN_TOOTH_VERTICES:
                n_skipped += 1
                continue

            # Target tooth sampling
            t_areas = face_areas(tooth['vertices'], tooth['faces'])
            target_pc = sample_surface(tooth['vertices'], tooth['faces'],
                                       t_areas, N_TARGET_POINTS, rng)

            # Partial arch
            pv, pf = remove_tooth_fast(vertices, faces, instances, tooth_id, face_inst)
            if pv.shape[0] < 100 or pf.shape[0] < 100:
                n_skipped += 1
                continue
            p_areas = face_areas(pv, pf)
            partial_pc = sample_surface(pv, pf, p_areas, N_ARCH_POINTS, rng)

            if NORMALIZE:
                partial_pc, _, _ = normalize_pc(partial_pc, centroid, scale)
                target_pc, _, _ = normalize_pc(target_pc, centroid, scale)

            out_name = f"{pid}_{jaw}_fdi{fdi:02d}_inst{tooth_id}.npz"
            out_path = os.path.join(OUT_DIR, out_name)
            if not dry_run:
                if COMPRESS:
                    np.savez_compressed(
                        out_path,
                        full_pc=full_pc,
                        partial_pc=partial_pc,
                        target_pc=target_pc,
                        fdi=np.int16(fdi),
                        tooth_id=np.int16(tooth_id),
                        jaw=jaw,
                        patient_id=pid,
                        centroid=centroid,
                        scale=scale,
                    )
                else:
                    np.savez(
                        out_path,
                        full_pc=full_pc,
                        partial_pc=partial_pc,
                        target_pc=target_pc,
                        fdi=np.int16(fdi),
                        tooth_id=np.int16(tooth_id),
                        jaw=jaw,
                        patient_id=pid,
                        centroid=centroid,
                        scale=scale,
                    )
            n_pairs += 1
    return n_pairs, n_skipped


def main(limit=None, dry_run=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_dataset(NPZ_PATH)

    patients = list(ds.keys())
    if limit:
        patients = patients[:limit]

    n_pairs_total = 0
    n_skipped_total = 0
    t0 = time.time()

    for i, pid in enumerate(patients):
        try:
            n_pairs, n_skipped = process_patient(pid, ds[pid], dry_run=dry_run)
        except Exception as e:
            print(f"  ERR {pid}: {e}")
            continue
        n_pairs_total += n_pairs
        n_skipped_total += n_skipped

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(patients) - i - 1) / rate
            print(f"  [{i+1}/{len(patients)}] {n_pairs_total} pairs, "
                  f"{n_skipped_total} skipped, {rate:.1f} pt/s, "
                  f"ETA {eta/60:.1f} min")

    elapsed = time.time() - t0
    print(f"\n=== Done ===")
    print(f"  {n_pairs_total} pairs in {OUT_DIR}")
    print(f"  {n_skipped_total} skipped (too small or no FDI)")
    print(f"  {elapsed/60:.1f} min total")
    if not dry_run:
        files = [f for f in os.listdir(OUT_DIR) if f.endswith('.npz')]
        total_size = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in files)
        print(f"  {len(files)} files, {total_size/1024**3:.2f} GB total")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    main(limit=args.limit, dry_run=args.dry_run)
