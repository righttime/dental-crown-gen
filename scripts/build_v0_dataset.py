#!/usr/bin/env python3
"""
Build v0 training dataset from 3DTeethSeg (MedShapeNetCore).

For each patient × jaw × tooth instance:
  - Sample point cloud of full arch (z_0 context)
  - Sample point cloud of partial arch (with one tooth removed)
  - Sample point cloud of the removed tooth (target)
  - Record FDI label, patient ID, jaw

Output: compressed NPZ files in /Volumes/extSSD/dental-data/v0_dataset/
"""
import os
import sys
import time
import argparse
from collections import OrderedDict
import numpy as np
import trimesh

# ==== Configuration ====
NPZ_PATH = "/Volumes/extSSD/dental-data/medshapenetcore/medshapenetcore_3DTeethSeg.npz"
OUT_DIR = "/Volumes/extSSD/dental-data/v0_dataset"
N_ARCH_POINTS = 4096       # points per arch (full + partial)
N_TARGET_POINTS = 4096     # points per isolated tooth
NORMALIZE = True           # center & scale to unit max-distance
MIN_TOOTH_VERTICES = 200   # filter tiny/artifactual teeth
EXCLUDE_WISDOM = True      # skip FDI 18, 28, 38, 48 for v0
COMPRESS = True            # savez_compressed vs savez

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
    # The top-level dict IS the patient dict (900 patients),
    # not nested under '3DTeethSeg'.
    ds = data['data'].item()
    print(f"  {len(ds)} patients loaded in {time.time()-t0:.1f}s")
    return ds


def extract_tooth_submeshes(vertices, faces, labels, instances):
    """Extract individual tooth submeshes from a full arch mesh.

    Returns: dict instance_id -> {vertices, faces, fdi}
    """
    teeth = {}
    for inst_id in np.unique(instances):
        if inst_id == 0:
            continue  # skip gingiva
        v_indices = np.where(instances == inst_id)[0]
        v_set = set(v_indices.tolist())
        f_mask = np.array([all(f in v_set for f in face) for face in faces])
        if not f_mask.any():
            continue
        f_local = faces[f_mask]
        # Remap face indices to local
        idx_map = {g: l for l, g in enumerate(v_indices)}
        f_local = np.vectorize(idx_map.get)(f_local)
        teeth[int(inst_id)] = {
            'vertices': vertices[v_indices],
            'faces': f_local,
            'fdi': int(labels[v_indices[0]]),
        }
    return teeth


def sample_surface(vertices, faces, n_points, seed=0):
    """Sample n_points uniformly from mesh surface."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    points, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    return points.astype(np.float32)


def remove_tooth_from_arch(vertices, faces, instances, tooth_id):
    """Return partial-arch vertices and faces with `tooth_id` instance removed."""
    keep_indices = np.where(instances != tooth_id)[0]
    keep_set = set(keep_indices.tolist())
    f_mask = np.array([all(f in keep_set for f in face) for face in faces])
    f_local = faces[f_mask]
    idx_map = {g: l for l, g in enumerate(keep_indices)}
    f_local = np.vectorize(idx_map.get)(f_local)
    return vertices[keep_indices], f_local


def normalize_pc(pc, centroid=None, scale=None):
    """Center at mean (or given centroid), scale to unit max-distance."""
    if centroid is None:
        centroid = pc.mean(axis=0)
    pc = pc - centroid
    if scale is None:
        scale = np.max(np.abs(pc)) + 1e-8
    pc = pc / scale
    return pc.astype(np.float32), centroid.astype(np.float32), np.float32(scale)


def process_patient(pid, pdata, dry_run=False, writer=None):
    """Process one patient; return (n_pairs, n_skipped) tuple."""
    n_pairs = 0
    n_skipped = 0
    for jaw in ['upper', 'lower']:
        if jaw not in pdata.get('mesh', {}):
            continue
        m = pdata['mesh'][jaw]
        vertices = m['vertices'].astype(np.float32)
        faces = m['faces']
        labels = m['labels']
        instances = m['instances']

        teeth = extract_tooth_submeshes(vertices, faces, labels, instances)

        for tooth_id, tooth in teeth.items():
            fdi = tooth['fdi']

            if EXCLUDE_WISDOM and fdi in [18, 28, 38, 48]:
                continue
            if tooth['vertices'].shape[0] < MIN_TOOTH_VERTICES:
                n_skipped += 1
                continue

            # 1) Target tooth point cloud
            target_pc = sample_surface(tooth['vertices'], tooth['faces'],
                                       N_TARGET_POINTS, seed=fdi)

            # 2) Partial arch (with this tooth removed)
            pv, pf = remove_tooth_from_arch(vertices, faces, instances, tooth_id)
            if pv.shape[0] < 100 or pf.shape[0] < 100:
                n_skipped += 1
                continue
            partial_pc = sample_surface(pv, pf, N_ARCH_POINTS, seed=fdi + 1)

            # 3) Full arch point cloud
            full_pc = sample_surface(vertices, faces, N_ARCH_POINTS, seed=fdi + 2)

            # Normalize using full arch
            if NORMALIZE:
                full_pc, centroid, scale = normalize_pc(full_pc)
                partial_pc, _, _ = normalize_pc(partial_pc, centroid, scale)
                target_pc, _, _ = normalize_pc(target_pc, centroid, scale)
            else:
                centroid = np.zeros(3, dtype=np.float32)
                scale = np.float32(1.0)

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

        if (i + 1) % 50 == 0:
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
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of patients (for testing)")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't write files; just count")
    args = p.parse_args()
    main(limit=args.limit, dry_run=args.dry_run)
