#!/usr/bin/env python3
"""Split v0 dataset into train/val by patient (no patient leakage)."""
import os
import glob
import random
import json

DATA_DIR = "/Volumes/extSSD/dental-data/v0_dataset"
OUT = "/Users/alf/Projects/AlfResearch/dental-crown-gen/data/v0_split.json"
TRAIN_RATIO = 0.9
SEED = 42


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"Total files: {len(files)}")

    # Extract patient_id from filename: <patient>_<jaw>_fdi<N>_inst<M>.npz
    patients = set()
    for f in files:
        name = os.path.basename(f)
        pid = name.split("_")[0]
        patients.add(pid)
    patients = sorted(patients)
    print(f"Unique patients: {len(patients)}")

    random.seed(SEED)
    random.shuffle(patients)
    n_train = int(len(patients) * TRAIN_RATIO)
    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:])
    print(f"Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")

    train_files = [f for f in files if os.path.basename(f).split("_")[0] in train_patients]
    val_files = [f for f in files if os.path.basename(f).split("_")[0] in val_patients]
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    # Sanity: no overlap
    train_pids = {os.path.basename(f).split("_")[0] for f in train_files}
    val_pids = {os.path.basename(f).split("_")[0] for f in val_files}
    assert not (train_pids & val_pids), "Patient leakage!"
    print("✓ No patient leakage")

    # Stats
    print("\n=== FDI distribution (val) ===")
    fdi_counts = {}
    for f in val_files:
        # FDI from filename: <patient>_<jaw>_fdi<N>_inst<M>.npz
        parts = os.path.basename(f).split("_")
        for p in parts:
            if p.startswith("fdi"):
                fdi = int(p[3:])
                fdi_counts[fdi] = fdi_counts.get(fdi, 0) + 1
                break
    for fdi in sorted(fdi_counts.keys()):
        print(f"  FDI {fdi:2d}: {fdi_counts[fdi]:3d}")

    # Save
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({
            'train_files': train_files,
            'val_files': val_files,
            'train_patients': sorted(train_patients),
            'val_patients': sorted(val_patients),
            'config': {
                'train_ratio': TRAIN_RATIO,
                'seed': SEED,
                'n_files_total': len(files),
                'n_files_train': len(train_files),
                'n_files_val': len(val_files),
            }
        }, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
