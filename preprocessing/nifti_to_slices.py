import os
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
from sklearn.model_selection import KFold

def extract_and_split(raw_root: str, out_dir: str, splits_path: str, target_size=(224, 224), n_splits=5):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)

    raw_path = Path(raw_root)
    print(f"Traversing '{raw_path.resolve()}'...")

    # Discover all patient directories
    all_nii = list(raw_path.rglob("*.nii.gz")) + list(raw_path.rglob("*.nii"))
    print(f"Total NIfTI volumes found: {len(all_nii)}")

    # Group files by patient identifier (e.g. TCGA-BC-4073)
    patient_buckets = {}
    for f in all_nii:
        parts = f.parts
        # Identify TCGA patient folder in the directory path
        pid = None
        for part in parts:
            if part.startswith("TCGA-"):
                pid = part
                break
        if pid is None:
            pid = f.parent.name

        if pid not in patient_buckets:
            patient_buckets[pid] = []
        patient_buckets[pid].append(f)

    manifest = []
    processed_pids = []

    for pid, files in patient_buckets.items():
        # Identify scan volumes vs mask volumes
        masks = [f for f in files if any(k in f.name.lower() for k in ["rater", "seg", "tumor", "mask"])]
        # Prefer arterial ('art') or portal-venous ('pv') or pre-contrast ('pre')
        scans = [f for f in files if f not in masks]

        if not scans or not masks:
            # If all files had 'seg' in their parent path, discriminate by file name:
            masks = [f for f in files if "rater" in f.name.lower() or "mask" in f.name.lower() or "tumor" in f.name.lower()]
            scans = [f for f in files if f not in masks]

        if not scans or not masks:
            continue

        # Choose the best scan (prioritize arterial phase, then any valid volume)
        scan_choice = scans[0]
        for s in scans:
            if "art" in s.name.lower() or "pv" in s.name.lower():
                scan_choice = s
                break

        # Choose the primary tumor mask
        mask_choice = masks[0]
        for m in masks:
            if "tumor" in m.name.lower() or "rater1" in m.name.lower():
                mask_choice = m
                break

        try:
            img_vol = nib.load(str(scan_choice)).get_fdata()
            mask_vol = nib.load(str(mask_choice)).get_fdata()
        except Exception as err:
            print(f"Skipping {pid} due to read error: {err}")
            continue

        # Check alignment
        if img_vol.shape[:2] != mask_vol.shape[:2]:
            continue

        num_slices = min(img_vol.shape[-1], mask_vol.shape[-1])
        slice_counter = 0

        for z in range(num_slices):
            slice_img = img_vol[..., z]
            slice_mask = mask_vol[..., z]

            # Intensity normalization [0, 1]
            s_min, s_max = np.min(slice_img), np.max(slice_img)
            if s_max > s_min:
                slice_img = (slice_img - s_min) / (s_max - s_min)
            else:
                slice_img = np.zeros_like(slice_img)

            slice_img = cv2.resize(slice_img, target_size, interpolation=cv2.INTER_LINEAR)
            slice_mask = cv2.resize((slice_mask > 0).astype(np.float32), target_size, interpolation=cv2.INTER_NEAREST)

            has_tumor = bool(np.sum(slice_mask) > 0)

            file_name = f"{pid}_slice_{z:03d}.npz"
            save_path = os.path.join(out_dir, file_name)

            np.savez_compressed(
                save_path,
                image=slice_img.astype(np.float32),
                mask=slice_mask.astype(np.uint8)
            )

            manifest.append({
                "patient_id": pid,
                "file_name": file_name,
                "has_tumor": has_tumor
            })
            slice_counter += 1

        if slice_counter > 0:
            processed_pids.append(pid)

    if not processed_pids:
        raise RuntimeError("No matching volume and mask pairs could be aligned. Please run: ls data/raw/nifti_and_segms/*")

    # Leak-free patient-wise cross validation split
    kf = KFold(n_splits=min(n_splits, len(processed_pids)), shuffle=True, random_state=42)
    pid_arr = np.array(processed_pids)
    folds = {}

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(pid_arr)):
        train_pids = set(pid_arr[train_idx])
        val_pids = set(pid_arr[val_idx])

        train_files = [m["file_name"] for m in manifest if m["patient_id"] in train_pids]
        val_files = [m["file_name"] for m in manifest if m["patient_id"] in val_pids]

        folds[f"fold_{fold_idx}"] = {
            "train": train_files,
            "val": val_files
        }

    with open(splits_path, "w") as f:
        json.dump(folds, f, indent=2)

    print(f"\nDone. Extracted {len(manifest)} slices from {len(processed_pids)} patients.")
    print(f"Splits saved to {splits_path}.")

if __name__ == "__main__":
    extract_and_split(
        raw_root="data/raw/nifti_and_segms",
        out_dir="data/processed_2d",
        splits_path="data/splits/patient_folds.json"
    )