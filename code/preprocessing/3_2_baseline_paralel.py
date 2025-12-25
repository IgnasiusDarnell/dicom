#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import cv2
import pydicom
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut

# CONFIG
TARGET_SIZE = 1024

SPLIT_ROOT = "../../dataset/split_dataset"
BASELINE_ROOT = "../../dataset/baseline"

PNG_ROOT = os.path.join(BASELINE_ROOT, "PNG")
NPY_ROOT = os.path.join(BASELINE_ROOT, "NPY")
MANIFEST_ROOT = os.path.join(BASELINE_ROOT, "manifest")
INVERSION_LOG_PATH = os.path.join(MANIFEST_ROOT, "log_inversion.txt") 

for p in [PNG_ROOT, NPY_ROOT, MANIFEST_ROOT]:
    os.makedirs(p, exist_ok=True)

with open(INVERSION_LOG_PATH, "w") as f:
    f.write("# Baseline Inversion Log (List of Inverted Files)\n")

SPLITS = ["train", "val", "test"]
LABELS = {"TB": 1, "NonTB": 0}

# UTIL
def extract_2d(img):
    img = np.asarray(img)

    if img.ndim == 2:
        return img.astype(np.float32)

    #(H, W, C) -> ambil channel pertama
    if img.ndim == 3 and img.shape[-1] > 1:
        return img[..., 0].astype(np.float32)

    #(C, H, W) -> ambil slice pertama
    if img.ndim == 3 and img.shape[0] > 1:
        return img[0].astype(np.float32)

    #multi-frame (S, H, W)
    if img.ndim == 3:
        # pilih frame dengan variansi tertinggi (paling informatif)
        vars_ = np.var(img.reshape(img.shape[0], -1), axis=1)
        idx = int(np.argmax(vars_))
        return img[idx].astype(np.float32)

    # 4D (S, H, W, C) -> ambil frame & channel pertama
    if img.ndim == 4:
        return img[0, ..., 0].astype(np.float32)

    return np.squeeze(img).astype(np.float32)

def needs_inversion(ds, img):
    pi = getattr(ds, "PhotometricInterpretation", "").upper()

    if pi == "MONOCHROME1":
        return True
    if pi == "MONOCHROME2":
        return False

    # fallback anatomical heuristic
    mn, mx = img.min(), img.max()
    if mx <= mn:
        return False
    ie = (img - mn) / (mx - mn)

    fg = ie > 0.05
    if fg.sum() < 0.1 * ie.size:
        return False

    return ie[~fg].mean() > ie[fg].mean()


def normalize(img):
    mn, mx = img.min(), img.max()
    if mx > mn:
        return (img - mn) / (mx - mn)
    return np.zeros_like(img)

def resize_with_padding(img, target=1024):
    h, w = img.shape
    scale = min(target / h, target / w)

    if scale < 1.0:
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        img = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )
    else:
        new_h, new_w = h, w

    pad_top = (target - new_h) // 2
    pad_bottom = target - new_h - pad_top
    pad_left = (target - new_w) // 2
    pad_right = target - new_w - pad_left

    img = np.pad(
        img,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0
    )
    return img, (pad_top, pad_bottom, pad_left, pad_right), (w, h)

# MAIN
manifest_rows = []
for split in SPLITS:
    print(f"\n[INFO] Processing split: {split}")

    X_list, y_list = [], []

    for label_name, y_val in LABELS.items():
        src_dir = os.path.join(SPLIT_ROOT, split, label_name)
        png_dir = os.path.join(PNG_ROOT, split, label_name)
        os.makedirs(png_dir, exist_ok=True)

        files = [f for f in os.listdir(src_dir) if f.lower().endswith(".dcm")]

        for fname in tqdm(files, desc=f"{split}-{label_name}"):
            dcm_path = os.path.join(src_dir, fname)

            try:
                # --- LOAD DICOM ---
                ds = pydicom.dcmread(dcm_path)
                
                if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
                     raw = apply_voi_lut(ds.pixel_array, ds)
                else:
                     raw = ds.pixel_array
                
                img = raw.astype(np.float32)
                img = extract_2d(img) 

                is_inverted = needs_inversion(ds, img)
                if is_inverted:
                    img = np.max(img) - img
                    with open(INVERSION_LOG_PATH, "a") as f_inv:
                        f_inv.write(f"{dcm_path}\n")

                # --- NORMALIZE ---
                img = normalize(img)

                # --- PAD TO 1024 ---
                img_pad, pads, orig_size = resize_with_padding(img, TARGET_SIZE)

                # --- SAVE PNG ---
                png_name = fname.replace(".dcm", ".png")
                png_path = os.path.join(png_dir, png_name)
                cv2.imwrite(png_path, (img_pad * 255).astype(np.uint8))

                # --- SAVE FOR NPY ---
                X_list.append(img_pad[..., None])  
                y_list.append(y_val)

                # --- MANIFEST ---
                manifest_rows.append({
                    "split": split,
                    "label": label_name,
                    "y": y_val,
                    "src_dicom": dcm_path,
                    "png_path": png_path,
                    "orig_width": orig_size[0],
                    "orig_height": orig_size[1],
                    "pad_top": pads[0],
                    "pad_bottom": pads[1],
                    "pad_left": pads[2],
                    "pad_right": pads[3],
                    "final_size": TARGET_SIZE,
                    "inverted": is_inverted 
                })
            
            except Exception as e:
                print(f"Error processing {dcm_path}: {e}")
                continue

    # SAVE NPY 
    if len(X_list) > 0:
        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list).astype(np.int64)

        os.makedirs(NPY_ROOT, exist_ok=True)

        np.save(os.path.join(NPY_ROOT, f"X_{split}.npy"), X)
        np.save(os.path.join(NPY_ROOT, f"y_{split}.npy"), y)

        print(f"[OK] Saved X_{split}.npy & y_{split}.npy | shape={X.shape}")
    else:
        print(f"[WARN] No data for split {split}")

# SAVE MANIFEST
if manifest_rows:
    manifest_path = os.path.join(MANIFEST_ROOT, "baseline_manifest.csv")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n[OK] Baseline manifest saved: {manifest_path}")
    
    # Summary
    n_inverted = sum(1 for row in manifest_rows if row.get("inverted", False))
    print(f"[INFO] Total Inverted Files: {n_inverted}")

print("\n[DONE] Baseline PNG + NPY conversion completed.")