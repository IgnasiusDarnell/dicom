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

for p in [PNG_ROOT, NPY_ROOT, MANIFEST_ROOT]:
    os.makedirs(p, exist_ok=True)

SPLITS = ["train", "val", "test"]
LABELS = {"TB": 1, "NonTB": 0}

# UTIL
def load_dicom_image(path):
    ds = pydicom.dcmread(path)
    img = apply_voi_lut(ds.pixel_array, ds) \
        if hasattr(ds, "WindowCenter") else ds.pixel_array
    img = img.astype(np.float32)
    return img

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

def normalize(img):
    mn, mx = img.min(), img.max()
    if mx > mn:
        return (img - mn) / (mx - mn)
    return np.zeros_like(img)

def resize_with_padding(img, target=1024):
    img = extract_2d(img)
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

            # --- LOAD ---
            img = load_dicom_image(dcm_path)
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
                "final_size": TARGET_SIZE
            })

    # --- SAVE NPY ---
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list).astype(np.int64)

    npy_dir = os.path.join(NPY_ROOT, split)
    os.makedirs(npy_dir, exist_ok=True)

    np.save(os.path.join(npy_dir, "X.npy"), X)
    np.save(os.path.join(npy_dir, "y.npy"), y)

# SAVE MANIFEST
manifest_path = os.path.join(MANIFEST_ROOT, "baseline_manifest.csv")

with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"\n[OK] Baseline manifest saved: {manifest_path}")

print("\n[DONE] Baseline PNG + NPY conversion completed.")
