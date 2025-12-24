#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import hashlib
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
def extract_2d(img):
    img = np.asarray(img)

    if img.ndim == 2:
        return img.astype(np.float32)

    if img.ndim == 3 and img.shape[-1] > 1:
        return img[..., 0].astype(np.float32)

    if img.ndim == 3 and img.shape[0] > 1:
        return img[0].astype(np.float32)

    if img.ndim == 3:
        vars_ = np.var(img.reshape(img.shape[0], -1), axis=1)
        return img[int(np.argmax(vars_))].astype(np.float32)

    if img.ndim == 4:
        return img[0, ..., 0].astype(np.float32)

    return np.squeeze(img).astype(np.float32)


def resize_with_padding_uint8(img, target=1024):
    h, w = img.shape
    scale = min(target / h, target / w)

    if scale < 1.0:
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        nh, nw = h, w

    pt = (target - nh) // 2
    pb = target - nh - pt
    pl = (target - nw) // 2
    pr = target - nw - pl

    img = np.pad(
        img,
        ((pt, pb), (pl, pr)),
        mode="constant",
        constant_values=0
    )

    return img, (pt, pb, pl, pr), (w, h)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# DICOM → PNG (BASELINE 0–255)
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

            # ---- LOAD DICOM ----
            ds = pydicom.dcmread(dcm_path)
            img = ds.pixel_array.astype(np.float32)

            try:
                img = apply_voi_lut(img, ds)
            except Exception:
                pass

            img = extract_2d(img)

            # ---- NORMALIZE TO [0,255] ----
            mn, mx = img.min(), img.max()
            img = (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img)
            img_u8 = (img * 255).clip(0, 255).astype(np.uint8)

            # ---- PAD TO 1024 ----
            img_u8, pads, orig = resize_with_padding_uint8(img_u8, TARGET_SIZE)

            # ---- SAVE PNG ----
            png_name = fname.replace(".dcm", ".png")
            png_path = os.path.join(png_dir, png_name)
            cv2.imwrite(png_path, img_u8)

            # PNG → NPY 
            png_loaded = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            img_npy = (png_loaded.astype(np.float32) / 255.0)[..., None]

            X_list.append(img_npy)
            y_list.append(y_val)

            # ---- MANIFEST ----
            manifest_rows.append({
                "split": split,
                "label": label_name,
                "y": y_val,
                "src_dicom": dcm_path,
                "png_path": png_path,
                "png_sha256": sha256(png_path),
                "orig_width": orig[0],
                "orig_height": orig[1],
                "pad_top": pads[0],
                "pad_bottom": pads[1],
                "pad_left": pads[2],
                "pad_right": pads[3],
                "final_size": TARGET_SIZE,
                "baseline_range": "0-255"
            })

    # ---- SAVE NPY PER SPLIT ----
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
print("\n[DONE] DICOM → PNG → NPY pipeline completed successfully.")
