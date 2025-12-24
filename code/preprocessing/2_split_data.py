#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import random
import shutil
from datetime import datetime
from collections import defaultdict

import pydicom
import pandas as pd
from tqdm import tqdm

# CONFIGURATION
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

CURATED_ROOT = "../../dataset/curated/"
SPLIT_ROOT   = "../../dataset/split_dataset/"

OUTPUT_DIR = "../../dataset/split_dataset/manifests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLIT_MANIFEST_PATH = os.path.join(OUTPUT_DIR, "split_manifest.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "split_summary_by_manufacturer.csv")

LABELS = ["TB", "NonTB"]
SPLITS = ["train", "val", "test"]

# Target fixed sizes
TARGET = {
    "test": {"TB": 2, "NonTB": 2},
    "val":  {"TB": 2, "NonTB": 2},
}

# CREATE SPLIT DIRECTORIES
print("[INFO] Creating split directory structure...")

for split in SPLITS:
    for label in LABELS:
        os.makedirs(
            os.path.join(SPLIT_ROOT, split, label),
            exist_ok=True
        )

# BUILD MASTER TABLE (ROW-LEVEL)
records = []

print("[INFO] Scanning curated dataset...")

for label in LABELS:
    folder = os.path.join(CURATED_ROOT, label)

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Missing folder: {folder}")

    files = [f for f in os.listdir(folder) if f.lower().endswith(".dcm")]

    for fname in tqdm(files, desc=f"Reading {label}", unit="file"):
        src_path = os.path.join(folder, fname)

        try:
            ds = pydicom.dcmread(src_path, stop_before_pixels=True)

            manufacturer = getattr(ds, "Manufacturer", "UNKNOWN")
            manufacturer = str(manufacturer).strip().upper()

            age_years = None
            try:
                age_years = float(getattr(ds, "PatientAge", None)[:-1]) \
                    if isinstance(getattr(ds, "PatientAge", None), str) else None
            except Exception:
                pass

            width  = getattr(ds, "Columns", None)
            height = getattr(ds, "Rows", None)

            records.append({
                "label": label,
                "manufacturer": manufacturer,
                "src_path": src_path,
                "width": width,
                "height": height,
                "age_years": age_years,
            })

        except Exception as e:
            print(f"[WARN] Failed to read {src_path}: {e}")

df = pd.DataFrame(records)
df["index"] = range(len(df))

print(f"[INFO] Total curated images: {len(df)}")

# STRATIFIED SAMPLING FUNCTION
def stratified_sample(df_subset, label, target_count):
    """
    Stratified sampling by manufacturer for a given class label.
    """
    subset = df_subset[df_subset["label"] == label]

    if len(subset) < target_count:
        raise ValueError(
            f"Not enough samples for label={label}: "
            f"required={target_count}, available={len(subset)}"
        )

    manuf_counts = subset["manufacturer"].value_counts()
    manuf_props = manuf_counts / manuf_counts.sum()

    selected_idx = []

    for manuf, prop in manuf_props.items():
        k = int(round(prop * target_count))
        candidates = subset[subset["manufacturer"] == manuf]
        candidates = candidates[~candidates.index.isin(selected_idx)]

        if k > 0:
            k = min(k, len(candidates))
            selected = random.sample(list(candidates.index), k)
            selected_idx.extend(selected)

    # Adjust for rounding errors
    if len(selected_idx) < target_count:
        remaining = subset[~subset.index.isin(selected_idx)]
        need = target_count - len(selected_idx)
        selected_idx.extend(
            random.sample(list(remaining.index), need)
        )

    return selected_idx[:target_count]

# ASSIGN SPLITS
df["split"] = "train"  # default

print("[INFO] Assigning TEST split...")
for label in LABELS:
    idx = stratified_sample(df, label, TARGET["test"][label])
    df.loc[idx, "split"] = "test"

print("[INFO] Assigning VALIDATION split...")
remaining = df[df["split"] == "train"]

for label in LABELS:
    idx = stratified_sample(remaining, label, TARGET["val"][label])
    df.loc[idx, "split"] = "val"

# COPY FILES INTO SPLIT FOLDERS
print("[INFO] Copying DICOM files into split folders...")

split_paths = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying files"):
    src = row["src_path"]
    split = row["split"]
    label = row["label"]

    dst = os.path.join(
        SPLIT_ROOT,
        split,
        label,
        os.path.basename(src)
    )

    if not os.path.exists(dst):
        shutil.copy2(src, dst)

    split_paths.append(dst)

df["split_path"] = split_paths

df_manifest = df[[
    "index",
    "split",
    "label",
    "manufacturer",
    "src_path",
    "split_path",
    "width",
    "height",
    "age_years",
]]

df_manifest.to_csv(SPLIT_MANIFEST_PATH, index=False)

df_manifest.to_csv(SPLIT_MANIFEST_PATH, index=False)

print(f"[OK] Split manifest saved to: {SPLIT_MANIFEST_PATH}")

summary = (
    df.groupby(["manufacturer", "label", "split"])
      .size()
      .unstack(fill_value=0)
      .reset_index()
)

summary.to_csv(SUMMARY_PATH, index=False)

print(f"[OK] Split summary saved to: {SUMMARY_PATH}")

print("\n=== SPLIT COUNTS (LABEL × SPLIT) ===")
print(df.groupby(["split", "label"]).size())

print("\n=== TEST SET DISTRIBUTION (MANUFACTURER × LABEL) ===")
print(df[df["split"] == "test"].groupby(["manufacturer", "label"]).size())

print("\n[DONE] Dataset splitting completed successfully.")
