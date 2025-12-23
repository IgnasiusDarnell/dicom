#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
from datetime import datetime, date
import pydicom
from typing import Tuple, Optional
import shutil

# CONFIG
CURATED_ROOT = "../dataset/curated/"
REQUIRE_CXR = True
REQUIRE_CHEST = True
MIN_AGE_YEARS = 18.0

DATA_DIRS = {
    "TB": "../dataset/raw/rspaw/TB",
    "NonTB": "../dataset/raw/rspaw/NonTB"
}

OUTPUT_DIR = "../dataset/curated/manifests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.csv")
EXCLUDE_LOG_PATH = os.path.join(OUTPUT_DIR, "excluded.log")

for lbl in DATA_DIRS.keys():
    os.makedirs(os.path.join(CURATED_ROOT, lbl), exist_ok=True)

# LOGGING
def log_exclude(msg: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(EXCLUDE_LOG_PATH, 'a', encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

# UTIL
def _norm_text(s: str) -> str:
    return (s or "").upper().replace("_", " ").replace("-", " ").strip()

# AGE PARSING
def parse_patient_age_years(ds) -> Optional[float]:
    age = getattr(ds, "PatientAge", None)

    if isinstance(age, str) and len(age) >= 2:
        try:
            val = float(re.sub(r"[^\d.]", "", age))
            if age.endswith(("Y","y")): return val
            if age.endswith(("M","m")): return val / 12.0
            if age.endswith(("W","w")): return val / 52.0
            if age.endswith(("D","d")): return val / 365.0
        except Exception:
            pass

    bdate = getattr(ds, "PatientBirthDate", None)
    ref   = getattr(ds, "StudyDate", None) or getattr(ds, "ContentDate", None)

    def ymd_to_date(s):
        try:
            return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
        except Exception:
            return None

    if bdate and ref and len(bdate) >= 8 and len(ref) >= 8:
        db = ymd_to_date(bdate)
        dr = ymd_to_date(ref)
        if db and dr:
            return max(0.0, (dr - db).days / 365.25)

    return None

# FILTER RULES
EXCLUDE_KEYS = ("THORAKAL", "SHOULDER")

def is_filtered_study(ds, filename) -> Tuple[bool, str]:
    # --- Modality ---
    modality = _norm_text(getattr(ds, "Modality", ""))
    if REQUIRE_CXR and modality not in ("CR", "DX", "DR"):
        return True, f"Non-Xray modality: {modality}"

    # --- Body Part ---
    body_part = _norm_text(getattr(ds, "BodyPartExamined", ""))
    if REQUIRE_CHEST and body_part not in ("CHEST", "THORAX", "THORACIC"):
        return True, f"Non-chest body part: {body_part}"

    # --- Filename keywords ---
    fname = _norm_text(os.path.basename(filename))
    for k in EXCLUDE_KEYS:
        if k in fname:
            return True, f"Filename contains excluded keyword: {k}"

    return False, "OK"

# MAIN PROCESS
manifest_rows = []

for label, folder in DATA_DIRS.items():
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".dcm"):
            continue

        fpath = os.path.join(folder, fname)

        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)

            # --- AGE FILTER ---
            age_years = parse_patient_age_years(ds)
            if age_years is None:
                excluded = True
                reason = "Missing age"
            elif age_years < MIN_AGE_YEARS:
                excluded = True
                reason = f"Age < {MIN_AGE_YEARS}"
            else:
                excluded, reason = is_filtered_study(ds, fname)

            # --- COPY INCLUDED ---
            curated_path = ""
            if not excluded:
                dst_dir = os.path.join(CURATED_ROOT, label)
                curated_path = os.path.join(dst_dir, fname)

                # hindari overwrite diam-diam
                if not os.path.exists(curated_path):
                    shutil.copy2(fpath, curated_path)

            # --- LOG EXCLUDED ONLY ---
            if excluded:
                log_exclude(f"{fname} | {label} | {reason}")

            # --- MANIFEST (ALL FILES) ---
            manifest_rows.append({
                "filename": fname,
                "raw_path": fpath,
                "curated_path": curated_path,
                "label": label,
                "included": not excluded,
                "exclude_reason": "" if not excluded else reason,
                "age_years": age_years,
                "modality": getattr(ds, "Modality", ""),
                "body_part": getattr(ds, "BodyPartExamined", ""),
                "manufacturer": getattr(ds, "Manufacturer", "")
            })

        except Exception as e:
            log_exclude(f"{fname} | {label} | Read error: {e}")

# WRITE MANIFEST
with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"[OK] Manifest written to: {MANIFEST_PATH}")
print(f"[OK] Exclusion log written to: {EXCLUDE_LOG_PATH}")
