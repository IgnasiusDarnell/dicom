#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, csv
import numpy as np
import pandas as pd
import cv2
import pydicom
from typing import List, Optional
from tqdm import tqdm

from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# CONFIGURATION
TARGET_SIZE = 1024

BG_THRESHOLD = 0.05
MIN_CONTENT_FRAC = 0.15

SIGMA_HOMO = 30.0
HIGH_GAIN = 1.5
LOW_GAIN = 0.5

PSNR_MIN = 25.0
SSIM_MIN = 0.80
CNR_MIN  = 0.8

ALPHA_MIN_SAFETY = 0.00
DELTA_ALPHA = 0.05

ALPHA_MIN = 0.7
ALPHA_MAX = 0.9

ROOT_SPLIT = "../../dataset/split_dataset"
OUT_ROOT   = "../../dataset/hyfusion_v2" 

NPY_ROOT   = os.path.join(OUT_ROOT, "NPY")
PNG_ROOT   = os.path.join(OUT_ROOT, "PNG")
MAN_ROOT   = os.path.join(OUT_ROOT, "manifest")
LOG_PATH = os.path.join(MAN_ROOT, "log_rollback.txt")
INVERSION_LOG_PATH = os.path.join(MAN_ROOT, "log_inversion.txt")

# Setup Directories
COMPONENTS = ["hyfusion", "freq", "spatial"]
for comp in COMPONENTS:
    os.makedirs(os.path.join(NPY_ROOT, comp), exist_ok=True)
    for split in ["train", "val", "test"]:
        for cls in ["TB", "NonTB"]:
            os.makedirs(os.path.join(PNG_ROOT, comp, split, cls), exist_ok=True)

os.makedirs(MAN_ROOT, exist_ok=True)

# Initialize Logs
with open(LOG_PATH, "w") as f:
    f.write("# HyFusion-v2 Rollback Log\n")

with open(INVERSION_LOG_PATH, "w") as f:
    f.write("# HyFusion-v2 Inversion Log (List of Inverted Files)\n")

# UTILITIES
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

def pad_to_1024(img: np.ndarray):
    h, w = img.shape
    scale = min(TARGET_SIZE / h, TARGET_SIZE / w)

    if scale < 1.0:
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape
    else:
        new_h, new_w = h, w

    pt = (TARGET_SIZE - new_h) // 2
    pb = TARGET_SIZE - new_h - pt
    pl = (TARGET_SIZE - new_w) // 2
    pr = TARGET_SIZE - new_w - pl

    assert pt >= 0 and pb >= 0 and pl >= 0 and pr >= 0, \
        f"Invalid padding: {(pt, pb, pl, pr)}"

    canvas = np.pad(
        img,
        ((pt, pb), (pl, pr)),
        mode="constant",
        constant_values=0
    )

    valid_mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=bool)
    valid_mask[pt:pt+new_h, pl:pl+new_w] = True

    pad_info = {
        "orig_h": int(h), 
        "orig_w": int(w),
        "scaled_h": new_h,
        "scaled_w": new_w,
        "scale": scale,
        "pad_top": pt,
        "pad_bottom": pb,
        "pad_left": pl,
        "pad_right": pr,
    }

    return canvas.astype(np.float32), valid_mask, pad_info

# FILTERS
def homomorphic_filter(img):
    mn, mx = img.min(), img.max()
    norm = (img - mn) / (mx - mn + 1e-12)
    L = np.log1p(norm)
    F = np.fft.fftshift(np.fft.fft2(L))
    r, c = img.shape
    cy, cx = r//2, c//2
    y, x = np.ogrid[:r, :c]
    D2 = (y-cy)**2 + (x-cx)**2
    H = (HIGH_GAIN-LOW_GAIN)*(1-np.exp(-D2/(2*SIGMA_HOMO**2))) + LOW_GAIN
    Lf = np.real(np.fft.ifft2(np.fft.ifftshift(F*H)))
    exp = np.expm1(Lf)
    exp = (exp-exp.min())/(exp.max()-exp.min()+1e-12)
    return exp*(mx-mn)+mn

def adaptive_gamma(img):
    mn, mx = img.min(), img.max()
    img_n = (img-mn)/(mx-mn+1e-12)
    lo, hi = np.percentile(img_n, [3, 97])
    img_c = np.clip((img_n-lo)/(hi-lo+1e-12),0,1)
    return img_c**(1/1.05)

def robust_spatial_norm(img, mask):
    vals = img[mask]
    med = np.median(vals)
    mad = np.median(np.abs(vals-med))+1e-12
    z = (img-med)/mad
    z = np.clip(z,-3,3)
    out = (z+3)/6
    out[~mask] = 0
    return out

# QUALITY GUARD (FIXED)
def compute_cnr(sig, ref, mask):
    s = sig[mask]; r = ref[mask]
    return (s.mean()-r.mean())/(r.std()+1e-9)

def quality_guard(ie, ifreq, ispat, mask, alpha_s):
    # 1. First Try
    blend_init = ispat.copy()
    blend_init[mask] = alpha_s * ifreq[mask] + (1 - alpha_s) * ispat[mask]
    
    psnr_i = peak_signal_noise_ratio(ie[mask], blend_init[mask], data_range=1)
    ssim_i = structural_similarity(ie, blend_init, data_range=1)
    cnr_i  = compute_cnr(blend_init, ie, mask)
    
    if psnr_i >= PSNR_MIN and ssim_i >= SSIM_MIN and cnr_i >= CNR_MIN:
        return blend_init, alpha_s, psnr_i, ssim_i, cnr_i, False

    # 2. Rollback Loop
    best = (psnr_i + ssim_i*10 + cnr_i, blend_init, alpha_s, psnr_i, ssim_i, cnr_i)
    
    a = alpha_s - DELTA_ALPHA
    while a >= ALPHA_MIN_SAFETY:
        blend = ispat.copy()
        blend[mask] = a * ifreq[mask] + (1 - a) * ispat[mask]
        
        psnr = peak_signal_noise_ratio(ie[mask], blend[mask], data_range=1)
        ssim = structural_similarity(ie, blend, data_range=1)
        cnr  = compute_cnr(blend, ie, mask)
        
        if psnr >= PSNR_MIN and ssim >= SSIM_MIN and cnr >= CNR_MIN:
            return blend, a, psnr, ssim, cnr, True 
        
        curr_score = psnr + ssim*10 + cnr
        if curr_score > best[0]:
            best = (curr_score, blend, a, psnr, ssim, cnr)
        
        a -= DELTA_ALPHA
        
    # 3. Fallback
    _, img, a, psnr, ssim, cnr = best
    
    # Check if value actually changed
    is_rollback = (abs(a - alpha_s) > 1e-6)
    
    return img, a, psnr, ssim, cnr, is_rollback

# DHI COMPUTATION 
def _norm_text(s: str) -> str:
    return (s or "").upper().replace("_", " ").replace("-", " ")

SITE_CANON_MAP = { 
    "RS PARU DR ARIO WIRAWAN SALATIGA": "RS Paru dr. Ario Wirawan",
    "RS PARU DR. ARIO WIRAWAN": "RS Paru dr. Ario Wirawan",
    "RS PARU DR ARIO WIRAWAN": "RS Paru dr. Ario Wirawan",
    "RS PARU DR ARIO WIRAWAN ": "RS Paru dr. Ario Wirawan",
    "RSP DR. ARIO WIRAWAN": "RS Paru dr. Ario Wirawan",
}

def canonicalize_site_id(raw: str) -> str:
    if not raw or not raw.strip():
        return "RS Paru dr. Ario Wirawan"
    s_norm = _norm_text(raw).strip()
    if s_norm in SITE_CANON_MAP:
        return SITE_CANON_MAP[s_norm]
    if "ARIO WIRAWAN" in s_norm and "PARU" in s_norm:
        return "RS Paru dr. Ario Wirawan"
    return raw.strip()

def get_site_id(ds) -> str:
    for tag in ["InstitutionName", "StationName", "ManufacturerModelName", "Manufacturer"]:
        val = getattr(ds, tag, None)
        if val not in (None, ""):
            return canonicalize_site_id(str(val))
    return "RS Paru dr. Ario Wirawan"

def compute_site_dhi(dicom_paths, alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX, eps=1e-6, save_dir=MAN_ROOT):
    records = []

    for p in tqdm(dicom_paths, desc="Collecting DHI metadata"):
        try:
            ds = pydicom.dcmread(p)
        except Exception:
            continue

        site = get_site_id(ds)
        bits = getattr(ds, "BitsStored", None)
        try: bits = int(bits) if bits else None
        except: bits = None

        d = None
        pxs = getattr(ds, "PixelSpacing", None)
        if pxs and len(pxs) >= 2:
            try: d = 0.5 * (float(pxs[0]) + float(pxs[1]))
            except: d = None

        rows = getattr(ds, "Rows", None)
        cols = getattr(ds, "Columns", None)
        try: A = int(rows) * int(cols)
        except: A = None

        m_i = None
        try:
            raw = apply_voi_lut(ds.pixel_array, ds)
            img = extract_2d(raw).astype(np.float32)
            if needs_inversion(ds, img): img = img.max() - img
            mn, mx = img.min(), img.max()
            if mx - mn > 1e-12:
                ie = (img - mn) / (mx - mn)
                mask = ie > BG_THRESHOLD
                if mask.sum() >= max(1, MIN_CONTENT_FRAC * ie.size):
                    m_i = float(np.mean(ie[mask]))
                else:
                    m_i = float(np.mean(ie))
        except:
            m_i = None

        records.append({
            "dicom_path": p,
            "site": site,
            "BitsStored": bits,
            "spacing": d,
            "area": A,
            "intensity_mean": m_i,
        })

    df = pd.DataFrame(records)
    
    # Save Raw Audit
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "dhi_audit_raw_metadata.csv"), index=False)
    print(f"Saved raw metadata audit to {os.path.join(save_dir, 'dhi_audit_raw_metadata.csv')}")

    site_rows = []
    for site, g in df.groupby("site"):
        N = len(g)
        bs = g["BitsStored"].dropna()
        if len(bs) > 0:
            dom = bs.value_counts().max()
            H_bit = 1.0 - (dom / len(bs))
        else: H_bit = 0.0

        dvals = g["spacing"].dropna()
        if len(dvals) > 1 and dvals.mean() > 0:
            H_spacing = min(dvals.std() / (dvals.mean() + eps), 1.0)
        else: H_spacing = 0.0

        Avals = g["area"].dropna()
        if len(Avals) > 1 and Avals.mean() > 0:
            H_size = min(Avals.std() / (Avals.mean() + eps), 1.0)
        else: H_size = 0.0

        ivals = g["intensity_mean"].dropna()
        if len(ivals) > 1 and ivals.mean() > 0:
            H_int = min(ivals.std() / (ivals.mean() + eps), 1.0)
        else: H_int = 0.0

        DHI = (H_bit + H_spacing + H_size + H_int) / 4.0
        alpha_s = alpha_min + (alpha_max - alpha_min) * DHI
        freq_pct = alpha_s * 100
        spat_pct = (1.0 - alpha_s) * 100

        site_rows.append({
            "site": site,
            "N_images": N,
            "H_bitdepth": H_bit,
            "H_spacing": H_spacing,
            "H_size": H_size,
            "H_intensity": H_int,
            "DHI": DHI,
            "alpha_s": alpha_s,
            "freq_pct": f"{freq_pct:.2f}%",
            "spat_pct": f"{spat_pct:.2f}%",
        })

    df_site = pd.DataFrame(site_rows)
    df_site.to_csv(os.path.join(save_dir, "site_dhi_components.csv"), index=False)
    SITE_ALPHA = dict(zip(df_site["site"], df_site["alpha_s"]))
    return SITE_ALPHA, df_site

# MAIN PIPELINE
manifest = []
all_paths = []

print("Computing DHI Site Variance...")
for split in ["train", "val", "test"]:
    for cls in ["TB", "NonTB"]:
        d = os.path.join(ROOT_SPLIT, split, cls)
        if os.path.exists(d):
            all_paths += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".dcm")]

SITE_ALPHA, df_site = compute_site_dhi(all_paths)

print("Starting processing...")

for split in ["train", "val", "test"]:
    X_hyf  = []
    X_freq = []
    X_spat = []
    y = []
    
    print(f"Processing split: {split}")

    for cls, yv in [("TB", 1), ("NonTB", 0)]:
        src = os.path.join(ROOT_SPLIT, split, cls)
        if not os.path.exists(src): continue

        for f in tqdm(os.listdir(src), desc=f"{split}-{cls}"):
            if not f.endswith(".dcm"): continue
            
            p = os.path.join(src, f)
            try:
                # 1. Read
                ds = pydicom.dcmread(p)
                raw = apply_voi_lut(ds.pixel_array, ds)
                img = extract_2d(raw)
                
                # 2. Inversion Check
                is_inverted = needs_inversion(ds, img)
                if is_inverted: 
                    img = img.max() - img
                    with open(INVERSION_LOG_PATH, "a") as f_inv:
                        f_inv.write(f"{p}\n")
                
                # 3. Preprocess
                ie = (img - img.min()) / (img.max() - img.min() + 1e-12)
                mask = ie > BG_THRESHOLD

                # 4. Filter Components
                ifreq = adaptive_gamma(homomorphic_filter(ie))
                ispat = robust_spatial_norm(ifreq, mask)
                
                # 5. HyFusion & Quality Guard
                alpha_site = SITE_ALPHA.get(get_site_id(ds), ALPHA_MIN)
                ihyf, a_f, psnr, ssim, cnr, is_rollback = quality_guard(ie, ifreq, ispat, mask, alpha_site)
                
                # Log Rollback
                if is_rollback:
                    with open(LOG_PATH, "a") as f_log:
                        f_log.write(f"{p} | {split} | {cls} | site={get_site_id(ds)} | ROLLBACK_OCCURRED | orig={alpha_site:.2f} | final={a_f:.2f}\n")

                # 6. Pad All Components to 1024
                ihyf_1024, _, pad_info = pad_to_1024(ihyf)
                ifreq_1024, _, _ = pad_to_1024(ifreq)
                ispat_1024, _, _ = pad_to_1024(ispat)

                # 7. Metadata
                freq_pct = a_f * 100
                spat_pct = (1.0 - a_f) * 100
                px_min = float(ihyf_1024.min())
                px_max = float(ihyf_1024.max())

                # 8. Save PNGs (Separated Folders)
                base_name = os.path.splitext(os.path.basename(p))[0] + ".png"
                
                path_hyf  = os.path.join(PNG_ROOT, "hyfusion", split, cls, base_name)
                path_freq = os.path.join(PNG_ROOT, "freq", split, cls, base_name)
                path_spat = os.path.join(PNG_ROOT, "spatial", split, cls, base_name)

                cv2.imwrite(path_hyf,  np.clip(ihyf_1024 * 255.0, 0, 255).astype(np.uint8))
                cv2.imwrite(path_freq, np.clip(ifreq_1024 * 255.0, 0, 255).astype(np.uint8))
                cv2.imwrite(path_spat, np.clip(ispat_1024 * 255.0, 0, 255).astype(np.uint8))

                # 9. Append to Arrays
                X_hyf.append(ihyf_1024[..., None])
                X_freq.append(ifreq_1024[..., None])
                X_spat.append(ispat_1024[..., None])
                y.append(yv)

                # 10. Manifest
                manifest.append({
                    "split": split,
                    "label": cls,
                    "src_dicom": p,
                    "path_hyfusion": path_hyf,
                    "path_freq": path_freq,
                    "path_spatial": path_spat,
                    "site": get_site_id(ds),
                    "rollback": is_rollback,
                    "alpha_s": alpha_site,
                    "alpha_final": a_f,
                    "freq_pct": f"{freq_pct:.2f}%",
                    "spat_pct": f"{spat_pct:.2f}%",
                    "orig_h": pad_info["orig_h"],
                    "orig_w": pad_info["orig_w"],
                    "pixel_min": px_min,
                    "pixel_max": px_max,
                    "inverted": is_inverted,
                    "psnr": psnr,
                    "ssim": ssim,
                    "cnr": cnr
                })

            except Exception as e:
                print(f"Error processing {p}: {e}")
                continue

    if len(y) > 0:
        print(f"Saving NPYs for {split}...")
        y_arr = np.array(y, np.int64)
        
        # Save HyFusion
        np.save(os.path.join(NPY_ROOT, "hyfusion", f"X_{split}.npy"), np.array(X_hyf, np.float32))
        np.save(os.path.join(NPY_ROOT, "hyfusion", f"y_{split}.npy"), y_arr) 
        
        # Save Frequency
        np.save(os.path.join(NPY_ROOT, "freq", f"X_{split}.npy"), np.array(X_freq, np.float32))
        np.save(os.path.join(NPY_ROOT, "freq", f"y_{split}.npy"), y_arr)     
        
        # Save Spatial
        np.save(os.path.join(NPY_ROOT, "spatial", f"X_{split}.npy"), np.array(X_spat, np.float32))
        np.save(os.path.join(NPY_ROOT, "spatial", f"y_{split}.npy"), y_arr)  
        
        print(f"Saved {len(y)} samples to component folders.")
    else:
        print(f"Warning: No data for split {split}")

# Save Final Manifest
pd.DataFrame(manifest).to_csv(os.path.join(MAN_ROOT, "hyfusion_manifest_separated.csv"), index=False)

# Summary to Log
df_m = pd.DataFrame(manifest)
if not df_m.empty:
    n_total = len(df_m)
    n_ok = (df_m["rollback"] == False).sum()
    n_rb = (df_m["rollback"] == True).sum()
    n_rb0 = (df_m["alpha_final"] <= 0.0).sum()
    n_inverted = df_m["inverted"].sum()

    with open(LOG_PATH, "a") as f:
        f.write("\n# SUMMARY\n")
        f.write(f"Total files       : {n_total}\n")
        f.write(f"OK (No Rollback)  : {n_ok}\n")
        f.write(f"Rollback Occurred : {n_rb}\n")
        f.write(f"Rollback to zero  : {n_rb0}\n")
        f.write(f"Inverted Files    : {n_inverted}\n")

print("[DONE] HyFusion-v2 Separated Pipeline completed.")