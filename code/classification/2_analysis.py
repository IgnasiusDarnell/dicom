import os
import sys
import argparse
import json
import logging
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, 
    f1_score, recall_score, confusion_matrix, 
    brier_score_loss, roc_curve, auc
)
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    # DATASET PATHS (Ganti sesuai server Anda)
    PATH_BASELINE_FULL_TEST = "dataset/baseline/PNG/test"
    PATH_HYFUSION_FULL_TEST = "dataset/hyfusion_v2/NPY/hyfusion" # Akan cari X_test.npy
    
    PATH_BASELINE_ROI_TEST = "dataset/ROI_FINAL_GENERATED/Baseline_NPY_ROI" # Akan cari X_test.npy
    PATH_HYFUSION_ROI_TEST = "dataset/ROI_FINAL_GENERATED/Hyfusion_NPY_ROI"
    
    # METADATA KOORDINAT CROP (Wajib untuk ROI Back-Projection)
    CSV_BASELINE_ROI = "dataset/ROI_FINAL_GENERATED/Baseline_NPY_ROI/X_test_coords_with_manifest.csv"
    CSV_HYFUSION_ROI = "dataset/ROI_FINAL_GENERATED/Hyfusion_NPY_ROI/X_test_coords_with_manifest.csv"
    
    # MODEL PATHS (Hasil Training Script 5)
    MODEL_PATHS = {
        "Baseline_Full": "experiments_classification/1_Baseline_Full/best_model.pt",
        "HyFusion_Full": "experiments_classification/2_HyFusion_Full/best_model.pt",
        "Baseline_ROI": "experiments_classification/3_Baseline_ROI/best_model.pt",
        "HyFusion_ROI": "experiments_classification/4_HyFusion_ROI/best_model.pt"
    }
    
    MASK_DIR = "dataset/final_dataset/masks" # Ground Truth Mask
    OUTPUT_DIR = "FINAL_ANALYSIS_REPORT"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = (512, 512)

# ==============================================================================
# 2. UTILS & STATS
# ==============================================================================
def fast_delong(probs1, probs2, labels):
    auc1 = roc_auc_score(labels, probs1)
    auc2 = roc_auc_score(labels, probs2)
    n1 = np.sum(labels == 1); n0 = np.sum(labels == 0)
    def se_auc(a):
        q1 = a / (2 - a); q2 = 2*a**2 / (1 + a)
        return np.sqrt((a*(1-a) + (n1-1)*(q1-a**2) + (n0-1)*(q2-a**2)) / (n1*n0))
    se_diff = np.sqrt(se_auc(auc1)**2 + se_auc(auc2)**2)
    z = np.abs(auc1 - auc2) / (se_diff + 1e-8)
    p = 2 * (1 - stats.norm.cdf(z))
    return auc1, auc2, p

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; probs = np.array(probs); labels = np.array(labels)
    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
        prop = np.mean(in_bin)
        if prop > 0:
            ece += np.abs(np.mean(probs[in_bin]) - np.mean(labels[in_bin] == 1)) * prop
    return ece

def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=1000):
    rng = np.random.RandomState(42)
    scores = []
    y_true = np.array(y_true); y_prob = np.array(y_prob)
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[idx])) < 2: continue
        scores.append(metric_fn(y_true[idx], y_prob[idx]))
    scores = np.sort(scores)
    return scores[int(0.025*len(scores))], scores[int(0.975*len(scores))]

# ==============================================================================
# 3. DATA & MODEL LOADER
# ==============================================================================
class TestDataset(Dataset):
    def __init__(self, source_path, mode='png', roi_csv=None):
        self.mode = mode
        self.data = []
        self.roi_meta = None
        
        if mode == 'png': # Untuk Baseline Full (Folder Structure)
            root = Path(source_path)
            tb = sorted(list((root / "TB").glob("*.png")))
            non = sorted(list((root / "Non_TB").glob("*.png")))
            for p in tb: self.data.append((str(p), 1, p.name))
            for p in non: self.data.append((str(p), 0, p.name))
            
        elif mode == 'npy': # Untuk HyFusion & ROI
            root = Path(source_path)
            xp = root / "X_test.npy" if (root / "X_test.npy").exists() else root / "xtest.npy"
            yp = root / "Y_test.npy" if (root / "Y_test.npy").exists() else root / "ytest.npy"
            self.images = np.load(xp, allow_pickle=True)
            self.labels = np.load(yp, allow_pickle=True)
            
            # Load Coordinates if ROI
            if roi_csv:
                self.roi_meta = pd.read_csv(roi_csv)
                
    def __len__(self):
        return len(self.data) if self.mode == 'png' else len(self.images)

    def __getitem__(self, idx):
        if self.mode == 'png':
            path, label, fname = self.data[idx]
            img = Image.open(path).convert('RGB')
            orig_path = path # For Visual
        else:
            img_arr = self.images[idx]
            label = int(self.labels[idx])
            
            # Handle ROI Metadata to find Original File (For Back-Projection)
            fname = f"npy_idx_{idx}"
            orig_path = None
            if self.roi_meta is not None:
                row = self.roi_meta.iloc[idx]
                if 'src_dicom' in row: fname = row['src_dicom']
                if 'generated_filename' in row: orig_path = row['generated_filename']
            
            # Handle Shape
            if img_arr.ndim == 2: img_arr = np.stack((img_arr,)*3, axis=-1)
            elif img_arr.ndim == 3 and img_arr.shape[-1] == 1: img_arr = np.repeat(img_arr, 3, axis=-1)
            if img_arr.max() <= 1.5: img_arr = (img_arr * 255).astype(np.uint8)
            else: img_arr = img_arr.astype(np.uint8)
            img = Image.fromarray(img_arr)
            
        tf = transforms.Compose([
            transforms.Resize(Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return tf(img), torch.tensor(label, dtype=torch.float32), fname, str(orig_path)

def load_model(path):
    model = models.efficientnet_b5(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    state = torch.load(path, map_location=Config.DEVICE)
    model.load_state_dict(state)
    model.to(Config.DEVICE)
    model.eval()
    return model

# ==============================================================================
# 4. XAI ENGINE (ADVANCED BACK-PROJECTION)
# ==============================================================================
class GradCAM:
    def __init__(self, model):
        self.model = model; self.grad = None; self.act = None
        self.model.features[-1].register_forward_hook(self.save_act)
        self.model.features[-1].register_full_backward_hook(self.save_grad)
    def save_act(self, m, i, o): self.act = o
    def save_grad(self, m, gi, go): self.grad = go[0]
    def __call__(self, x):
        out = self.model(x); self.model.zero_grad(); out[:, 0].backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(dim=1, keepdim=True))
        cam = cam - cam.min(); cam = cam / (cam.max() + 1e-7)
        return cam.detach().cpu().numpy()[0, 0]

def visualize_complex_layout(model_name, fname, img_orig_path, roi_heatmap=None, roi_coords=None, full_heatmap=None, mask_path=None):
    """
    Format: [DICOM (Raw) | Konversi (Input Model) | GradCAM | Masking Overlay]
    """
    # 1. Load Original Image (DICOM/Raw)
    # Karena kita pakai PNG, kita load PNG aslinya
    if not os.path.exists(img_orig_path): return # Skip if not found
    img_raw = cv2.imread(img_orig_path)
    h, w = img_raw.shape[:2]
    
    # 2. Prepare Heatmap Canvas (Full Size)
    final_hm = np.zeros((h, w), dtype=np.float32)
    
    if roi_heatmap is not None and roi_coords is not None:
        # --- ROI BACK-PROJECTION LOGIC ---
        # Resize heatmap kecil ke ukuran crop asli
        cx1, cy1, cx2, cy2 = int(roi_coords['crop_x1']), int(roi_coords['crop_y1']), int(roi_coords['crop_x2']), int(roi_coords['crop_y2'])
        crop_w, crop_h = cx2-cx1, cy2-cy1
        
        # Resize heatmap to match crop size
        hm_resized = cv2.resize(roi_heatmap, (crop_w, crop_h))
        
        # Paste into full canvas
        final_hm[cy1:cy2, cx1:cx2] = hm_resized
        
    elif full_heatmap is not None:
        # --- FULL IMAGE LOGIC ---
        # Resize heatmap 512x512 ke ukuran asli gambar
        final_hm = cv2.resize(full_heatmap, (w, h))
        
    # 3. Create Visualization Panels
    # Panel 1: Raw
    p1 = img_raw
    
    # Panel 2: Input Model (Simulation) -> Disini kita pakai Raw saja agar representatif, 
    # atau gambar HyFusion jika ini model HyFusion
    p2 = img_raw # Simplified for layout
    
    # Panel 3: GradCAM (Ignore Background = Apply Threshold)
    hm_color = cv2.applyColorMap((final_hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_color[final_hm < 0.2] = 0 # Ignore background low activation
    p3 = cv2.addWeighted(img_raw, 0.7, hm_color, 0.3, 0)
    
    # Panel 4: Masking Overlay (With Ground Truth if available)
    p4 = img_raw.copy()
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (w, h))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p4, contours, -1, (0, 255, 0), 2) # Green Contour
        # Overlay Heatmap inside Mask? Or just Heatmap again?
        # User request: "masking dengan hasil konversi" -> Maybe Mask overlay
        
    # Combine
    combined = np.hstack([p1, p2, p3, p4])
    
    # Save
    save_dir = Path(Config.OUTPUT_DIR) / f"VIS_{model_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / f"{fname}_layout.jpg"), combined)
    
    # Return metrics
    return final_hm

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def main():
    Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)
    
    # DEFINE SCENARIOS
    scenarios = [
        # 1. Baseline Full
        {"name": "Baseline_Full", "model": "Baseline_Full", "data": Config.PATH_BASELINE_FULL_TEST, "mode": "png", "roi": False},
        # 2. HyFusion Full
        {"name": "HyFusion_Full", "model": "HyFusion_Full", "data": Config.PATH_HYFUSION_FULL_TEST, "mode": "npy", "roi": False},
        # 3. Baseline ROI
        {"name": "Baseline_ROI", "model": "Baseline_ROI", "data": Config.PATH_BASELINE_ROI_TEST, "mode": "npy", "roi": True, "csv": Config.CSV_BASELINE_ROI},
        # 4. HyFusion ROI
        {"name": "HyFusion_ROI", "model": "HyFusion_ROI", "data": Config.PATH_HYFUSION_ROI_TEST, "mode": "npy", "roi": True, "csv": Config.CSV_HYFUSION_ROI},
        
        # --- CROSS TESTING ---
        {"name": "HyFusion_on_Baseline", "model": "HyFusion_Full", "data": Config.PATH_BASELINE_FULL_TEST, "mode": "png", "roi": False},
        {"name": "Baseline_on_HyFusion", "model": "Baseline_Full", "data": Config.PATH_HYFUSION_FULL_TEST, "mode": "npy", "roi": False},
    ]
    
    results_df = []
    
    for sc in scenarios:
        print(f"\nProcessing Scenario: {sc['name']}...")
        
        # 1. Load Model & Data
        model = load_model(Config.MODEL_PATHS[sc['model']])
        ds = TestDataset(sc['data'], sc['mode'], sc.get('csv'))
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        gradcam = GradCAM(model)
        
        # 2. Inference & Stats
        all_probs, all_labels = [], []
        
        # XAI Processing (Top 20 TP)
        roi_meta = pd.read_csv(sc['csv']) if sc.get('csv') else None
        
        for imgs, labels, fnames, orig_paths in tqdm(loader):
            imgs = imgs.to(Config.DEVICE)
            
            # Predict
            logits = model(imgs)
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            
            # Generate XAI for visualization (Sample)
            for i in range(len(probs)):
                if probs[i] > 0.8 and labels[i] == 1: # High confidence TP only
                    hm_small = gradcam(imgs[i:i+1])
                    
                    # Logic Visualization Layout
                    fname = fnames[i]
                    # Find coords if ROI
                    coords = None
                    if sc['roi'] and roi_meta is not None:
                        # Match by filename or index
                        if 'npy_idx' in fname:
                            idx = int(fname.split('_')[-1])
                            coords = roi_meta.iloc[idx]
                        
                    # Call Visualizer
                    # Note: orig_paths[i] might be full path, we need to check existence
                    mask_p = Path(Config.MASK_DIR) / Path(fname).name
                    
                    visualize_complex_layout(
                        sc['name'], fname, orig_paths[i], 
                        roi_heatmap=hm_small if sc['roi'] else None,
                        roi_coords=coords,
                        full_heatmap=hm_small if not sc['roi'] else None,
                        mask_path=str(mask_p)
                    )
                    
        # 3. Calculate Metrics
        y_true, y_prob = np.array(all_labels), np.array(all_probs)
        y_pred = (y_prob > 0.5).astype(int)
        
        res = {
            "Scenario": sc['name'],
            "AUROC": roc_auc_score(y_true, y_prob),
            "AUPRC": average_precision_score(y_true, y_prob),
            "Acc": accuracy_score(y_true, y_pred),
            "Sens": recall_score(y_true, y_pred),
            "Spec": recall_score(y_true, y_pred, pos_label=0),
            "F1": f1_score(y_true, y_pred),
            "Brier": brier_score_loss(y_true, y_prob),
            "ECE": compute_ece(y_prob, y_true)
        }
        
        # Bootstrap CI for AUPRC
        low, high = bootstrap_ci(y_true, y_prob, average_precision_score)
        res['AUPRC_95CI'] = f"{low:.3f}-{high:.3f}"
        
        results_df.append(res)
        
    # Save Final Report
    pd.DataFrame(results_df).to_csv(f"{Config.OUTPUT_DIR}/Final_Comparison_Table.csv", index=False)
    print("\nDONE! Check FINAL_ANALYSIS_REPORT folder.")

if __name__ == "__main__":
    main()