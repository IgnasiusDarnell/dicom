import os
import sys
import argparse
import json
import logging
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import warnings

# Plotting
import matplotlib
matplotlib.use('Agg') # Headless mode
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, brier_score_loss, 
    confusion_matrix, roc_curve, auc, precision_score, recall_score
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import models, transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Filter warning
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIG ---
class Config:
    IMG_SIZE = (512, 512)   
    BATCH_SIZE = 16          
    EPOCHS = 50
    LR = 1e-4
    PATIENCE = 10
    NUM_WORKERS = 4
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Toggle Augmentasi (True/False)
    USE_AUGMENTATION = False 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(save_dir, name):
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    # Overwrite mode 'w'
    fh = logging.FileHandler(save_dir / "training.log", mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

# --- PLOTTING FUNCTIONS (Modified to use specific dir) ---
def plot_history(history, save_dir):
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # AUC Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_auc'], label='Train AUC')
    plt.plot(history['val_auc'], label='Val AUC')
    plt.title('AUC Curve')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / "history.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-TB', 'TB'], yticklabels=['Non-TB', 'TB'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_true, y_prob, save_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_dir / "roc_curve.png")
    plt.close()

# --- DATASET ---
class UniversalDataset(Dataset):
    def __init__(self, source_path, split='train', mode='png', transform=None):
        self.mode = mode
        self.transform = transform
        self.data = [] 
        
        root = Path(source_path)
        
        if mode == 'png':
            split_root = root / split
            if not split_root.exists(): split_root = root 
            
            tb = sorted(list((split_root / "TB").glob("*.png")))
            non = sorted(list((split_root / "Non_TB").glob("*.png")))
            
            if not tb and not non:
                tb = sorted(list(split_root.rglob("TB/*.png")))
                non = sorted(list(split_root.rglob("Non_TB/*.png")))

            for p in tb: self.data.append((str(p), 1, p.name))
            for p in non: self.data.append((str(p), 0, p.name))
            
            if len(self.data) == 0:
                print(f"WARNING: Tidak ada gambar PNG ditemukan di {split_root}")
            
        elif mode == 'npy':
            candidates_x = [f"X_{split}.npy", f"x_{split}.npy", f"X{split}.npy", f"x{split}.npy"]
            candidates_y = [f"Y_{split}.npy", f"y_{split}.npy", f"Y{split}.npy", f"y{split}.npy"]
            
            xp, yp = None, None
            for name in candidates_x:
                if (root / name).exists(): xp = root / name; break
            for name in candidates_y:
                if (root / name).exists(): yp = root / name; break
            
            if xp is None or yp is None:
                raise FileNotFoundError(f"File NPY split '{split}' tidak ditemukan di {root}")

            print(f"[{split.upper()}] Loading NPY: {xp.name} & {yp.name}")
            self.images = np.load(xp, allow_pickle=True) 
            self.labels = np.load(yp, allow_pickle=True)
                
    def __len__(self):
        return len(self.data) if self.mode == 'png' else len(self.images)

    def __getitem__(self, idx):
        fname = "unknown"
        if self.mode == 'png':
            path, label, fname = self.data[idx]
            img = Image.open(path).convert('RGB')
        else:
            img_arr = self.images[idx]
            label = int(self.labels[idx])
            fname = f"npy_idx_{idx}"
            
            if img_arr.ndim == 2: img_arr = np.stack((img_arr,)*3, axis=-1)
            elif img_arr.ndim == 3 and img_arr.shape[-1] == 1: img_arr = np.repeat(img_arr, 3, axis=-1)
            if img_arr.max() <= 1.5: img_arr = (img_arr * 255).astype(np.uint8)
            else: img_arr = img_arr.astype(np.uint8)
            img = Image.fromarray(img_arr)

        if self.transform: img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32), fname

def get_transforms(img_size, is_train=True):
    if is_train and Config.USE_AUGMENTATION:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def build_model():
    model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1) 
    return model

class Engine:
    def __init__(self, model, device, criterion, optimizer=None, scaler=None):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}

    def train_epoch(self, loader):
        self.model.train()
        losses, preds, labels = [], [], []
        for x, y, _ in tqdm(loader, desc="Train", leave=False):
            x, y = x.to(self.device), y.to(self.device).unsqueeze(1)
            with autocast(enabled=True): 
                logits = self.model(x)
                loss = self.criterion(logits, y)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            losses.append(loss.item())
            preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            labels.extend(y.cpu().numpy())
        
        try: 
            if len(np.unique(labels)) > 1: auc = roc_auc_score(labels, preds)
            else: auc = 0.5 
        except: auc = 0.5
        return np.mean(losses), auc

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        losses, probs, labels, filenames = [], [], [], []
        for x, y, fnames in loader:
            x, y = x.to(self.device), y.to(self.device).unsqueeze(1)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            losses.append(loss.item())
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(y.cpu().numpy())
            filenames.extend(fnames)
        
        y_true, y_prob = np.array(labels), np.array(probs)
        try:
            if len(np.unique(y_true)) > 1: auc = roc_auc_score(y_true, y_prob)
            else: auc = 0.5
        except: auc = 0.5
        
        try: brier = brier_score_loss(y_true, y_prob)
        except: brier = 0.0
        
        return np.mean(losses), auc, brier, y_prob, y_true, filenames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="png", choices=["png", "npy"])
    args = parser.parse_args()
    
    set_seed(Config.SEED)
    save_dir = Path("experiments_classification") / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # --- SETUP VISUALIZATION FOLDER ---
    vis_dir = save_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(save_dir, "Train")
    logger.info(f"Training Config: Augmentation={Config.USE_AUGMENTATION} | Batch={Config.BATCH_SIZE}")
    logger.info(f"Visual outputs will be saved to: {vis_dir}")
    
    try:
        train_ds = UniversalDataset(args.data_path, 'train', args.mode, get_transforms(Config.IMG_SIZE, True))
        val_ds   = UniversalDataset(args.data_path, 'val',   args.mode, get_transforms(Config.IMG_SIZE, False))
        test_ds  = UniversalDataset(args.data_path, 'test',  args.mode, get_transforms(Config.IMG_SIZE, False))
        
        logger.info(f"Loaded: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
        if len(val_ds) == 0: raise ValueError("Data Validasi Kosong!")

        loaders = {
            "train": DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS),
            "val":   DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS),
            "test":  DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
        }
    except Exception as e:
        logger.error(f"Error loading data: {e}"); return

    model = build_model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    scaler = GradScaler()
    engine = Engine(model, Config.DEVICE, criterion, optimizer, scaler)
    
    best_auc = 0.0
    patience_cnt = 0
    
    for epoch in range(1, Config.EPOCHS+1):
        t_loss, t_auc = engine.train_epoch(loaders['train'])
        v_loss, v_auc, v_brier, _, _, _ = engine.evaluate(loaders['val'])
        
        # Save History
        engine.history['train_loss'].append(t_loss)
        engine.history['train_auc'].append(t_auc)
        engine.history['val_loss'].append(v_loss)
        engine.history['val_auc'].append(v_auc)
        
        scheduler.step(v_auc)
        logger.info(f"Ep {epoch:02d} | T_Loss:{t_loss:.4f} T_AUC:{t_auc:.4f} | V_Loss:{v_loss:.4f} V_AUC:{v_auc:.4f} V_Brier:{v_brier:.4f}")
        
        if v_auc > best_auc:
            best_auc = v_auc
            patience_cnt = 0
            
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            logger.info("--> Saved Best Model")
            try:
                model.eval()
                dummy = torch.randn(1, 3, *Config.IMG_SIZE).to(Config.DEVICE)
                torch.onnx.export(model, dummy, save_dir / "best_model.onnx", 
                                  input_names=['input'], output_names=['output'],
                                  dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}},
                                  opset_version=11)
                model.train()
                logger.info("--> Saved ONNX")
            except Exception as e:
                logger.error(f"ONNX Export Failed: {e}")
        else:
            patience_cnt += 1
            if patience_cnt >= Config.PATIENCE: break
            
    # --- FINAL EVALUATION & EXPORT ---
    final_model_path = save_dir / "best_model.pt"
    if not final_model_path.exists():
        logger.warning("Best model tidak ditemukan. Menyimpan model epoch terakhir.")
        torch.save(model.state_dict(), final_model_path)
    
    model.load_state_dict(torch.load(final_model_path))
    
    # 1. Evaluate Test Set
    test_loss, test_auc, test_brier, y_prob, y_true, fnames = engine.evaluate(loaders['test'])
    y_pred = (y_prob > 0.5).astype(int)
    
    # 2. Metrics JSON (Save to ROOT folder)
    metrics = {
        "final_test_loss": test_loss,
        "final_test_auc": test_auc,
        "final_test_brier": test_brier,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    }
    with open(save_dir / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # 3. Save Plots (Save to VIS folder)
    plot_history(engine.history, vis_dir)
    plot_confusion_matrix(y_true, y_pred, vis_dir)
    plot_roc_curve(y_true, y_prob, vis_dir)
    
    # 4. Save CSV (Save to ROOT folder)
    df_preds = pd.DataFrame({
        "filename": fnames,
        "label_true": y_true.flatten(),
        "prob_pred": y_prob.flatten(),
        "label_pred": y_pred.flatten()
    })
    df_preds.to_csv(save_dir / "test_predictions.csv", index=False)
    
    logger.info("==========================================")
    logger.info(f"FINAL RESULTS saved to {save_dir}")
    logger.info(f"Visuals saved to {vis_dir}")
    logger.info(f"AUC: {test_auc:.4f} | Acc: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()