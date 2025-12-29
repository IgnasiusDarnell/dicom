import os
import sys
import glob
import re
import time
import json
import random
import logging
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import cv2

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

import albumentations as A
import segmentation_models_pytorch as smp
from monai.losses import DiceCELoss
from sklearn.model_selection import train_test_split
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- 1. CONFIGURATION ---
class Config:
    # Path Setup
    DATA_ROOT = Path("/workspace/abusalam_dsn/darnell/dokumentasi/dicom/dataset_segmentation/segmentasi/dataset/final_dataset")
    OUTPUT_DIR = Path("./experiments/augmentation_run")
    TEMP_AUG_ROOT = Path("dataset/temp_aug_training") 
    
    # Model Setup
    MODEL_NAME = "mit_FPN"
    ENCODER = "mit_b5"
    IN_CHANNELS = 1
    CLASSES = 1
    
    # Augmentation Settings
    AUG_REPEAT = 2     
    
    # Training Hyperparameters
    IMG_SIZE = (512, 512) 
    BATCH_SIZE = 8
    EPOCHS = 100
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 12
    CLIP_LIMIT = 3.0
    
    # System
    NUM_WORKERS = 4
    SEED = 42 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

# --- 2. UTILS & LOGGING ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def setup_logger(save_dir):
    logger = logging.getLogger("aug_train")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(save_dir / "training.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def make_dirs(cfg):
    run_id = datetime.now().strftime("%Y%m%d")
    exp_dir = cfg.OUTPUT_DIR / f"{cfg.MODEL_NAME}_AUG_PHYSICAL_{run_id}"
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(exp_dir / "weights", exist_ok=True)
    os.makedirs(exp_dir / "plots", exist_ok=True)
    return exp_dir

# --- 3. PHYSICAL AUGMENTATION LOGIC ---
def get_aug_pipeline():
    return A.Compose([
        A.Rotate(limit=10, p=0.8),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.05, rotate_limit=0,
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.8
        ),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    ], p=1.0)

def generate_augmented_data(files, root_dir, split_name, repeat=0, pipeline=None):
    img_dir = root_dir / split_name / "images"
    msk_dir = root_dir / split_name / "masks"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)
    
    print(f"Generating physical dataset for [{split_name}]...")
    
    for item in tqdm(files):
        # Baca Original
        img = cv2.imread(item["image"]) 
        msk = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE)
        stem = Path(item["image"]).stem
        
        # 1. Simpan Original
        cv2.imwrite(str(img_dir / f"{stem}.png"), img)
        cv2.imwrite(str(msk_dir / f"{stem}.png"), msk)
        
        # 2. Lakukan Augmentasi (Jika ada pipeline)
        if pipeline and repeat > 0:
            for i in range(repeat):
                aug = pipeline(image=img, mask=msk)
                aug_img, aug_msk = aug["image"], aug["mask"]
                
                # Simpan dengan suffix _aug_X
                cv2.imwrite(str(img_dir / f"{stem}_aug_{i}.png"), aug_img)
                cv2.imwrite(str(msk_dir / f"{stem}_aug_{i}.png"), aug_msk)

# --- 4. DATASET ---
class SimpleDataset(Dataset):
    def __init__(self, img_dir, msk_dir, size=(512, 512), clip_limit=3.0):
        self.size = size
        self.clahe = A.CLAHE(clip_limit=clip_limit, tile_grid_size=(8, 8), p=1.0)

        img_paths = sorted(glob.glob(str(Path(img_dir) / "*.png")))
        msk_paths = sorted(glob.glob(str(Path(msk_dir) / "*.png")))

        # build paired list
        self.files = []
        for img_path in img_paths:
            stem = Path(img_path).stem
            mask_path = Path(msk_dir) / f"{stem}.png"
            if mask_path.exists():
                self.files.append({
                    "image": img_path,
                    "mask": str(mask_path)
                })

        if len(self.files) == 0:
            raise RuntimeError(f"No paired image-mask found in {img_dir}")

    def _resize_pad(self, img, is_mask=False):
        th, tw = self.size
        h, w = img.shape[:2]

        scale = min(th / max(1, h), tw / max(1, w))
        nh, nw = int(round(h * scale)), int(round(w * scale))

        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        resized = cv2.resize(img, (nw, nh), interpolation=interp)

        ph, pw = th - nh, tw - nw
        top, bottom = ph // 2, ph - ph // 2
        left, right = pw // 2, pw - pw // 2

        return cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=0
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.files[idx]

        img = cv2.imread(item["image"], cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(item["image"])
        if msk is None:
            raise FileNotFoundError(item["mask"])

        img = self.clahe(image=img)["image"]

        img = self._resize_pad(img, is_mask=False)
        msk = self._resize_pad(msk, is_mask=True)

        img = img.astype(np.float32) / 255.0
        msk = (msk > 127).astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        msk = torch.from_numpy(msk).unsqueeze(0)

        return {
            "image": img,
            "mask": msk,
            "path": item["image"]
        }


def prepare_and_load_data(cfg):
    # 1. Logic Split SAMA PERSIS dengan Baseline
    img_dir = cfg.DATA_ROOT / "images"
    mask_dir = cfg.DATA_ROOT / "masks"
    imgs = sorted(glob.glob(str(img_dir / "*.png")))
    pairs = []
    for img_path in imgs:
        stem = Path(img_path).stem
        
        mask_path = mask_dir / f"{stem}.png" 
        
        if mask_path.exists():
            pairs.append({"image": str(img_path), "mask": str(mask_path)})
            
    if not pairs: raise RuntimeError("Data tidak ditemukan! Cek path dataset.")

    train_val, test_files = train_test_split(pairs, test_size=0.05, random_state=cfg.SEED, shuffle=True)
    train_files, val_files = train_test_split(train_val, test_size=0.05/0.95, random_state=cfg.SEED, shuffle=True)

    # 2. Reset Temp Folder
    if cfg.TEMP_AUG_ROOT.exists():
        try:
            shutil.rmtree(cfg.TEMP_AUG_ROOT)
        except:
            pass # Ignore if busy
    
    # 3. Generate Fisik
    aug_pipe = get_aug_pipeline()
    
    # Train: Original + Aug
    generate_augmented_data(train_files, cfg.TEMP_AUG_ROOT, "train", repeat=cfg.AUG_REPEAT, pipeline=aug_pipe)
    # Val & Test: Original Only
    generate_augmented_data(val_files, cfg.TEMP_AUG_ROOT, "val", repeat=0)
    generate_augmented_data(test_files, cfg.TEMP_AUG_ROOT, "test", repeat=0)

    # 4. Create Loaders
    train_ds = SimpleDataset(cfg.TEMP_AUG_ROOT / "train" / "images", cfg.TEMP_AUG_ROOT / "train" / "masks", cfg.IMG_SIZE)
    val_ds   = SimpleDataset(cfg.TEMP_AUG_ROOT / "val" / "images",   cfg.TEMP_AUG_ROOT / "val" / "masks",   cfg.IMG_SIZE)
    test_ds  = SimpleDataset(cfg.TEMP_AUG_ROOT / "test" / "images",  cfg.TEMP_AUG_ROOT / "test" / "masks",  cfg.IMG_SIZE)

    loaders = {
        "train": DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True),
        "val":   DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True),
        "test":  DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    }
    
    return loaders, len(train_ds), len(val_ds), len(test_ds)

# --- 5. MODEL ARCHITECTURE ---
class GrayToRGB(nn.Module):
    def forward(self, x): return x.repeat(1, 3, 1, 1)

class CXR_Segmenter(nn.Module):
    def __init__(self, encoder_name="mit_b5", classes=1):
        super().__init__()
        self.adapter = GrayToRGB()
        self.backbone = smp.FPN(encoder_name=encoder_name, encoder_weights="imagenet", in_channels=3, classes=classes)
    def forward(self, x): return self.backbone(self.adapter(x))

# --- 6. TRAINER ENGINE ---
class Trainer:
    def __init__(self, model, loaders, cfg, save_dir, logger):
        self.model = model.to(cfg.DEVICE)
        self.loaders = loaders
        self.cfg = cfg
        self.save_dir = save_dir
        self.logger = logger
        
        self.criterion = DiceCELoss(sigmoid=True, squared_pred=True, smooth_nr=1e-5, smooth_dr=1e-5)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
        self.scaler = GradScaler()
        
        self.best_dice = -1.0
        self.history = []

    def _calc_metrics(self, preds, targets):
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = (preds + targets).sum(dim=(1, 2, 3)) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        dice = (2 * intersection + 1e-7) / (preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + 1e-7)
        return dice.mean().item(), iou.mean().item()

    def train_one_epoch(self):
        self.model.train()
        losses, dices, ious = [], [], []
        
        pbar = tqdm(self.loaders['train'], desc="Train", leave=False)
        for batch in pbar:
            x = batch["image"].to(self.cfg.DEVICE)
            y = batch["mask"].to(self.cfg.DEVICE)
            
            with autocast(enabled=True):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            losses.append(loss.item())
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                d, i = self._calc_metrics(preds, y)
                dices.append(d); ious.append(i)
                
        return np.mean(losses), np.mean(dices), np.mean(ious)

    @torch.no_grad()
    def validate(self, split='val'):
        self.model.eval()
        losses, dices, ious = [], [], []
        
        for batch in self.loaders[split]:
            x = batch["image"].to(self.cfg.DEVICE)
            y = batch["mask"].to(self.cfg.DEVICE)
            
            with autocast(enabled=True):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            d, i = self._calc_metrics(preds, y)
            
            losses.append(loss.item())
            dices.append(d)
            ious.append(i)
            
        return np.mean(losses), np.mean(dices), np.mean(ious)

    def fit(self):
        start_time = time.time()
        early_stop_counter = 0
        
        self.logger.info(f"Start Training: {self.cfg.MODEL_NAME} (Augmented Physical) | Epochs: {self.cfg.EPOCHS}")
        
        for epoch in range(1, self.cfg.EPOCHS + 1):
            t_loss, t_dice, t_iou = self.train_one_epoch()
            v_loss, v_dice, v_iou = self.validate('val')
            self.scheduler.step()
            
            self.logger.info(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {t_loss:.4f} Dice: {t_dice:.4f} | "
                f"Val Loss: {v_loss:.4f} Dice: {v_dice:.4f} IoU: {v_iou:.4f}"
            )
            
            self.history.append({
                "epoch": epoch,
                "train_loss": t_loss, "train_dice": t_dice,
                "val_loss": v_loss, "val_dice": v_dice, "val_iou": v_iou,
                "lr": self.optimizer.param_groups[0]['lr']
            })
            
            if v_dice > self.best_dice:
                self.best_dice = v_dice
                early_stop_counter = 0
                torch.save(self.model.state_dict(), self.save_dir / "weights" / "best_model.pt")
                self.logger.info(f"--> Saved Best Model (Dice: {self.best_dice:.4f})")
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= self.cfg.PATIENCE:
                self.logger.info("Early stopping triggered.")
                break
        
        total_time = (time.time() - start_time) / 60
        self.logger.info(f"Training finished in {total_time:.1f} minutes.")
        
        pd.DataFrame(self.history).to_csv(self.save_dir / "history.csv", index=False)

# --- 7. EXPORT RESULTS  ---
def export_results(trainer):
    save_dir = trainer.save_dir
    
    # 1. Plot Metrics
    df = pd.DataFrame(trainer.history)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train')
    plt.plot(df['epoch'], df['val_loss'], label='Val')
    plt.title('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['val_dice'], label='Val Dice')
    plt.plot(df['epoch'], df['val_iou'], label='Val IoU')
    plt.title('Metrics'); plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "plots" / "training_curves.png", dpi=300)
    plt.close()
    
    # 2. Evaluation on Test Set
    trainer.model.load_state_dict(torch.load(save_dir / "weights" / "best_model.pt"))
    test_loss, test_dice, test_iou = trainer.validate('test')
    
    results = {
        "model": trainer.cfg.MODEL_NAME + "_AUG",
        "test_loss": test_loss,
        "test_dice": test_dice,
        "test_iou": test_iou,
        "best_val_dice": trainer.best_dice
    }
    
    with open(save_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=4)
    trainer.logger.info(f"FINAL TEST RESULTS: {results}")
    
    # 3. Visual Samples 
    trainer.model.eval()
    batch = next(iter(trainer.loaders['test']))
    images = batch["image"][:5].to(trainer.cfg.DEVICE)
    masks = batch["mask"][:5]
    
    with torch.no_grad():
        preds = torch.sigmoid(trainer.model(images)) > 0.5
        
    fig, axes = plt.subplots(5, 3, figsize=(10, 15))
    for i in range(5):
        # Input
        axes[i,0].imshow(images[i,0].cpu().numpy(), cmap='gray')
        axes[i,0].set_title("Input"); axes[i,0].axis('off')
        # GT
        axes[i,1].imshow(masks[i,0].numpy(), cmap='gray')
        axes[i,1].set_title("Ground Truth"); axes[i,1].axis('off')
        # Pred + Overlay
        axes[i,2].imshow(images[i,0].cpu().numpy(), cmap='gray')
        axes[i,2].imshow(preds[i,0].cpu().numpy(), cmap='Reds', alpha=0.4)
        axes[i,2].set_title("Prediction"); axes[i,2].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_dir / "plots" / "prediction_samples.png", dpi=300)
    plt.close()
    
    # 4. Export ONNX
    dummy = torch.randn(1, 1, *trainer.cfg.IMG_SIZE).to(trainer.cfg.DEVICE)
    torch.onnx.export(
        trainer.model, dummy, save_dir / "weights" / "model.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11
    )
    trainer.logger.info("ONNX Model exported.")

# --- 8. MAIN EXECUTION ---
if __name__ == "__main__":
    cfg = Config()
    set_seed(cfg.SEED)
    
    # Setup Dirs & Logger
    save_dir = make_dirs(cfg)
    logger = setup_logger(save_dir)
    
    logger.info(f"Initialized Experiment. Output: {save_dir}")
    logger.info(f"Device: {cfg.DEVICE}")
    
    try:
        # Load Data 
        loaders, n_train, n_val, n_test = prepare_and_load_data(cfg)
        logger.info(f"Dataset Split (Based on Original): Train={n_train}, Val={n_val}, Test={n_test}")
        logger.info(f"Note: Train size includes Augmented Copies (Repeat={cfg.AUG_REPEAT})")
        
        # Build Model
        model = CXR_Segmenter(encoder_name=cfg.ENCODER, classes=cfg.CLASSES)
        logger.info(f"Model Built: {cfg.MODEL_NAME} with {cfg.ENCODER}")
        
        # Train
        trainer = Trainer(model, loaders, cfg, save_dir, logger)
        trainer.fit()
        
        # Export Assets 
        export_results(trainer)
        
        logger.info("ALL DONE SUCCESSFULLY.")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        sys.exit(1)