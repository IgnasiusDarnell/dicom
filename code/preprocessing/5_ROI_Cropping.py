import os
import glob
import shutil
import cv2
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from pathlib import Path
from tqdm import tqdm

# Import ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    ort = None

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- 1. KONFIGURASI ---
class Config:
    # Path Model (Ganti .pt atau .onnx sesuai kebutuhan)
    SEG_MODEL_PATH = "../segmentation/experiments/augmentation_run/mit_FPN_AUG_PHYSICAL_20251227/weights/best_model.onnx"
    
    BASE_DIR = Path("../../dataset")
    MANIFEST_PATH = BASE_DIR / "manifest.csv" 
    
    # Path Input
    PATHS = {
        "Baseline_NPY": BASE_DIR / "baseline/NPY",
        "Baseline_PNG": BASE_DIR / "baseline/PNG",
        "Hyfusion_NPY": BASE_DIR / "hyfusion_v2/NPY/hyfusion"
    }
    
    # Output
    OUTPUT_ROOT = Path("../../dataset/ROI_FINAL_GENERATED")
    
    # Settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE_MODEL = (512, 512) # Hanya untuk input ke Model Segmentasi
    # OUTPUT_SIZE_NPY = TIDAK DIGUNAKAN LAGI (Sesuai request No Resize)
    
    THRESHOLD = 0.5
    PADDING = 0.0 
    
    # Visualization
    EXPORT_DEBUG = True
    DEBUG_LIMIT = 50

# --- 2. UTILS ---
class GrayToRGB(torch.nn.Module):
    def forward(self, x): return x.repeat(1, 3, 1, 1)

def load_segmenter(path):
    path = str(path)
    print(f"Loading Segmenter: {path}")
    
    if path.endswith('.onnx'):
        if ort is None: raise ImportError("Library 'onnxruntime' belum terinstall.")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if Config.DEVICE == 'cuda' else ['CPUExecutionProvider']
        try:
            session = ort.InferenceSession(path, providers=providers)
            print(f"ONNX Session Loaded using: {session.get_providers()}")
            return session
        except Exception as e:
            print(f"Error loading ONNX with CUDA: {e}, using CPU.")
            return ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    else:
        model = smp.FPN(encoder_name="mit_b5", encoder_weights=None, in_channels=3, classes=1)
        full_model = torch.nn.Sequential(GrayToRGB(), model)
        state = torch.load(path, map_location=Config.DEVICE)
        full_model.load_state_dict(state)
        full_model.to(Config.DEVICE)
        full_model.eval()
        return full_model

def export_visualization(stem, save_dir, img_orig, mask, crop):
    """
    Format: Raw | Predict | Overlay | ROI CROP
    """
    # Pastikan folder debug ada DI DALAM folder output spesifik
    debug_dir = save_dir / "debug_visualization"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Raw Image (Gray -> BGR)
    img_c = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2BGR)
    
    # 2. Predict Mask (Binary -> BGR White)
    mask_vis = (mask * 255).astype(np.uint8)
    mask_c = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
    
    # 3. Overlay (Raw + Red Mask)
    # Buat mask merah
    zeros = np.zeros_like(mask_vis)
    mask_red = cv2.merge([zeros, zeros, mask_vis]) # BGR -> Red channel only
    # Overlay dengan addWeighted
    overlay_c = cv2.addWeighted(img_c, 0.7, mask_red, 0.5, 0)
    
    # 4. ROI Crop
    h_orig, w_orig = img_orig.shape
    h_crop, w_crop = crop.shape
    
    # Resize Crop height agar sama dengan original (hanya untuk visualisasi grid)
    scale = h_orig / h_crop if h_crop > 0 else 1
    new_w = int(w_crop * scale)
    crop_resized = cv2.resize(crop, (new_w, h_orig), interpolation=cv2.INTER_NEAREST)
    crop_c = cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2BGR)
    
    # Labels
    def add_label(img, text):
        cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return img
    
    img_c = add_label(img_c, "Raw")
    mask_c = add_label(mask_c, "Predict")
    overlay_c = add_label(overlay_c, "Overlay")
    crop_c = add_label(crop_c, "ROI Crop")
    
    # Combine
    combined = np.hstack([img_c, mask_c, overlay_c, crop_c])
    cv2.imwrite(str(debug_dir / f"VIS_{stem}.jpg"), combined)

def get_bbox_and_mask(img_orig, model):
    h_orig, w_orig = img_orig.shape
    
    # Resize hanya untuk Input Model
    img_small = cv2.resize(img_orig, Config.IMG_SIZE_MODEL)
    img_np = img_small.astype(np.float32) / 255.0
    img_np = np.expand_dims(img_np, axis=0)
    img_np = np.expand_dims(img_np, axis=0)
    
    if isinstance(model, torch.nn.Module):
        img_t = torch.from_numpy(img_np).to(Config.DEVICE)
        with torch.no_grad():
            logits = model(img_t)
            probs = torch.sigmoid(logits)
            pred_small = (probs > Config.THRESHOLD).cpu().numpy()[0, 0]
    else:
        input_name = model.get_inputs()[0].name
        input_shape = model.get_inputs()[0].shape
        if len(input_shape) == 4 and input_shape[1] == 3 and img_np.shape[1] == 1:
            img_np = np.repeat(img_np, 3, axis=1)
        logits = model.run(None, {input_name: img_np})[0]
        probs = 1 / (1 + np.exp(-logits))
        pred_small = (probs > Config.THRESHOLD)[0, 0]
        
    mask_orig = cv2.resize(pred_small.astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    
    rows = np.any(mask_orig, axis=1)
    cols = np.any(mask_orig, axis=0)
    
    if not np.any(rows) or not np.any(cols): return mask_orig, None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    h_box = y_max - y_min; w_box = x_max - x_min
    pad_y = int(h_box * Config.PADDING); pad_x = int(w_box * Config.PADDING)
    y_min = max(0, y_min - pad_y); y_max = min(h_orig, y_max + pad_y)
    x_min = max(0, x_min - pad_x); x_max = min(w_orig, x_max + pad_x)
    
    return mask_orig, (x_min, y_min, x_max, y_max)

# --- 3. PROCESSORS ---

def process_png_structure(input_root, output_root, model, df_manifest):
    print(f"\n--- Processing PNG Folder: {input_root.name} ---")
    all_files = list(input_root.rglob("*.png"))
    metadata = []
    debug_counter = 0
    
    manifest_dict = {}
    if df_manifest is not None:
        key_col = 'filename' if 'filename' in df_manifest.columns else 'src_dicom'
        if key_col in df_manifest.columns:
            manifest_dict = {str(Path(x).stem): row for x, row in zip(df_manifest[key_col], df_manifest.to_dict('records'))}

    for path in tqdm(all_files, desc="PNG Processing"):
        rel_path = path.relative_to(input_root)
        save_path = output_root / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        img_orig = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img_orig is None: continue
        
        h, w = img_orig.shape
        mask, bbox = get_bbox_and_mask(img_orig, model)
        
        # Masking (Background Black) -> Lalu Crop
        mask_uint8 = (mask * 255).astype(np.uint8)
        img_masked = cv2.bitwise_and(img_orig, img_orig, mask=mask_uint8)
        
        if bbox is None:
            final_img = img_masked; coords = (0, 0, w, h)
        else:
            x1, y1, x2, y2 = bbox
            # NO RESIZE HERE - Crop sesuai aslinya
            final_img = img_masked[y1:y2, x1:x2]
            coords = bbox
            
        cv2.imwrite(str(save_path), final_img)
        
        # Export Visual (Raw | Predict | Overlay | Crop)
        # Simpan di output_root/debug_visualization
        if Config.EXPORT_DEBUG and debug_counter < Config.DEBUG_LIMIT:
            export_visualization(path.stem, output_root, img_orig, mask, final_img)
            debug_counter += 1
            
        meta_row = {
            "generated_filename": str(rel_path),
            "crop_x1": coords[0], "crop_y1": coords[1], "crop_x2": coords[2], "crop_y2": coords[3],
            "crop_w_real": coords[2]-coords[0], "crop_h_real": coords[3]-coords[1]
        }
        if path.stem in manifest_dict:
            orig = manifest_dict[path.stem]
            for col in ['src_dicom', 'site', 'label', 'psnr', 'ssim', 'split']:
                if col in orig: meta_row[col] = orig[col]
        
        metadata.append(meta_row)
        
    pd.DataFrame(metadata).to_csv(output_root / "crop_coords_with_manifest.csv", index=False)
    print(f"Saved Metadata to: {output_root / 'crop_coords_with_manifest.csv'}")

def process_npy_structure(input_root, output_root, model, df_manifest):
    print(f"\n--- Processing NPY Folder: {input_root.name} ---")
    output_root.mkdir(parents=True, exist_ok=True)
    x_files = sorted(list(input_root.glob("X_*.npy"))) 
    
    for x_path in x_files:
        print(f"Processing Array: {x_path.name}")
        data = np.load(x_path)
        if data.ndim == 4: data = data.squeeze(-1)
        has_channel = True
        
        new_data = [] # Akan berisi array dengan ukuran BEDA-BEDA
        metadata_list = []
        debug_counter = 0
        
        current_split = "train" if "train" in x_path.name else "test" if "test" in x_path.name else "val"
        manifest_subset = df_manifest[df_manifest['split'] == current_split].reset_index(drop=True) if (df_manifest is not None and 'split' in df_manifest.columns) else None
        
        for i in tqdm(range(len(data)), desc=x_path.name):
            img = data[i]
            is_float = False
            if img.max() <= 1.5: img = (img * 255).astype(np.uint8); is_float = True
            else: img = img.astype(np.uint8)
                
            mask, bbox = get_bbox_and_mask(img, model)
            
            mask_uint8 = (mask * 255).astype(np.uint8)
            img_masked = cv2.bitwise_and(img, img, mask=mask_uint8)
            
            h, w = img.shape
            coords = (0, 0, w, h)
            if bbox is None: crop = img_masked
            else:
                x1, y1, x2, y2 = bbox
                # NO RESIZE - Crop Asli
                crop = img_masked[y1:y2, x1:x2]
                coords = bbox
            
            # Export Visual
            if Config.EXPORT_DEBUG and debug_counter < Config.DEBUG_LIMIT:
                export_visualization(f"{x_path.stem}_idx{i}", output_root, img, mask, crop)
                debug_counter += 1
            
            meta_row = {
                "index": i, "orig_w": w, "orig_h": h,
                "crop_x1": coords[0], "crop_y1": coords[1], "crop_x2": coords[2], "crop_y2": coords[3],
                "crop_w_real": coords[2]-coords[0], "crop_h_real": coords[3]-coords[1]
            }
            if manifest_subset is not None and i < len(manifest_subset):
                orig = manifest_subset.iloc[i]
                for col in ['src_dicom', 'site', 'label', 'psnr', 'ssim']:
                    if col in orig: meta_row[col] = orig[col]

            metadata_list.append(meta_row)

            # Kembalikan ke float jika perlu, tapi TETAP UKURAN ASLI
            if is_float: crop = crop.astype(np.float32) / 255.0
            new_data.append(crop)
            
        # --- SAVE NPY AS OBJECT ARRAY ---
        # Karena ukuran crop beda-beda, kita tidak bisa pakai np.array() biasa
        # Kita pakai dtype=object untuk menyimpan list of arrays
        new_data_arr = np.array(new_data, dtype=object)
        np.save(output_root / x_path.name, new_data_arr, allow_pickle=True)
        
        meta_name = x_path.stem + "_coords_with_manifest.csv"
        pd.DataFrame(metadata_list).to_csv(output_root / meta_name, index=False)
        
    y_files = list(input_root.glob("Y_*.npy")) + list(input_root.glob("y_*.npy"))
    for y_path in y_files: shutil.copy(y_path, output_root / y_path.name)

# --- 4. MAIN ---
if __name__ == "__main__":
    if not os.path.exists(Config.SEG_MODEL_PATH):
        print(f"Error: Model tidak ditemukan di {Config.SEG_MODEL_PATH}"); exit()
    
    df_manifest = None
    if Config.MANIFEST_PATH.exists():
        print(f"Loading Manifest from: {Config.MANIFEST_PATH}")
        df_manifest = pd.read_csv(Config.MANIFEST_PATH)
    else:
        print("Warning: Manifest file tidak ditemukan!")
        
    model = load_segmenter(Config.SEG_MODEL_PATH)
    
    # Proses
    process_npy_structure(Config.PATHS["Baseline_NPY"], Config.OUTPUT_ROOT / "Baseline_NPY_ROI", model, df_manifest)
    process_png_structure(Config.PATHS["Baseline_PNG"], Config.OUTPUT_ROOT / "Baseline_PNG_ROI", model, df_manifest)
    process_npy_structure(Config.PATHS["Hyfusion_NPY"], Config.OUTPUT_ROOT / "Hyfusion_NPY_ROI", model, df_manifest)
    
    print("\nAll Processing Completed!")
    print(f"Results saved in: {Config.OUTPUT_ROOT}")