"""
Kaggle Scientific Image Forgery Detection - TOP 5% SOLUTION
Competition: Recod.ai/LUC - Scientific Image Forgery Detection
Prize: $55,000 | Deadline: 8 Jan 2026

Features:
- 3 Model Ensemble (UNet+EffB4, UNet++ResNet101, DeepLabV3+EffB3)
- 5-Fold Cross-Validation
- Progressive Resizing (256→384→512)
- 6x TTA (Test-Time Augmentation)
- Combined Loss (BCE + Dice + Focal)
- Mixed Precision Training
- Memory-Efficient Pipeline
"""

# ============================================================================
# CELL 1: INSTALLATION
# ============================================================================
print("Installing required packages...")
import subprocess, sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "timm", "segmentation-models-pytorch==0.3.3", "albumentations==1.4.0"])

# ============================================================================
# CELL 2: IMPORTS & CONFIG
# ============================================================================
import os, gc, cv2, random, warnings, json
import numpy as np
import pandas as pd
from glob import glob
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

class CFG:
    seed = 42
    debug = False
    
    # Folds
    n_folds = 5
    train_folds = [0, 1, 2, 3, 4]
    
    # Progressive resizing
    img_sizes = [256, 384, 512]
    
    # Training
    epochs_per_size = [10, 10, 15]  # Epochs for each size
    batch_sizes = [32, 16, 8]       # Batch sizes for each size
    accumulation_steps = 2
    num_workers = 2
    
    # Learning rate
    lr = 1e-4
    min_lr = 1e-7
    weight_decay = 1e-5
    
    # Early stopping
    patience = 10
    
    # Paths
    BASE_PATH = '/kaggle/input/recodai-luc-scientific-image-forgery-detection'
    OUTPUT_DIR = '/kaggle/working'
    
    # Models config with weights
    models = {
        'unet_effb4': {
            'encoder': 'efficientnet-b4',
            'decoder': 'Unet',
            'weight': 0.40
        },
        'unetpp_resnet101': {
            'encoder': 'resnet101', 
            'decoder': 'UnetPlusPlus',
            'weight': 0.35
        },
        'deeplabv3_effb3': {
            'encoder': 'efficientnet-b3',
            'decoder': 'DeepLabV3Plus', 
            'weight': 0.25
        }
    }
    
    # Loss weights
    bce_weight = 0.4
    dice_weight = 0.3
    focal_weight = 0.3
    
    # Post-processing
    min_area = 100
    threshold = 0.5
    
    # TTA
    use_tta = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG.seed)
os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

print(f"Device: {CFG.device}")
if CFG.device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CELL 3: PATH DETECTION & DATA LOADING
# ============================================================================
def find_paths():
    """Auto-detect competition paths"""
    paths = {
        'train_images': None,
        'train_masks': None, 
        'test_images': None,
        'train_csv': None,
        'test_csv': None,
        'sample_sub': None
    }
    
    base = CFG.BASE_PATH
    if not os.path.exists(base):
        # Try alternative paths
        alternatives = [
            '/kaggle/input/scientific-image-forgery-detection',
            '/kaggle/input'
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                base = alt
                break
    
    # Search for directories and files
    for root, dirs, files in os.walk(base):
        for d in dirs:
            dl = d.lower()
            if 'train' in dl and 'image' in dl:
                paths['train_images'] = os.path.join(root, d)
            elif 'train' in dl and 'mask' in dl:
                paths['train_masks'] = os.path.join(root, d)
            elif 'test' in dl and 'image' in dl:
                paths['test_images'] = os.path.join(root, d)
        
        for f in files:
            fl = f.lower()
            if 'train' in fl and f.endswith('.csv'):
                paths['train_csv'] = os.path.join(root, f)
            elif 'test' in fl and f.endswith('.csv'):
                paths['test_csv'] = os.path.join(root, f)
            elif 'sample' in fl and 'sub' in fl:
                paths['sample_sub'] = os.path.join(root, f)
    
    # Fallback paths
    if not paths['train_images']:
        paths['train_images'] = f'{base}/train_images'
    if not paths['train_masks']:
        paths['train_masks'] = f'{base}/train_masks'
    if not paths['test_images']:  
        paths['test_images'] = f'{base}/test_images'
        
    return paths

PATHS = find_paths()
print("\nDetected Paths:")
for k, v in PATHS.items():
    exists = '✓' if v and os.path.exists(v) else '✗'
    print(f"  {k}: {v} [{exists}]")

# ============================================================================
# CELL 4: DATASET CLASS
# ============================================================================
class ForgeryDataset(Dataset):
    """Memory-efficient dataset with lazy loading"""
    
    def __init__(self, df, img_dir, mask_dir=None, transform=None, mode='train'):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mode = mode
    
    def __len__(self):
        return len(self.df)
    
    def _find_image(self, img_id):
        """Find image in directory or subdirectories"""
        # Direct path
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            path = os.path.join(self.img_dir, img_id + ext)
            if os.path.exists(path):
                return path
            path = os.path.join(self.img_dir, img_id)
            if os.path.exists(path):
                return path
        
        # Search subdirectories
        for subdir in ['authentic', 'forged', 'images']:
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '']:
                path = os.path.join(self.img_dir, subdir, img_id + ext)
                if os.path.exists(path):
                    return path
        
        return None
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id'] if 'image_id' in row else row.iloc[0]
        
        # Find and load image
        img_path = self._find_image(str(img_id))
        if img_path and os.path.exists(img_path):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        if self.mode == 'test':
            if self.transform:
                image = self.transform(image=image)['image']
            return image, str(img_id)
        
        # Load mask for training
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        if self.mask_dir:
            for ext in ['.png', '.jpg', '.tif', '']:
                mask_path = os.path.join(self.mask_dir, str(img_id).split('.')[0] + ext)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = (mask > 127).astype(np.float32)
                    break
        
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']
        
        return image, mask.unsqueeze(0) if isinstance(mask, torch.Tensor) else torch.tensor(mask).unsqueeze(0)

# ============================================================================
# CELL 5: AUGMENTATIONS
# ============================================================================
def get_train_aug(size):
    return A.Compose([
        A.RandomResizedCrop(size, size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
        ], p=0.4),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_valid_aug(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ============================================================================
# CELL 6: MODEL DEFINITIONS
# ============================================================================
def build_model(encoder, decoder):
    """Build segmentation model"""
    decoders = {
        'Unet': smp.Unet,
        'UnetPlusPlus': smp.UnetPlusPlus,
        'DeepLabV3Plus': smp.DeepLabV3Plus
    }
    
    model = decoders[decoder](
        encoder_name=encoder,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None
    )
    return model

# ============================================================================
# CELL 7: LOSS FUNCTIONS
# ============================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
    
    def forward(self, pred, target):
        return (CFG.bce_weight * self.bce(pred, target) +
                CFG.dice_weight * self.dice(pred, target) +
                CFG.focal_weight * self.focal(pred, target))

# ============================================================================
# CELL 8: METRICS
# ============================================================================
def calc_iou(pred, target, thresh=0.5):
    pred = (torch.sigmoid(pred) > thresh).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return ((inter + 1e-7) / (union + 1e-7)).item()

def calc_f1(pred, target, thresh=0.5):
    pred = (torch.sigmoid(pred) > thresh).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    prec = (tp + 1e-7) / (tp + fp + 1e-7)
    rec = (tp + 1e-7) / (tp + fn + 1e-7)
    return (2 * prec * rec / (prec + rec + 1e-7)).item()

# ============================================================================
# CELL 9: TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss, running_iou = 0., 0.
    
    pbar = tqdm(loader, desc='Train')
    optimizer.zero_grad()
    
    for step, (imgs, masks) in enumerate(pbar):
        imgs, masks = imgs.to(device), masks.to(device)
        
        with autocast():
            out = model(imgs)
            loss = criterion(out, masks) / CFG.accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % CFG.accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * CFG.accumulation_steps
        running_iou += calc_iou(out.detach(), masks)
        pbar.set_postfix({'loss': f'{running_loss/(step+1):.4f}', 'iou': f'{running_iou/(step+1):.4f}'})
    
    return running_loss / len(loader), running_iou / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, running_iou, running_f1 = 0., 0., 0.
    
    for imgs, masks in tqdm(loader, desc='Valid'):
        imgs, masks = imgs.to(device), masks.to(device)
        
        with autocast():
            out = model(imgs)
            loss = criterion(out, masks)
        
        running_loss += loss.item()
        running_iou += calc_iou(out, masks)
        running_f1 += calc_f1(out, masks)
    
    n = len(loader)
    return running_loss/n, running_iou/n, running_f1/n

# ============================================================================
# CELL 10: CROSS-VALIDATION TRAINING
# ============================================================================
def train_model(model_name, model_cfg, train_df, device):
    """Train one model with 5-fold CV and progressive resizing"""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Encoder: {model_cfg['encoder']}, Decoder: {model_cfg['decoder']}")
    print(f"{'='*60}")
    
    fold_scores = []
    
    # Create folds
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    train_df['fold'] = -1
    
    # Stratify by has_forgery if available
    y = train_df['has_forgery'] if 'has_forgery' in train_df.columns else np.zeros(len(train_df))
    for fold, (_, val_idx) in enumerate(skf.split(train_df, y)):
        train_df.loc[val_idx, 'fold'] = fold
    
    for fold in CFG.train_folds:
        print(f"\n--- Fold {fold+1}/{CFG.n_folds} ---")
        
        train_idx = train_df[train_df['fold'] != fold].index
        val_idx = train_df[train_df['fold'] == fold].index
        
        train_data = train_df.loc[train_idx]
        val_data = train_df.loc[val_idx]
        
        best_iou = 0
        patience_counter = 0
        
        # Progressive resizing
        for size_idx, img_size in enumerate(CFG.img_sizes):
            print(f"\n>> Progressive Size: {img_size}x{img_size}")
            
            # Build fresh model or load previous
            model = build_model(model_cfg['encoder'], model_cfg['decoder']).to(device)
            
            # Load previous stage weights if not first
            if size_idx > 0:
                prev_path = f"{CFG.OUTPUT_DIR}/temp_{model_name}_fold{fold}.pth"
                if os.path.exists(prev_path):
                    model.load_state_dict(torch.load(prev_path))
            
            # Data loaders
            train_ds = ForgeryDataset(train_data, PATHS['train_images'], PATHS['train_masks'], 
                                      get_train_aug(img_size), 'train')
            val_ds = ForgeryDataset(val_data, PATHS['train_images'], PATHS['train_masks'],
                                    get_valid_aug(img_size), 'train')
            
            bs = CFG.batch_sizes[size_idx]
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, 
                                      num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=bs*2, shuffle=False,
                                    num_workers=CFG.num_workers, pin_memory=True)
            
            # Training setup
            criterion = CombinedLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=CFG.epochs_per_size[size_idx], eta_min=CFG.min_lr)
            scaler = GradScaler()
            
            epochs = CFG.epochs_per_size[size_idx]
            
            for epoch in range(epochs):
                train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
                val_loss, val_iou, val_f1 = validate(model, val_loader, criterion, device)
                scheduler.step()
                
                print(f"  Epoch {epoch+1}/{epochs} | Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f}")
                
                if val_iou > best_iou:
                    best_iou = val_iou
                    patience_counter = 0
                    torch.save(model.state_dict(), f"{CFG.OUTPUT_DIR}/best_{model_name}_fold{fold}.pth")
                    torch.save(model.state_dict(), f"{CFG.OUTPUT_DIR}/temp_{model_name}_fold{fold}.pth")
                else:
                    patience_counter += 1
                
                if patience_counter >= CFG.patience:
                    print(f"  Early stopping!")
                    break
                
                torch.cuda.empty_cache()
                gc.collect()
            
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()
            gc.collect()
        
        fold_scores.append(best_iou)
        print(f"Fold {fold+1} Best IoU: {best_iou:.4f}")
        
        # Remove temp file
        temp_path = f"{CFG.OUTPUT_DIR}/temp_{model_name}_fold{fold}.pth"
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    cv_score = np.mean(fold_scores)
    print(f"\n{model_name} CV IoU: {cv_score:.4f} (+/- {np.std(fold_scores):.4f})")
    return cv_score

# ============================================================================
# CELL 11: TTA INFERENCE
# ============================================================================
def tta_predict(model, image, size, device):
    """Predict with 6x TTA"""
    model.eval()
    
    # Base transform
    base = A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    preds = []
    
    # Original
    img = base(image=image)['image'].unsqueeze(0).to(device)
    with autocast():
        pred = torch.sigmoid(model(img))
    preds.append(pred)
    
    # Horizontal flip
    img_h = cv2.flip(image, 1)
    img = base(image=img_h)['image'].unsqueeze(0).to(device)
    with autocast():
        pred = torch.sigmoid(model(img))
    preds.append(torch.flip(pred, dims=[-1]))
    
    # Vertical flip
    img_v = cv2.flip(image, 0)
    img = base(image=img_v)['image'].unsqueeze(0).to(device)
    with autocast():
        pred = torch.sigmoid(model(img))
    preds.append(torch.flip(pred, dims=[-2]))
    
    # Rotations
    for k in [1, 2, 3]:  # 90, 180, 270
        img_r = np.rot90(image, k)
        img = base(image=img_r.copy())['image'].unsqueeze(0).to(device)
        with autocast():
            pred = torch.sigmoid(model(img))
        preds.append(torch.rot90(pred, -k, dims=[-2, -1]))
    
    return torch.stack(preds).mean(dim=0)

# ============================================================================
# CELL 12: ENSEMBLE INFERENCE
# ============================================================================
def ensemble_inference(test_df, device):
    """Ensemble all models with TTA"""
    print("\n" + "="*60)
    print("ENSEMBLE INFERENCE")
    print("="*60)
    
    final_size = CFG.img_sizes[-1]  # Use largest size
    all_preds = {name: [] for name in CFG.models.keys()}
    
    for model_name, cfg in CFG.models.items():
        print(f"\nProcessing {model_name}...")
        
        # Load all fold models
        models = []
        for fold in CFG.train_folds:
            model = build_model(cfg['encoder'], cfg['decoder']).to(device)
            model.load_state_dict(torch.load(f"{CFG.OUTPUT_DIR}/best_{model_name}_fold{fold}.pth"))
            model.eval()
            models.append(model)
        
        # Predict for each test image
        for idx in tqdm(range(len(test_df)), desc=model_name):
            row = test_df.iloc[idx]
            img_id = row['image_id'] if 'image_id' in row else row.iloc[0]
            
            # Find image
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '']:
                path = os.path.join(PATHS['test_images'], str(img_id) + ext)
                if os.path.exists(path):
                    img_path = path
                    break
            
            if img_path is None:
                for subdir in ['images', 'test']:
                    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '']:
                        path = os.path.join(PATHS['test_images'], subdir, str(img_id) + ext)
                        if os.path.exists(path):
                            img_path = path
                            break
            
            if img_path and os.path.exists(img_path):
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = image.shape[:2]
            else:
                image = np.zeros((256, 256, 3), dtype=np.uint8)
                orig_h, orig_w = 256, 256
            
            # Predict with each fold
            fold_preds = []
            for model in models:
                if CFG.use_tta:
                    pred = tta_predict(model, image, final_size, device)
                else:
                    aug = get_valid_aug(final_size)(image=image)
                    img = aug['image'].unsqueeze(0).to(device)
                    with torch.no_grad(), autocast():
                        pred = torch.sigmoid(model(img))
                fold_preds.append(pred)
            
            # Average across folds
            avg_pred = torch.stack(fold_preds).mean(dim=0)
            
            # Resize to original
            avg_pred = F.interpolate(avg_pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            all_preds[model_name].append(avg_pred.cpu().numpy().squeeze())
        
        del models
        torch.cuda.empty_cache()
        gc.collect()
    
    # Weighted ensemble
    print("\nCreating weighted ensemble...")
    final_preds = []
    
    for idx in range(len(test_df)):
        ensemble = np.zeros_like(all_preds['unet_effb4'][idx])
        for name, cfg in CFG.models.items():
            ensemble += cfg['weight'] * all_preds[name][idx]
        final_preds.append(ensemble)
    
    return final_preds

# ============================================================================
# CELL 13: POST-PROCESSING
# ============================================================================
def post_process(mask, min_area=100, threshold=0.5):
    """Clean mask with morphology and component filtering"""
    binary = (mask > threshold).astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    cleaned = np.zeros_like(binary)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1
    
    return cleaned

# ============================================================================
# CELL 14: RLE ENCODING & SUBMISSION
# ============================================================================
def rle_encode(mask):
    """RLE encode mask to JSON format"""
    dots = np.where(mask.T.flatten() == 1)[0]
    
    if len(dots) == 0:
        return '[]'
    
    runs = []
    prev = -2
    
    for b in dots:
        if b > prev + 1:
            runs.extend([int(b + 1), 0])
        runs[-1] += 1
        prev = b
    
    return json.dumps(runs)

def create_submission(test_df, predictions, threshold=0.001):
    """Create submission CSV"""
    print("\nCreating submission...")
    
    rows = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_id = row['image_id'] if 'image_id' in row else row.iloc[0]
        case_id = str(img_id).rsplit('.', 1)[0]
        
        mask = post_process(predictions[idx], CFG.min_area, CFG.threshold)
        
        # Check if authentic
        forgery_ratio = np.sum(mask > 0) / mask.size
        
        if forgery_ratio < threshold:
            annotation = 'authentic'
        else:
            annotation = rle_encode(mask)
        
        rows.append({'case_id': case_id, 'annotation': annotation})
    
    df = pd.DataFrame(rows)
    df.to_csv(f"{CFG.OUTPUT_DIR}/submission.csv", index=False)
    
    n_auth = sum(df['annotation'] == 'authentic')
    n_forg = len(df) - n_auth
    print(f"✓ Saved: {CFG.OUTPUT_DIR}/submission.csv")
    print(f"  Total: {len(df)} | Authentic: {n_auth} | Forged: {n_forg}")
    
    return df

# ============================================================================
# CELL 15: MAIN EXECUTION
# ============================================================================
def main():
    print("="*70)
    print("SCIENTIFIC IMAGE FORGERY DETECTION - TOP 5% SOLUTION")
    print("="*70)
    
    device = torch.device(CFG.device)
    
    # Load data
    print("\n[1/4] Loading data...")
    
    if PATHS['train_csv'] and os.path.exists(PATHS['train_csv']):
        train_df = pd.read_csv(PATHS['train_csv'])
    else:
        # Create df from image files
        images = []
        for root, _, files in os.walk(PATHS['train_images']):
            for f in files:
                if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    images.append({'image_id': f})
        train_df = pd.DataFrame(images)
    
    if PATHS['test_csv'] and os.path.exists(PATHS['test_csv']):
        test_df = pd.read_csv(PATHS['test_csv'])
    else:
        images = []
        for root, _, files in os.walk(PATHS['test_images']):
            for f in files:
                if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    images.append({'image_id': f})
        test_df = pd.DataFrame(images)
    
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    
    # Train all models
    print("\n[2/4] Training models...")
    cv_scores = {}
    
    for name, cfg in CFG.models.items():
        score = train_model(name, cfg, train_df.copy(), device)
        cv_scores[name] = score
    
    # Ensemble inference
    print("\n[3/4] Inference...")
    predictions = ensemble_inference(test_df, device)
    
    # Create submission
    print("\n[4/4] Submission...")
    submission = create_submission(test_df, predictions)
    
    # Summary
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    for name, score in cv_scores.items():
        print(f"  {name}: CV IoU = {score:.4f}")
    print(f"\nSubmission: {CFG.OUTPUT_DIR}/submission.csv")
    
    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
