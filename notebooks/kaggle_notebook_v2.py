"""
Kaggle Notebook - Scientific Image Forgery Detection
WORKING VERSION - Tested on Kaggle with kernel restart

USAGE:
1. Create TWO cells in Kaggle:
   - Cell 1: Package installation (run once, then restart kernel)
   - Cell 2: Training code (run after restart)
"""

# ============================================================================
# CELL 1: PACKAGE INSTALLATION (Run this first, then restart kernel)
# ============================================================================

"""
import subprocess
import sys

print("Installing compatible packages...")

# Install in one go with constraints
packages = [
    "albumentations==1.3.1",
    "opencv-python-headless==4.8.0.76",
    "timm",
    "segmentation-models-pytorch",
]

for pkg in packages:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

print("\n✓ Installation complete!")
print("\n⚠️  IMPORTANT: Click 'Restart & Run All' or restart kernel now!\n")
"""

# ============================================================================
# CELL 2: TRAINING CODE (Run after kernel restart)
# ============================================================================

import os
import random
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import timm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import f1_score, jaccard_score

print("✓ All imports successful!\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # ⚠️ UPDATE THIS PATH
    BASE_PATH = '/kaggle/input/recodai-luc-scientific-image-forgery-detection'

    TRAIN_IMG_DIR = f'{BASE_PATH}/train_images'
    TRAIN_MASK_DIR = f'{BASE_PATH}/train_masks'
    TEST_IMG_DIR = f'{BASE_PATH}/test_images'
    OUTPUT_DIR = '/kaggle/working/outputs'
    CHECKPOINT_DIR = '/kaggle/working/models'

    # Model
    ENCODER = 'efficientnet-b2'
    PRETRAINED = 'imagenet'

    # Training
    BATCH_SIZE = 16        # Reduce to 8 if OOM
    NUM_EPOCHS = 50        # 5 for testing, 50-100 for competition
    LEARNING_RATE = 1e-4
    IMG_SIZE = 384         # Reduce to 256 if OOM
    VAL_SPLIT = 0.2

    # Options
    MIXED_PRECISION = True
    EARLY_STOPPING = 10

    # System
    SEED = 42
    NUM_WORKERS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CFG = Config()

# ============================================================================
# SETUP
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CFG.SEED)
os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)

print("="*80)
print("SYSTEM INFO")
print("="*80)
print(f"Device: {CFG.DEVICE}")
if CFG.DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# ============================================================================
# DATASET
# ============================================================================

class ForgeryDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None, mode='train'):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mode = mode

        # Find all images including in subfolders
        self.images = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.tif', '.tiff')):
                    self.images.append(os.path.relpath(os.path.join(root, f), img_dir))

        self.images = sorted(self.images)
        print(f"{mode.upper()}: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mode == 'test':
            if self.transform:
                img = self.transform(image=img)['image']
            return img, os.path.basename(self.images[idx])

        # Training mode
        mask_name = os.path.splitext(os.path.basename(self.images[idx]))[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)

        mask = cv2.imread(mask_path, 0) if os.path.exists(mask_path) else np.zeros(img.shape[:2], np.uint8)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']

        return img, mask

# ============================================================================
# TRANSFORMS
# ============================================================================

def get_train_transforms(size):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ============================================================================
# MODEL & LOSS
# ============================================================================

def create_model():
    return smp.Unet(
        encoder_name=CFG.ENCODER,
        encoder_weights=CFG.PRETRAINED,
        in_channels=3,
        classes=1,
        activation=None,
    )

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum()
        dice_loss = 1 - (2 * intersection + 1) / (pred_sig.sum() + target.sum() + 1)

        return 0.5 * bce_loss + 0.5 * dice_loss

# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        self.model = model.to(CFG.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler() if CFG.MIXED_PRECISION else None
        self.best_iou = 0
        self.patience = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for imgs, masks in tqdm(self.train_loader, desc='Training'):
            imgs, masks = imgs.to(CFG.DEVICE), masks.to(CFG.DEVICE).unsqueeze(1)

            self.optimizer.zero_grad()

            if CFG.MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    out = self.model(imgs)
                    loss = self.criterion(out, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(imgs)
                loss = self.criterion(out, masks)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        preds, targets = [], []

        with torch.no_grad():
            for imgs, masks in tqdm(self.val_loader, desc='Validating'):
                imgs, masks = imgs.to(CFG.DEVICE), masks.to(CFG.DEVICE).unsqueeze(1)
                out = self.model(imgs)
                loss = self.criterion(out, masks)
                total_loss += loss.item()

                pred = (torch.sigmoid(out) > 0.5).cpu().numpy()
                preds.append(pred)
                targets.append(masks.cpu().numpy())

        preds = np.concatenate(preds).flatten()
        targets = np.concatenate(targets).flatten()

        return {
            'loss': total_loss / len(self.val_loader),
            'iou': jaccard_score(targets, preds, zero_division=0),
            'f1': f1_score(targets, preds, zero_division=0)
        }

    def train(self, epochs):
        print(f"\nTraining {epochs} epochs...\n")

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val = self.validate()

            if self.scheduler:
                self.scheduler.step()

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val['loss']:.4f} | IoU: {val['iou']:.4f} | F1: {val['f1']:.4f}")

            if val['iou'] > self.best_iou:
                self.best_iou = val['iou']
                torch.save(self.model.state_dict(), f"{CFG.CHECKPOINT_DIR}/best.pth")
                print(f"  ✓ Saved (IoU: {self.best_iou:.4f})")
                self.patience = 0
            else:
                self.patience += 1

            if self.patience >= CFG.EARLY_STOPPING:
                print(f"\nEarly stop at epoch {epoch+1}")
                break
            print()

# ============================================================================
# INFERENCE
# ============================================================================

def predict(model, loader):
    model.eval()
    preds, fnames = [], []

    with torch.no_grad():
        for imgs, names in tqdm(loader, desc='Predicting'):
            imgs = imgs.to(CFG.DEVICE)
            out = torch.sigmoid(model(imgs)) > 0.5
            masks = out.squeeze().cpu().numpy()

            if masks.ndim == 2:
                masks = [masks]

            preds.extend(masks)
            fnames.extend(names)

    return preds, fnames

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def create_submission(preds, fnames, path):
    data = [{'image_id': f, 'rle': rle_encode((p > 0).astype(np.uint8))}
            for p, f in zip(preds, fnames)]
    pd.DataFrame(data).to_csv(path, index=False)
    print(f"✓ Saved: {path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("TRAINING START")
    print("="*80 + "\n")

    # Data
    full_ds = ForgeryDataset(CFG.TRAIN_IMG_DIR, CFG.TRAIN_MASK_DIR,
                             get_train_transforms(CFG.IMG_SIZE), 'train')

    train_sz = int((1 - CFG.VAL_SPLIT) * len(full_ds))
    val_sz = len(full_ds) - train_sz
    train_ds, val_ds = random_split(full_ds, [train_sz, val_sz])
    val_ds.dataset.transform = get_val_transforms(CFG.IMG_SIZE)

    train_loader = DataLoader(train_ds, CFG.BATCH_SIZE, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, CFG.BATCH_SIZE, shuffle=False,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}\n")

    # Model
    model = create_model()
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG.NUM_EPOCHS)

    # Train
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler)
    trainer.train(CFG.NUM_EPOCHS)

    # Inference
    print("\n" + "="*80)
    print("INFERENCE")
    print("="*80 + "\n")

    model.load_state_dict(torch.load(f"{CFG.CHECKPOINT_DIR}/best.pth"))

    test_ds = ForgeryDataset(CFG.TEST_IMG_DIR, transform=get_val_transforms(CFG.IMG_SIZE), mode='test')
    test_loader = DataLoader(test_ds, CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

    preds, fnames = predict(model, test_loader)

    sub_path = f"{CFG.OUTPUT_DIR}/submission.csv"
    create_submission(preds, fnames, sub_path)

    print("\n" + "="*80)
    print("COMPLETED!")
    print("="*80)
    print(f"✓ Best IoU: {trainer.best_iou:.4f}")
    print(f"✓ Submission: {sub_path}")

if __name__ == "__main__":
    main()
