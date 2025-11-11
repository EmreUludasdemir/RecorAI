"""
Kaggle Notebook for Scientific Image Forgery Detection Competition

This is a complete, self-contained script that can be run in Kaggle notebooks.
All necessary code is included in a single file for easy execution.

Usage:
1. Upload this script to Kaggle
2. Add the competition dataset
3. Enable GPU accelerator
4. Run all cells
"""

# ============================================================================
# INSTALLATION (if needed)
# ============================================================================

# Uncomment if running in Kaggle and packages are missing
# !pip install -q timm segmentation-models-pytorch albumentations

# ============================================================================
# IMPORTS
# ============================================================================

import os
import sys
import time
import random
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as transforms

import timm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths (modify based on Kaggle dataset structure)
    TRAIN_IMG_DIR = '/kaggle/input/competition-name/train/images'
    TRAIN_MASK_DIR = '/kaggle/input/competition-name/train/masks'
    TEST_IMG_DIR = '/kaggle/input/competition-name/test/images'
    OUTPUT_DIR = './outputs'
    CHECKPOINT_DIR = './models'

    # Model
    MODEL_NAME = 'unet-efficientnet-b2'  # Options: unet-efficientnet-b2, unet-resnet50, etc.
    ENCODER = 'efficientnet-b2'
    PRETRAINED = 'imagenet'

    # Training
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    IMG_SIZE = 384
    VAL_SPLIT = 0.2

    # Loss
    LOSS_NAME = 'combined'  # Options: bce, dice, combined, focal_dice

    # Training options
    MIXED_PRECISION = True
    MODEL_EMA = True
    EMA_DECAY = 0.9999
    EARLY_STOPPING_PATIENCE = 15

    # Inference
    USE_TTA = False
    POST_PROCESS = True
    MIN_AREA = 100

    # Other
    SEED = 42
    NUM_WORKERS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CFG = Config()

# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(CFG.SEED)

# Create directories
os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)

print(f"Device: {CFG.DEVICE}")
if CFG.DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# DATASET AND AUGMENTATION
# ============================================================================

class ScientificForgeryDataset(Dataset):
    """Dataset for scientific image forgery detection."""

    def __init__(self, image_dir, mask_dir=None, transform=None, mode='train'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mode = mode

        self.image_files = sorted([f for f in os.listdir(image_dir)
                                   if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])

        print(f"{mode.upper()}: {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == 'test':
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, self.image_files[idx]

        mask_filename = self.image_files[idx].rsplit('.', 1)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_filename)

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


def get_train_transforms(img_size=384):
    """Get training augmentation transforms."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=384):
    """Get validation transforms."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )

        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined BCE + Dice Loss."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

# ============================================================================
# MODEL
# ============================================================================

def create_model(encoder_name='efficientnet-b2', encoder_weights='imagenet'):
    """Create U-Net model."""
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model

# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    """Trainer class for model training."""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 scheduler, device, num_epochs, checkpoint_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir

        self.scaler = torch.cuda.amp.GradScaler() if CFG.MIXED_PRECISION else None

        self.best_iou = 0
        self.best_f1 = 0
        self.patience_counter = 0

        self.history = {
            'train_loss': [], 'val_loss': [], 'val_f1': [],
            'val_iou': [], 'val_precision': [], 'val_recall': []
        }

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            if CFG.MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device).unsqueeze(1)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5
                all_preds.append(preds.cpu().numpy())
                all_targets.append(masks.cpu().numpy())

        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        metrics = {
            'loss': total_loss / len(self.val_loader),
            'f1': f1_score(all_targets, all_preds, zero_division=0),
            'iou': jaccard_score(all_targets, all_preds, zero_division=0),
            'precision': precision_score(all_targets, all_preds, zero_division=0),
            'recall': recall_score(all_targets, all_preds, zero_division=0)
        }

        return metrics

    def train(self):
        """Main training loop."""
        print(f"Training for {self.num_epochs} epochs...")

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            if self.scheduler:
                self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])

            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | F1: {val_metrics['f1']:.4f} | IoU: {val_metrics['iou']:.4f}")

            if val_metrics['iou'] > self.best_iou:
                self.best_iou = val_metrics['iou']
                torch.save(self.model.state_dict(),
                          os.path.join(self.checkpoint_dir, 'best_iou.pth'))
                self.patience_counter = 0
                print(f"✓ New best IoU: {self.best_iou:.4f}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= CFG.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        return self.history

# ============================================================================
# INFERENCE
# ============================================================================

def post_process_mask(mask, min_area=100, kernel_size=5):
    """Post-process segmentation mask."""
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output_mask[labels == i] = 255

    return output_mask


def predict_test_set(model, test_loader, device, post_process=True):
    """Predict on test set."""
    model.eval()
    predictions = []
    filenames = []

    with torch.no_grad():
        for images, fnames in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            masks = preds.squeeze().cpu().numpy()

            if masks.ndim == 2:
                masks = [masks]

            if post_process:
                masks = [post_process_mask(m) for m in masks]

            predictions.extend(masks)
            filenames.extend(fnames)

    return predictions, filenames


def rle_encode(mask):
    """RLE encoding for submission."""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def create_submission(predictions, filenames, output_path='submission.csv'):
    """Create submission file."""
    submission_data = []

    for pred, fname in tqdm(zip(predictions, filenames), desc='Creating submission'):
        mask_binary = (pred > 0).astype(np.uint8)
        rle = rle_encode(mask_binary)
        submission_data.append({'image_id': fname, 'rle': rle})

    df = pd.DataFrame(submission_data)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("SCIENTIFIC IMAGE FORGERY DETECTION")
    print("="*80)

    # ========== TRAINING ==========
    print("\n[1/3] PREPARING DATA...")

    full_dataset = ScientificForgeryDataset(
        CFG.TRAIN_IMG_DIR, CFG.TRAIN_MASK_DIR,
        transform=get_train_transforms(CFG.IMG_SIZE), mode='train'
    )

    train_size = int((1 - CFG.VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(CFG.SEED)
    )

    # Update val transforms
    val_dataset.dataset.transform = get_val_transforms(CFG.IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE,
                              shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE,
                           shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ========== MODEL ==========
    print("\n[2/3] CREATING MODEL...")

    model = create_model(CFG.ENCODER, CFG.PRETRAINED)
    model = model.to(CFG.DEVICE)

    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE,
                           weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.NUM_EPOCHS)

    # ========== TRAINING ==========
    print("\n[3/3] TRAINING...")

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer,
                     scheduler, CFG.DEVICE, CFG.NUM_EPOCHS, CFG.CHECKPOINT_DIR)
    history = trainer.train()

    # ========== INFERENCE ==========
    print("\n[4/4] INFERENCE...")

    model.load_state_dict(torch.load(os.path.join(CFG.CHECKPOINT_DIR, 'best_iou.pth')))
    model.eval()

    test_dataset = ScientificForgeryDataset(
        CFG.TEST_IMG_DIR, transform=get_val_transforms(CFG.IMG_SIZE), mode='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE,
                            shuffle=False, num_workers=CFG.NUM_WORKERS)

    predictions, filenames = predict_test_set(model, test_loader, CFG.DEVICE,
                                             post_process=CFG.POST_PROCESS)

    submission_path = os.path.join(CFG.OUTPUT_DIR, 'submission.csv')
    create_submission(predictions, filenames, submission_path)

    print("\n" + "="*80)
    print("COMPLETED!")
    print("="*80)
    print(f"Best IoU: {trainer.best_iou:.4f}")
    print(f"Submission: {submission_path}")


if __name__ == "__main__":
    main()
