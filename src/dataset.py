"""
Dataset classes and augmentation transforms for Scientific Image Forgery Detection.
Supports copy-move forgery detection in biomedical images.
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ScientificForgeryDataset(Dataset):
    """
    Dataset for scientific image forgery detection and segmentation.

    Args:
        image_dir (str): Path to directory containing images
        mask_dir (str, optional): Path to directory containing masks (for train/val)
        transform (albumentations.Compose, optional): Augmentation transforms
        mode (str): One of 'train', 'val', or 'test'
    """

    def __init__(self, image_dir, mask_dir=None, transform=None, mode='train'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mode = mode

        # Get all image files
        self.image_files = sorted([f for f in os.listdir(image_dir)
                                   if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])

        print(f"{mode.upper()} dataset: {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Test mode: return image and filename only
        if self.mode == 'test':
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, self.image_files[idx]

        # Train/Val mode: load mask
        mask_filename = self.image_files[idx].rsplit('.', 1)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_filename)

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Create empty mask if not found
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Convert to binary mask (0 or 1)
        mask = (mask > 127).astype(np.float32)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


def get_train_transforms(img_size=384):
    """
    Get training augmentation transforms.
    Heavy augmentation to improve model robustness.

    Args:
        img_size (int): Target image size

    Returns:
        albumentations.Compose: Composition of transforms
    """
    return A.Compose([
        A.Resize(img_size, img_size),

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.Transpose(p=0.5),

        # Random crops and zooms
        A.OneOf([
            A.RandomCrop(int(img_size * 0.9), int(img_size * 0.9)),
            A.CenterCrop(int(img_size * 0.9), int(img_size * 0.9)),
        ], p=0.3),
        A.Resize(img_size, img_size),

        # Image quality augmentations (simulating forgery post-processing)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
            A.MedianBlur(blur_limit=7),
        ], p=0.3),

        # Color/contrast adjustments
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),

        # JPEG compression (common forgery artifact)
        A.ImageCompression(
            quality_lower=70,
            quality_upper=100,
            compression_type=A.ImageCompression.ImageCompressionType.JPEG,
            p=0.3
        ),

        # Additional artifacts
        A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
        ], p=0.2),

        # Normalize to ImageNet statistics
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=384):
    """
    Get validation/test augmentation transforms.
    Only resize and normalize, no augmentation.

    Args:
        img_size (int): Target image size

    Returns:
        albumentations.Compose: Composition of transforms
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size=384):
    """
    Get Test-Time Augmentation (TTA) transforms.
    Returns list of transforms for TTA ensemble.

    Args:
        img_size (int): Target image size

    Returns:
        list: List of transform compositions
    """
    base_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    hflip_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    vflip_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    rotate90_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Rotate(limit=(90, 90), p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return [base_transform, hflip_transform, vflip_transform, rotate90_transform]


# Additional utility for creating synthetic forgeries (optional)
class SyntheticForgeryGenerator:
    """
    Generate synthetic copy-move forgeries for additional training data.
    Useful for data augmentation and domain adaptation.
    """

    def __init__(self, min_region_size=50, max_region_size=200):
        self.min_region_size = min_region_size
        self.max_region_size = max_region_size

    def create_forgery(self, image):
        """
        Create a synthetic copy-move forgery from an authentic image.

        Args:
            image (np.ndarray): Input image (H, W, C)

        Returns:
            tuple: (forged_image, mask) where mask indicates forged regions
        """
        h, w = image.shape[:2]
        forged_image = image.copy()
        mask = np.zeros((h, w), dtype=np.uint8)

        # Random region size
        region_h = np.random.randint(self.min_region_size,
                                     min(self.max_region_size, h // 3))
        region_w = np.random.randint(self.min_region_size,
                                     min(self.max_region_size, w // 3))

        # Source region (to be copied)
        src_y = np.random.randint(0, h - region_h)
        src_x = np.random.randint(0, w - region_w)
        source_region = image[src_y:src_y+region_h, src_x:src_x+region_w].copy()

        # Target region (where to paste)
        # Ensure it doesn't overlap with source
        attempts = 0
        while attempts < 10:
            tgt_y = np.random.randint(0, h - region_h)
            tgt_x = np.random.randint(0, w - region_w)

            # Check for overlap
            if abs(tgt_y - src_y) > region_h or abs(tgt_x - src_x) > region_w:
                break
            attempts += 1

        # Apply random transformations to copied region
        if np.random.rand() > 0.5:
            # Rotate slightly
            angle = np.random.uniform(-15, 15)
            center = (region_w // 2, region_h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            source_region = cv2.warpAffine(source_region, M, (region_w, region_h))

        if np.random.rand() > 0.5:
            # Flip
            source_region = cv2.flip(source_region, np.random.choice([0, 1]))

        if np.random.rand() > 0.5:
            # Adjust brightness
            factor = np.random.uniform(0.8, 1.2)
            source_region = np.clip(source_region * factor, 0, 255).astype(np.uint8)

        # Paste the region
        forged_image[tgt_y:tgt_y+region_h, tgt_x:tgt_x+region_w] = source_region
        mask[tgt_y:tgt_y+region_h, tgt_x:tgt_x+region_w] = 255

        return forged_image, mask


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")

    # Example usage
    train_transform = get_train_transforms(img_size=384)
    val_transform = get_val_transforms(img_size=384)

    print("Train transforms:", train_transform)
    print("Val transforms:", val_transform)

    print("\nDataset module loaded successfully!")
