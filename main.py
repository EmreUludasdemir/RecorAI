"""
Main entry point for Scientific Image Forgery Detection.

This script provides a unified interface for:
1. Training models
2. Running inference on test data
3. Creating submissions
4. Cross-validation experiments
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset import ScientificForgeryDataset, get_train_transforms, get_val_transforms
from models import get_model, create_ensemble
from losses import get_loss
from train import Trainer
from inference import run_inference
from utils import set_seed, get_device, print_model_summary, plot_training_history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Scientific Image Forgery Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'inference', 'both'],
                        help='Mode: train, inference, or both')

    # Data paths
    parser.add_argument('--train-img-dir', type=str, default='./data/train/images',
                        help='Training images directory')
    parser.add_argument('--train-mask-dir', type=str, default='./data/train/masks',
                        help='Training masks directory')
    parser.add_argument('--test-img-dir', type=str, default='./data/test/images',
                        help='Test images directory')

    # Model
    parser.add_argument('--model', type=str, default='unet-efficientnet-b2',
                        help='Model architecture (e.g., unet-efficientnet-b2, hybrid-efficientnet-b2)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for inference or resuming training')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--img-size', type=int, default=384,
                        help='Image size for training')

    # Loss function
    parser.add_argument('--loss', type=str, default='combined',
                        choices=['bce', 'dice', 'iou', 'focal', 'combined', 'bce_dice_iou', 'focal_dice'],
                        help='Loss function')

    # Training options
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--model-ema', action='store_true', default=True,
                        help='Use exponential moving average of model weights')
    parser.add_argument('--early-stopping', type=int, default=15,
                        help='Early stopping patience')

    # Inference options
    parser.add_argument('--use-tta', action='store_true', default=False,
                        help='Use test-time augmentation')
    parser.add_argument('--post-process', action='store_true', default=True,
                        help='Apply post-processing to predictions')
    parser.add_argument('--min-area', type=int, default=100,
                        help='Minimum area for post-processing')

    # Output
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./models',
                        help='Checkpoint directory')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    return parser.parse_args()


def train(args, device):
    """
    Train a model.

    Args:
        args: Command line arguments
        device: Device to train on
    """
    print("\n" + "="*80)
    print("TRAINING MODE")
    print("="*80 + "\n")

    # Set seed
    set_seed(args.seed)

    # Create datasets
    print("Loading datasets...")
    train_dataset = ScientificForgeryDataset(
        image_dir=args.train_img_dir,
        mask_dir=args.train_mask_dir,
        transform=get_train_transforms(args.img_size),
        mode='train'
    )

    val_dataset = ScientificForgeryDataset(
        image_dir=args.train_img_dir,
        mask_dir=args.train_mask_dir,
        transform=get_val_transforms(args.img_size),
        mode='val'
    )

    # Split dataset
    from torch.utils.data import random_split
    train_size = int((1 - args.val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model
    print(f"\nCreating model: {args.model}")
    model = get_model(args.model, pretrained=args.pretrained, num_classes=1)
    model = model.to(device)
    print_model_summary(model)

    # Create loss function
    criterion = get_loss(args.loss)
    print(f"Loss function: {args.loss}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        checkpoint_dir=args.checkpoint_dir,
        mixed_precision=args.mixed_precision,
        model_ema=args.model_ema
    )

    # Train
    history = trainer.train()

    # Plot training history
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    os.makedirs(args.output_dir, exist_ok=True)
    plot_training_history(history, save_path=plot_path)

    print(f"\nTraining completed!")
    print(f"Best checkpoint saved in: {args.checkpoint_dir}")
    print(f"Training history plot saved to: {plot_path}")


def inference(args, device):
    """
    Run inference on test data.

    Args:
        args: Command line arguments
        device: Device to run inference on
    """
    print("\n" + "="*80)
    print("INFERENCE MODE")
    print("="*80 + "\n")

    # Check if checkpoint exists
    if args.checkpoint is None:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best_iou.pth')
        if not os.path.exists(checkpoint_path):
            print(f"Error: No checkpoint found at {checkpoint_path}")
            print("Please train a model first or specify a checkpoint with --checkpoint")
            return
        args.checkpoint = checkpoint_path

    print(f"Loading checkpoint: {args.checkpoint}")

    # Create model
    print(f"Creating model: {args.model}")
    model = get_model(args.model, pretrained=False, num_classes=1)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'ema_model_state_dict' in checkpoint:
        print("Using EMA model weights")
        model.load_state_dict(checkpoint['ema_model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    if 'metrics' in checkpoint:
        print(f"Model metrics: {checkpoint['metrics']}")

    # Create test dataset
    print("\nLoading test dataset...")
    test_dataset = ScientificForgeryDataset(
        image_dir=args.test_img_dir,
        transform=get_val_transforms(args.img_size),
        mode='test'
    )

    print(f"Test size: {len(test_dataset)}")

    # Run inference
    submission_path = run_inference(
        model=model,
        test_dataset=test_dataset,
        device=device,
        batch_size=args.batch_size,
        use_tta=args.use_tta,
        post_process=args.post_process,
        min_area=args.min_area,
        output_dir=args.output_dir
    )

    print(f"\nInference completed!")
    print(f"Submission file: {submission_path}")


def main():
    """Main function."""
    args = parse_args()

    # Print configuration
    print("\n" + "="*80)
    print("SCIENTIFIC IMAGE FORGERY DETECTION")
    print("="*80)
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Get device
    device = get_device()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Run mode
    if args.mode == 'train':
        train(args, device)
    elif args.mode == 'inference':
        inference(args, device)
    elif args.mode == 'both':
        train(args, device)
        inference(args, device)


if __name__ == "__main__":
    main()
