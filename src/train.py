"""
Training pipeline for Scientific Image Forgery Detection.

Includes:
1. Training and validation loops
2. Mixed precision training
3. Learning rate scheduling
4. Early stopping
5. Model checkpointing
6. Metrics tracking (F1, IoU, Precision, Recall)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score


class Trainer:
    """
    Trainer class for managing the complete training pipeline.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device='cuda',
        num_epochs=100,
        early_stopping_patience=15,
        checkpoint_dir='./models',
        mixed_precision=True,
        model_ema=True,
        ema_decay=0.9999
    ):
        """
        Args:
            model (nn.Module): PyTorch model
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            criterion (nn.Module): Loss function
            optimizer (Optimizer): Optimizer
            scheduler (Scheduler, optional): Learning rate scheduler
            device (str): Device to train on ('cuda' or 'cpu')
            num_epochs (int): Number of training epochs
            early_stopping_patience (int): Patience for early stopping
            checkpoint_dir (str): Directory to save model checkpoints
            mixed_precision (bool): Use mixed precision training
            model_ema (bool): Use exponential moving average of model weights
            ema_decay (float): Decay rate for EMA
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.mixed_precision = mixed_precision
        self.model_ema = model_ema
        self.ema_decay = ema_decay

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # EMA model
        if model_ema:
            self.ema_model = self._create_ema_model()
        else:
            self.ema_model = None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_iou': [],
            'val_precision': [],
            'val_recall': [],
            'learning_rates': []
        }

        # Best metrics for checkpointing
        self.best_val_iou = 0
        self.best_val_f1 = 0
        self.patience_counter = 0

    def _create_ema_model(self):
        """Create EMA model (copy of main model)."""
        ema_model = type(self.model)(
            **{k: v for k, v in self.model.__dict__.items() if not k.startswith('_')}
        )
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.to(self.device)
        ema_model.eval()
        return ema_model

    def _update_ema_model(self):
        """Update EMA model weights."""
        if self.ema_model is None:
            return

        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def train_epoch(self, epoch):
        """
        Train for one epoch.

        Args:
            epoch (int): Current epoch number

        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device).unsqueeze(1)  # [B, 1, H, W]

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.mixed_precision:
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

            # Update EMA model
            if self.model_ema:
                self._update_ema_model()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, epoch, use_ema=False):
        """
        Validate the model.

        Args:
            epoch (int): Current epoch number
            use_ema (bool): Use EMA model for validation

        Returns:
            dict: Validation metrics
        """
        model = self.ema_model if (use_ema and self.ema_model) else self.model
        model.eval()

        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device).unsqueeze(1)

                # Forward pass
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = model(images)
                    loss = self.criterion(outputs, masks)

                total_loss += loss.item()

                # Convert to binary predictions
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.append(preds.cpu().numpy())
                all_targets.append(masks.cpu().numpy())

        # Concatenate all predictions and targets
        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        iou = jaccard_score(all_targets, all_preds, zero_division=0)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)

        metrics = {
            'loss': avg_loss,
            'f1': f1,
            'iou': iou,
            'precision': precision,
            'recall': recall
        }

        return metrics

    def save_checkpoint(self, epoch, metrics, filename):
        """
        Save model checkpoint.

        Args:
            epoch (int): Current epoch
            metrics (dict): Validation metrics
            filename (str): Checkpoint filename
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }

        if self.ema_model:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath):
        """
        Load model checkpoint.

        Args:
            filepath (str): Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'ema_model_state_dict' in checkpoint and self.ema_model:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        print(f"Checkpoint loaded: {filepath}")
        return checkpoint.get('epoch', 0)

    def train(self):
        """
        Main training loop.

        Returns:
            dict: Training history
        """
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.mixed_precision}")
        print(f"Model EMA: {self.model_ema}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("-" * 80)

        for epoch in range(self.num_epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch, use_ema=self.model_ema)

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['iou'])
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['learning_rates'].append(current_lr)

            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{self.num_epochs} - {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | "
                  f"IoU: {val_metrics['iou']:.4f}")
            print(f"Precision: {val_metrics['precision']:.4f} | "
                  f"Recall: {val_metrics['recall']:.4f}")
            print(f"LR: {current_lr:.6f}")

            # Save best model based on IoU
            if val_metrics['iou'] > self.best_val_iou:
                self.best_val_iou = val_metrics['iou']
                self.save_checkpoint(epoch, val_metrics, 'best_iou.pth')
                self.patience_counter = 0
                print(f"✓ New best IoU: {self.best_val_iou:.4f}")
            else:
                self.patience_counter += 1

            # Save best model based on F1
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.save_checkpoint(epoch, val_metrics, 'best_f1.pth')
                print(f"✓ New best F1: {self.best_val_f1:.4f}")

            # Save last checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, f'checkpoint_epoch_{epoch+1}.pth')

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best IoU: {self.best_val_iou:.4f}")
                print(f"Best F1: {self.best_val_f1:.4f}")
                break

            print("-" * 80)

        print("\nTraining completed!")
        print(f"Best IoU: {self.best_val_iou:.4f}")
        print(f"Best F1: {self.best_val_f1:.4f}")

        return self.history


def train_model(
    model,
    train_dataset,
    val_dataset,
    criterion,
    batch_size=16,
    num_epochs=100,
    learning_rate=1e-4,
    weight_decay=1e-4,
    device='cuda',
    checkpoint_dir='./models',
    **kwargs
):
    """
    Convenience function to train a model.

    Args:
        model (nn.Module): PyTorch model
        train_dataset (Dataset): Training dataset
        val_dataset (Dataset): Validation dataset
        criterion (nn.Module): Loss function
        batch_size (int): Batch size
        num_epochs (int): Number of epochs
        learning_rate (float): Initial learning rate
        weight_decay (float): Weight decay for optimizer
        device (str): Device to train on
        checkpoint_dir (str): Directory to save checkpoints
        **kwargs: Additional arguments for Trainer

    Returns:
        tuple: (trained_model, history)
    """
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
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
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
        **kwargs
    )

    # Train
    history = trainer.train()

    return model, history


if __name__ == "__main__":
    print("Training module loaded successfully!")
