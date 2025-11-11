"""
Utility functions for Scientific Image Forgery Detection.

Includes:
1. Metric computation (F1, IoU, Precision, Recall)
2. Visualization utilities
3. Model utilities (parameter counting, layer freezing)
4. Data utilities (train/val split, cross-validation)
5. Reproducibility utilities (seed setting)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score, jaccard_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


# ==================== Reproducibility ====================

def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to {seed}")


# ==================== Metrics ====================

def compute_metrics(y_true, y_pred, threshold=0.5):
    """
    Compute evaluation metrics.

    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted probabilities or labels
        threshold (float): Threshold for binary classification

    Returns:
        dict: Dictionary of metrics
    """
    # Convert to binary if probabilities
    if y_pred.dtype == np.float32 or y_pred.dtype == np.float64:
        y_pred_binary = (y_pred > threshold).astype(int)
    else:
        y_pred_binary = y_pred

    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_binary.flatten()

    # Compute metrics
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    iou = jaccard_score(y_true_flat, y_pred_flat, zero_division=0)
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat).ravel()

    metrics = {
        'f1_score': f1,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'accuracy': (tp + tn) / (tp + tn + fp + fn)
    }

    return metrics


def compute_batch_metrics(predictions, targets, threshold=0.5):
    """
    Compute metrics for a batch of predictions.

    Args:
        predictions (list): List of predicted masks
        targets (list): List of ground truth masks
        threshold (float): Threshold for binary classification

    Returns:
        dict: Dictionary of averaged metrics
    """
    all_metrics = []

    for pred, target in zip(predictions, targets):
        metrics = compute_metrics(target, pred, threshold)
        all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }

    return avg_metrics


def print_metrics(metrics, title="Metrics"):
    """
    Print metrics in a formatted way.

    Args:
        metrics (dict): Dictionary of metrics
        title (str): Title for the metrics
    """
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")
    print(f"{'=' * 50}\n")


# ==================== Model Utilities ====================

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model (nn.Module): PyTorch model

    Returns:
        dict: Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'total_MB': total_params * 4 / (1024 ** 2),  # Assuming float32
    }


def print_model_summary(model):
    """
    Print model summary.

    Args:
        model (nn.Module): PyTorch model
    """
    params = count_parameters(model)

    print(f"\n{'=' * 50}")
    print(f"{'Model Summary':^50}")
    print(f"{'=' * 50}")
    print(f"{'Total parameters:':<30} {params['total_parameters']:>15,}")
    print(f"{'Trainable parameters:':<30} {params['trainable_parameters']:>15,}")
    print(f"{'Non-trainable parameters:':<30} {params['non_trainable_parameters']:>15,}")
    print(f"{'Model size (MB):':<30} {params['total_MB']:>15.2f}")
    print(f"{'=' * 50}\n")


def freeze_layers(model, layer_names=None, freeze_all=False):
    """
    Freeze specific layers or all layers in a model.

    Args:
        model (nn.Module): PyTorch model
        layer_names (list, optional): List of layer names to freeze
        freeze_all (bool): Freeze all layers

    Returns:
        nn.Module: Model with frozen layers
    """
    if freeze_all:
        for param in model.parameters():
            param.requires_grad = False
        print("All layers frozen")
        return model

    if layer_names:
        for name, param in model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
        print(f"Frozen layers: {layer_names}")

    return model


def unfreeze_layers(model, layer_names=None, unfreeze_all=False):
    """
    Unfreeze specific layers or all layers in a model.

    Args:
        model (nn.Module): PyTorch model
        layer_names (list, optional): List of layer names to unfreeze
        unfreeze_all (bool): Unfreeze all layers

    Returns:
        nn.Module: Model with unfrozen layers
    """
    if unfreeze_all:
        for param in model.parameters():
            param.requires_grad = True
        print("All layers unfrozen")
        return model

    if layer_names:
        for name, param in model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = True
        print(f"Unfrozen layers: {layer_names}")

    return model


# ==================== Data Utilities ====================

def split_dataset(dataset, val_split=0.2, test_split=0.0, random_state=42):
    """
    Split dataset into train, validation, and test sets.

    Args:
        dataset (Dataset): PyTorch dataset
        val_split (float): Validation split ratio
        test_split (float): Test split ratio
        random_state (int): Random seed

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if test_split > 0:
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_split, random_state=random_state
        )
        val_size = val_split / (1 - test_split)
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size, random_state=random_state
        )
    else:
        train_indices, val_indices = train_test_split(
            indices, test_size=val_split, random_state=random_state
        )
        test_indices = []

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices) if test_indices else None

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}", end="")
    if test_dataset:
        print(f", Test: {len(test_dataset)}")
    else:
        print()

    return train_dataset, val_dataset, test_dataset


def create_kfold_splits(dataset, n_splits=5, random_state=42):
    """
    Create K-Fold cross-validation splits.

    Args:
        dataset (Dataset): PyTorch dataset
        n_splits (int): Number of folds
        random_state (int): Random seed

    Returns:
        list: List of (train_indices, val_indices) tuples
    """
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = list(range(len(dataset)))

    # Create dummy labels (assuming equal distribution for now)
    # In practice, you might want to use actual forgery labels
    labels = np.zeros(len(dataset))

    splits = []
    for train_idx, val_idx in kfold.split(indices, labels):
        splits.append((train_idx.tolist(), val_idx.tolist()))

    print(f"Created {n_splits}-fold cross-validation splits")
    return splits


# ==================== Visualization ====================

def plot_training_history(history, save_path=None):
    """
    Plot training history (loss, metrics).

    Args:
        history (dict): Training history dictionary
        save_path (str, optional): Path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # F1 Score
    axes[0, 1].plot(history['val_f1'], label='Val F1', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # IoU
    axes[0, 2].plot(history['val_iou'], label='Val IoU', color='orange')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('IoU')
    axes[0, 2].set_title('IoU')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Precision
    axes[1, 0].plot(history['val_precision'], label='Val Precision', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Recall
    axes[1, 1].plot(history['val_recall'], label='Val Recall', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Learning Rate
    axes[1, 2].plot(history['learning_rates'], label='Learning Rate', color='brown')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_yscale('log')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix.

    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels
        save_path (str, optional): Path to save plot
    """
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


# ==================== File Utilities ====================

def ensure_dir(directory):
    """
    Create directory if it doesn't exist.

    Args:
        directory (str): Directory path
    """
    os.makedirs(directory, exist_ok=True)


def get_device():
    """
    Get available device (CUDA or CPU).

    Returns:
        torch.device: Device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """
    Save model checkpoint.

    Args:
        model (nn.Module): Model
        optimizer (Optimizer): Optimizer
        scheduler (Scheduler): Learning rate scheduler
        epoch (int): Current epoch
        metrics (dict): Metrics dictionary
        filepath (str): Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, scheduler, filepath, device='cuda'):
    """
    Load model checkpoint.

    Args:
        model (nn.Module): Model
        optimizer (Optimizer): Optimizer
        scheduler (Scheduler): Learning rate scheduler
        filepath (str): Path to checkpoint file
        device (str): Device to load model on

    Returns:
        int: Epoch number
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})

    print(f"Checkpoint loaded: {filepath}")
    print(f"Epoch: {epoch}")
    if metrics:
        print(f"Metrics: {metrics}")

    return epoch


if __name__ == "__main__":
    print("Utils module loaded successfully!")

    # Test seed setting
    set_seed(42)

    # Test device detection
    device = get_device()

    print("\nAll utility functions loaded successfully!")
