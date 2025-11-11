"""
Inference and post-processing for Scientific Image Forgery Detection.

Includes:
1. Single image and batch prediction
2. Test-Time Augmentation (TTA)
3. Post-processing (morphological operations, small component removal)
4. Submission file creation
5. Visualization utilities
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


# ==================== Inference Functions ====================

def predict_single(model, image, device='cuda', threshold=0.5):
    """
    Predict forgery mask for a single image.

    Args:
        model (nn.Module): Trained model
        image (torch.Tensor): Input image tensor [1, C, H, W]
        device (str): Device to run inference on
        threshold (float): Threshold for binary mask

    Returns:
        np.ndarray: Binary mask prediction [H, W]
    """
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        pred = torch.sigmoid(output)
        mask = (pred > threshold).squeeze().cpu().numpy()

    return mask.astype(np.uint8)


def predict_batch(model, dataloader, device='cuda', threshold=0.5):
    """
    Predict forgery masks for a batch of images.

    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): Data loader for test images
        device (str): Device to run inference on
        threshold (float): Threshold for binary mask

    Returns:
        tuple: (predictions, filenames)
            predictions: List of binary masks
            filenames: List of image filenames
    """
    model.eval()
    predictions = []
    filenames = []

    with torch.no_grad():
        for images, fnames in tqdm(dataloader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            masks = (preds > threshold).squeeze().cpu().numpy()

            if masks.ndim == 2:
                masks = [masks]
            else:
                masks = list(masks)

            predictions.extend(masks)
            filenames.extend(fnames)

    return predictions, filenames


def predict_with_tta(model, image, device='cuda', threshold=0.5):
    """
    Predict with Test-Time Augmentation (TTA).

    Args:
        model (nn.Module): Trained model
        image (torch.Tensor): Input image tensor [1, C, H, W]
        device (str): Device to run inference on
        threshold (float): Threshold for binary mask

    Returns:
        np.ndarray: Binary mask prediction [H, W]
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        image = image.to(device)

        # Original
        output = model(image)
        pred = torch.sigmoid(output)
        predictions.append(pred)

        # Horizontal flip
        output_hflip = model(torch.flip(image, dims=[3]))
        pred_hflip = torch.sigmoid(output_hflip)
        pred_hflip = torch.flip(pred_hflip, dims=[3])
        predictions.append(pred_hflip)

        # Vertical flip
        output_vflip = model(torch.flip(image, dims=[2]))
        pred_vflip = torch.sigmoid(output_vflip)
        pred_vflip = torch.flip(pred_vflip, dims=[2])
        predictions.append(pred_vflip)

        # Both flips
        output_hvflip = model(torch.flip(image, dims=[2, 3]))
        pred_hvflip = torch.sigmoid(output_hvflip)
        pred_hvflip = torch.flip(pred_hvflip, dims=[2, 3])
        predictions.append(pred_hvflip)

    # Average predictions
    avg_pred = torch.mean(torch.stack(predictions), dim=0)
    mask = (avg_pred > threshold).squeeze().cpu().numpy()

    return mask.astype(np.uint8)


# ==================== Post-Processing ====================

def post_process_mask(mask, min_area=100, kernel_size=5):
    """
    Post-process segmentation mask with morphological operations.

    Args:
        mask (np.ndarray): Binary mask [H, W]
        min_area (int): Minimum area for connected components (pixels)
        kernel_size (int): Kernel size for morphological operations

    Returns:
        np.ndarray: Post-processed binary mask
    """
    # Convert to uint8 if necessary
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    # Morphological closing (fill small holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Morphological opening (remove small noise)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Remove small connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Create output mask
    output_mask = np.zeros_like(mask)

    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            output_mask[labels == i] = 255

    return output_mask


def batch_post_process(predictions, min_area=100, kernel_size=5):
    """
    Post-process a batch of predictions.

    Args:
        predictions (list): List of binary masks
        min_area (int): Minimum area for connected components
        kernel_size (int): Kernel size for morphological operations

    Returns:
        list: List of post-processed masks
    """
    processed = []
    for pred in tqdm(predictions, desc='Post-processing'):
        processed_mask = post_process_mask(pred, min_area, kernel_size)
        processed.append(processed_mask)

    return processed


# ==================== Submission Creation ====================

def create_submission_csv(predictions, filenames, output_path='submission.csv'):
    """
    Create submission CSV file.

    Args:
        predictions (list): List of binary masks
        filenames (list): List of image filenames
        output_path (str): Path to save submission file

    Note:
        This is a template. Adjust format based on competition requirements.
    """
    submission_data = []

    for pred, fname in zip(predictions, filenames):
        # Flatten mask
        mask_flat = pred.flatten()

        submission_data.append({
            'image_id': fname,
            'prediction': ' '.join(map(str, mask_flat))
        })

    df = pd.DataFrame(submission_data)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def rle_encode(mask):
    """
    Run-Length Encoding (RLE) for efficient mask storage.

    Args:
        mask (np.ndarray): Binary mask [H, W]

    Returns:
        str: RLE encoded string
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def create_submission_rle(predictions, filenames, output_path='submission.csv'):
    """
    Create submission CSV with RLE encoding.

    Args:
        predictions (list): List of binary masks
        filenames (list): List of image filenames
        output_path (str): Path to save submission file
    """
    submission_data = []

    for pred, fname in tqdm(zip(predictions, filenames), desc='Creating submission', total=len(filenames)):
        # Convert to binary (0 or 1)
        mask_binary = (pred > 0).astype(np.uint8)

        # RLE encode
        rle = rle_encode(mask_binary)

        submission_data.append({
            'image_id': fname,
            'rle': rle
        })

    df = pd.DataFrame(submission_data)
    df.to_csv(output_path, index=False)
    print(f"Submission (RLE) saved to {output_path}")


def save_prediction_masks(predictions, filenames, output_dir='./outputs/predictions'):
    """
    Save prediction masks as PNG files.

    Args:
        predictions (list): List of binary masks
        filenames (list): List of image filenames
        output_dir (str): Directory to save masks
    """
    os.makedirs(output_dir, exist_ok=True)

    for pred, fname in tqdm(zip(predictions, filenames), desc='Saving masks', total=len(filenames)):
        # Convert to uint8
        mask = (pred * 255).astype(np.uint8) if pred.max() <= 1 else pred.astype(np.uint8)

        # Save
        output_path = os.path.join(output_dir, fname.replace('.jpg', '.png'))
        cv2.imwrite(output_path, mask)

    print(f"Masks saved to {output_dir}")


# ==================== Visualization ====================

def visualize_prediction(image, mask, prediction, save_path=None):
    """
    Visualize original image, ground truth mask, and prediction.

    Args:
        image (np.ndarray): Original image [H, W, C]
        mask (np.ndarray): Ground truth mask [H, W]
        prediction (np.ndarray): Predicted mask [H, W]
        save_path (str, optional): Path to save visualization
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Ground truth mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Prediction
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    # Overlay
    overlay = image.copy()
    if prediction.max() > 1:
        prediction_binary = prediction > 127
    else:
        prediction_binary = prediction > 0.5

    overlay[prediction_binary] = [255, 0, 0]  # Red for forgery
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_batch(images, predictions, filenames, save_dir='./outputs/visualizations', num_samples=10):
    """
    Visualize a batch of predictions.

    Args:
        images (list): List of images
        predictions (list): List of predicted masks
        filenames (list): List of image filenames
        save_dir (str): Directory to save visualizations
        num_samples (int): Number of samples to visualize
    """
    os.makedirs(save_dir, exist_ok=True)

    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)

    for idx in indices:
        image = images[idx]
        prediction = predictions[idx]
        filename = filenames[idx]

        # Create overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original')
        axes[0].axis('off')

        # Prediction
        axes[1].imshow(prediction, cmap='gray')
        axes[1].set_title('Prediction')
        axes[1].axis('off')

        # Overlay
        overlay = image.copy()
        if prediction.max() > 1:
            prediction_binary = prediction > 127
        else:
            prediction_binary = prediction > 0.5

        overlay[prediction_binary] = [255, 0, 0]
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.suptitle(f'{filename}')
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{filename}_viz.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Visualizations saved to {save_dir}")


# ==================== Main Inference Pipeline ====================

def run_inference(
    model,
    test_dataset,
    device='cuda',
    batch_size=16,
    use_tta=False,
    post_process=True,
    min_area=100,
    output_dir='./outputs',
    submission_format='rle'
):
    """
    Complete inference pipeline.

    Args:
        model (nn.Module): Trained model
        test_dataset (Dataset): Test dataset
        device (str): Device to run inference on
        batch_size (int): Batch size for inference
        use_tta (bool): Use test-time augmentation
        post_process (bool): Apply post-processing
        min_area (int): Minimum area for post-processing
        output_dir (str): Directory to save outputs
        submission_format (str): 'rle' or 'csv'

    Returns:
        str: Path to submission file
    """
    print(f"Running inference on {len(test_dataset)} images...")
    print(f"Device: {device}")
    print(f"TTA: {use_tta}")
    print(f"Post-processing: {post_process}")

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Predict
    if use_tta:
        print("Warning: TTA is not implemented for batch prediction yet.")
        print("Using standard batch prediction...")

    predictions, filenames = predict_batch(model, test_loader, device)

    # Post-process
    if post_process:
        predictions = batch_post_process(predictions, min_area=min_area)

    # Save prediction masks
    os.makedirs(output_dir, exist_ok=True)
    save_prediction_masks(predictions, filenames, os.path.join(output_dir, 'masks'))

    # Create submission
    if submission_format == 'rle':
        submission_path = os.path.join(output_dir, 'submission_rle.csv')
        create_submission_rle(predictions, filenames, submission_path)
    else:
        submission_path = os.path.join(output_dir, 'submission.csv')
        create_submission_csv(predictions, filenames, submission_path)

    print(f"\nInference completed!")
    print(f"Submission file: {submission_path}")

    return submission_path


if __name__ == "__main__":
    print("Inference module loaded successfully!")
