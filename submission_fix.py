"""
FIXED Submission Format for Kaggle Competition
Based on official metric code
"""

import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def rle_encode(mask, fg_val=1):
    """
    Official RLE encoder from competition metric.
    Returns JSON-encoded list format: "[start, length, start, length, ...]"

    Args:
        mask: numpy array of shape (height, width), 1 - mask, 0 - background
        fg_val: foreground value (default 1)

    Returns:
        JSON string of RLE encoding
    """
    # Transpose and flatten (column-major order, Fortran style)
    dots = np.where(mask.T.flatten() == fg_val)[0]

    run_lengths = []
    prev = -2

    for b in dots:
        if b > prev + 1:
            run_lengths.extend([int(b + 1), 0])  # Start position (1-based)
        run_lengths[-1] += 1  # Increment length
        prev = b

    # Return JSON-encoded array
    return json.dumps(run_lengths)


def is_authentic(mask, threshold=0.001):
    """
    Check if mask is authentic (no forgery).

    Args:
        mask: binary mask (height, width)
        threshold: minimum ratio of forgery pixels to consider as forged

    Returns:
        True if authentic (no significant forgery)
    """
    if mask is None:
        return True

    forgery_ratio = np.sum(mask > 0) / mask.size
    return forgery_ratio < threshold


def create_submission(predictions, filenames, output_path='submission.csv',
                     case_ids=None, threshold=0.001):
    """
    Create submission file in correct format for competition.

    Args:
        predictions: list of predicted masks (numpy arrays)
        filenames: list of image filenames
        output_path: path to save CSV
        case_ids: optional list of case IDs (if None, uses filenames without extension)
        threshold: threshold for considering image as authentic

    Format:
        case_id,annotation
        1,authentic
        2,"[123, 4]"
        3,"[123, 4];[567, 8]"
    """
    submission_data = []

    for idx, (pred, fname) in enumerate(tqdm(zip(predictions, filenames),
                                             desc='Creating submission',
                                             total=len(filenames))):
        # Get case_id (remove extension from filename)
        if case_ids is not None:
            case_id = case_ids[idx]
        else:
            case_id = fname.rsplit('.', 1)[0]  # Remove extension

        # Convert prediction to binary mask
        mask_binary = (pred > 0).astype(np.uint8)

        # Check if authentic
        if is_authentic(mask_binary, threshold=threshold):
            annotation = 'authentic'
        else:
            # Encode to RLE (JSON format)
            annotation = rle_encode(mask_binary, fg_val=1)

        submission_data.append({
            'case_id': case_id,
            'annotation': annotation
        })

    # Create DataFrame
    df = pd.DataFrame(submission_data)

    # Ensure correct column order
    df = df[['case_id', 'annotation']]

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"\n✓ Submission saved to {output_path}")
    print(f"  Total images: {len(df)}")
    print(f"  Authentic: {sum(df['annotation'] == 'authentic')}")
    print(f"  Forged: {sum(df['annotation'] != 'authentic')}")

    # Show sample
    print(f"\nSample submission:")
    print(df.head(10))

    return df


def verify_submission(submission_path='submission.csv'):
    """
    Verify submission format is correct.
    """
    df = pd.read_csv(submission_path)

    print("="*70)
    print("SUBMISSION VERIFICATION")
    print("="*70)

    # Check columns
    required_cols = ['case_id', 'annotation']
    if list(df.columns) != required_cols:
        print(f"❌ WRONG COLUMNS: {list(df.columns)}")
        print(f"   Expected: {required_cols}")
        return False
    else:
        print(f"✓ Columns correct: {list(df.columns)}")

    # Check for missing values
    if df.isnull().any().any():
        print(f"❌ Missing values found!")
        print(df.isnull().sum())
        return False
    else:
        print(f"✓ No missing values")

    # Check annotation format
    authentic_count = sum(df['annotation'] == 'authentic')
    rle_count = len(df) - authentic_count

    print(f"✓ Authentic images: {authentic_count}")
    print(f"✓ RLE encoded images: {rle_count}")

    # Validate RLE format (should be valid JSON)
    invalid_rle = []
    for idx, row in df.iterrows():
        if row['annotation'] != 'authentic':
            try:
                rle_data = json.loads(row['annotation'])
                if not isinstance(rle_data, list):
                    invalid_rle.append(idx)
            except:
                invalid_rle.append(idx)

    if invalid_rle:
        print(f"❌ Invalid RLE format in rows: {invalid_rle[:10]}")
        return False
    else:
        print(f"✓ All RLE encodings are valid JSON")

    print("="*70)
    print("✅ SUBMISSION FORMAT IS CORRECT!")
    print("="*70)

    return True


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create dummy predictions
    print("Creating example submission...\n")

    # Simulate some predictions
    predictions = [
        np.zeros((100, 100)),  # Authentic
        np.ones((100, 100)),   # Full forgery
        np.zeros((100, 100)),  # Authentic
    ]
    predictions[1][10:20, 10:20] = 1  # Small forgery region

    filenames = ['image_001.jpg', 'image_002.jpg', 'image_003.jpg']

    # Create submission
    df = create_submission(predictions, filenames, 'example_submission.csv')

    # Verify
    verify_submission('example_submission.csv')
