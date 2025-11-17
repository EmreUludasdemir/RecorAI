"""
Test RLE encoding format for Kaggle submission
"""
import numpy as np
import pandas as pd

def rle_encode(mask):
    """RLE encoding for submission."""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Test 1: Empty mask (no forgery)
print("Test 1: Empty mask (all zeros)")
empty_mask = np.zeros((10, 10), dtype=np.uint8)
rle = rle_encode(empty_mask)
print(f"RLE: '{rle}'")
print(f"Expected: '' (empty string)")
print()

# Test 2: Full mask (all forgery)
print("Test 2: Full mask (all ones)")
full_mask = np.ones((10, 10), dtype=np.uint8)
rle = rle_encode(full_mask)
print(f"RLE: '{rle}'")
print(f"Expected: '1 100' (starts at position 1, length 100)")
print()

# Test 3: Simple pattern
print("Test 3: Simple pattern (first 5 pixels are 1)")
pattern_mask = np.zeros((2, 5), dtype=np.uint8)
pattern_mask[0, :] = 1  # First row all 1s
rle = rle_encode(pattern_mask)
print(f"RLE: '{rle}'")
print(f"Shape: {pattern_mask.shape}")
print(f"Pattern:\n{pattern_mask}")
print(f"Expected: '1 5' (starts at 1, length 5)")
print()

# Test 4: Two separate regions
print("Test 4: Two separate regions")
two_region_mask = np.zeros((4, 4), dtype=np.uint8)
two_region_mask[0, 0:2] = 1  # First 2 pixels
two_region_mask[2, 2:4] = 1  # Pixels at position 10-11
rle = rle_encode(two_region_mask)
print(f"Pattern:\n{two_region_mask}")
print(f"Flattened: {two_region_mask.flatten()}")
print(f"RLE: '{rle}'")
print(f"Expected: '1 2 11 2' (two regions)")
print()

# Test 5: Create sample submission
print("Test 5: Sample submission format")
submission_data = [
    {'image_id': 'image_001.jpg', 'rle': '1 100 150 50'},
    {'image_id': 'image_002.jpg', 'rle': ''},  # No forgery
    {'image_id': 'image_003.jpg', 'rle': '1 500'},
]

df = pd.DataFrame(submission_data)
print("\nSubmission DataFrame:")
print(df)
print(f"\nColumns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")

# Save to CSV
df.to_csv('/home/user/RecorAI/test_submission.csv', index=False)
print("\n✓ Sample submission saved to test_submission.csv")

# Verify CSV format
print("\nCSV Content:")
with open('/home/user/RecorAI/test_submission.csv', 'r') as f:
    content = f.read()
    print(content)
