# Scientific Image Forgery Detection

Complete solution for the **Recod.ai & Loyola University Chicago - Scientific Image Forgery Detection** Kaggle competition. This project implements state-of-the-art deep learning models for detecting and segmenting copy-move forgeries in biomedical research images.

## Competition Overview

- **Organizers**: Recod.ai & Loyola University Chicago
- **Prize Pool**: $55,000
- **Deadline**: January 8, 2026
- **Task**: Detect and segment copy-move forgeries in scientific images
- **Dataset**: ~39,423 biomedical images (microscopy, Western blots, gel electrophoresis)
- **Evaluation**: F1-score and IoU (mIoU)

## Project Structure

```
RecorAI/
├── data/
│   ├── train/
│   │   ├── images/          # Training images
│   │   └── masks/           # Training masks (ground truth)
│   └── test/
│       └── images/          # Test images
├── src/
│   ├── dataset.py           # Dataset classes and augmentations
│   ├── models.py            # Model architectures (U-Net, Hybrid, Ensemble)
│   ├── losses.py            # Loss functions (BCE, Dice, IoU, Focal, etc.)
│   ├── train.py             # Training pipeline with mixed precision
│   ├── inference.py         # Inference and post-processing
│   └── utils.py             # Utility functions
├── notebooks/
│   └── kaggle_notebook.ipynb    # Kaggle-ready notebook
├── models/                  # Saved model checkpoints
├── outputs/                 # Predictions and submissions
├── main.py                  # Main entry point
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Features

### Model Architectures
- **U-Net**: Multiple encoder options (EfficientNet-B2/B3/B4, ResNet50/101, etc.)
- **U-Net++**: Enhanced feature aggregation
- **FPN**: Feature Pyramid Networks
- **DeepLabV3+**: ASPP for multi-scale features
- **Hybrid CNN-Transformer**: Combining local and global features
- **Attention U-Net**: CBAM attention mechanisms
- **Ensemble Models**: Combining multiple architectures

### Loss Functions
- Binary Cross-Entropy (BCE)
- Dice Loss
- IoU (Jaccard) Loss
- Focal Loss (for class imbalance)
- Tversky Loss (precision-recall trade-off)
- Combined losses (BCE+Dice, BCE+Dice+IoU, Focal+Dice)
- Boundary Loss
- Structure Loss

### Training Features
- Mixed Precision Training (FP16) for 2-3x speedup
- Exponential Moving Average (EMA) of model weights
- Cosine Annealing LR scheduling
- Early stopping with patience
- Model checkpointing (best IoU, best F1)
- Comprehensive metrics tracking (F1, IoU, Precision, Recall)

### Data Augmentation
- **Geometric**: Flips, rotations, shifts, scaling
- **Image Quality**: Gaussian noise, blur, motion blur, JPEG compression
- **Color**: Brightness, contrast, hue, saturation adjustments
- **Forgery-specific**: Simulating post-processing artifacts

### Inference & Post-processing
- Batch prediction
- Test-Time Augmentation (TTA)
- Morphological operations (closing, opening)
- Small component removal
- RLE encoding for submissions

## Installation

### Local Environment

```bash
# Clone repository
git clone <repository-url>
cd RecorAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Kaggle Environment

All dependencies are pre-installed in Kaggle notebooks. Simply upload the notebook from `notebooks/` directory.

## Quick Start

### 1. Prepare Data

Place your data in the following structure:

```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── ...
│   └── masks/
│       ├── image1.png
│       └── ...
└── test/
    └── images/
        ├── test1.jpg
        └── ...
```

### 2. Train a Model

Basic training with default settings:

```bash
python main.py --mode train
```

Advanced training with custom settings:

```bash
python main.py --mode train \
    --model unet-efficientnet-b2 \
    --batch-size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --img-size 384 \
    --loss combined \
    --mixed-precision \
    --model-ema
```

### 3. Run Inference

```bash
python main.py --mode inference \
    --checkpoint ./models/best_iou.pth \
    --test-img-dir ./data/test/images \
    --post-process \
    --use-tta
```

### 4. Train and Inference Together

```bash
python main.py --mode both
```

## Command Line Arguments

### General
- `--mode`: Mode (train/inference/both)
- `--seed`: Random seed for reproducibility (default: 42)

### Data
- `--train-img-dir`: Training images directory
- `--train-mask-dir`: Training masks directory
- `--test-img-dir`: Test images directory
- `--val-split`: Validation split ratio (default: 0.2)

### Model
- `--model`: Model architecture (default: unet-efficientnet-b2)
  - Options: `unet-efficientnet-b2`, `unet-resnet50`, `hybrid-efficientnet-b2`, etc.
- `--pretrained`: Use ImageNet pretrained weights
- `--checkpoint`: Path to checkpoint for inference/resuming

### Training
- `--batch-size`: Batch size (default: 16)
- `--epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay (default: 1e-4)
- `--img-size`: Image size (default: 384)
- `--loss`: Loss function (default: combined)
  - Options: `bce`, `dice`, `iou`, `focal`, `combined`, `bce_dice_iou`, `focal_dice`
- `--mixed-precision`: Enable mixed precision training
- `--model-ema`: Enable model EMA
- `--early-stopping`: Early stopping patience (default: 15)

### Inference
- `--use-tta`: Use test-time augmentation
- `--post-process`: Apply post-processing
- `--min-area`: Minimum area for post-processing (default: 100)

### Output
- `--output-dir`: Output directory (default: ./outputs)
- `--checkpoint-dir`: Checkpoint directory (default: ./models)

## Model Zoo

### Available Architectures

| Model | Parameters | Speed | Description |
|-------|-----------|-------|-------------|
| `unet-efficientnet-b2` | ~9M | Fast | Recommended baseline |
| `unet-efficientnet-b3` | ~12M | Medium | Better accuracy |
| `unet-efficientnet-b4` | ~19M | Slow | High accuracy |
| `unet-resnet50` | ~35M | Medium | Strong baseline |
| `unetplusplus-efficientnet-b2` | ~10M | Medium | Better feature aggregation |
| `fpn-efficientnet-b2` | ~11M | Fast | Multi-scale features |
| `deeplabv3-efficientnet-b2` | ~13M | Medium | ASPP for multi-scale |
| `hybrid-efficientnet-b2` | ~10M | Medium | CNN + Attention |

### Ensemble Strategy

For best results, ensemble multiple models:

```python
from src.models import create_ensemble

model_names = [
    'unet-efficientnet-b2',
    'unet-resnet50',
    'fpn-efficientnet-b2'
]

ensemble = create_ensemble(model_names, weights=[0.4, 0.3, 0.3])
```

## Training Strategies

### 1. Quick Baseline (1-2 hours)

```bash
python main.py --mode train \
    --model unet-efficientnet-b2 \
    --epochs 50 \
    --batch-size 32 \
    --img-size 384
```

### 2. High Accuracy (4-6 hours)

```bash
python main.py --mode train \
    --model unet-efficientnet-b3 \
    --epochs 100 \
    --batch-size 16 \
    --img-size 512 \
    --loss bce_dice_iou \
    --mixed-precision \
    --model-ema
```

### 3. Competition Winning Strategy

1. Train multiple models:
   - EfficientNet-B2/B3 U-Net
   - ResNet50 U-Net
   - FPN with EfficientNet-B2

2. Use cross-validation (5-fold)

3. Ensemble predictions with weighted voting

4. Apply TTA during inference

5. Careful post-processing tuning

## Expected Performance

### Baseline (EfficientNet-B2 U-Net)
- **F1 Score**: 0.85-0.90
- **IoU**: 0.75-0.80
- **Training Time**: 2-3 hours (single GPU)

### Advanced (Ensemble + TTA)
- **F1 Score**: 0.90-0.95
- **IoU**: 0.80-0.85
- **Training Time**: 8-12 hours (multiple models)

### State-of-the-Art (Competition Target)
- **F1 Score**: 0.95+
- **IoU**: 0.85+
- **Strategy**: Multiple architectures, cross-validation, heavy augmentation

## Tips for Kaggle Competition

1. **Start with baseline**: EfficientNet-B2 U-Net is fast and effective
2. **Use heavy augmentation**: Biomedical images are unique, augment heavily
3. **Monitor both F1 and IoU**: Don't optimize for just one metric
4. **Post-processing matters**: Morphological operations significantly improve results
5. **Ensemble is key**: Top solutions always use ensembles
6. **Cross-validation**: Ensure your model generalizes well
7. **TTA helps**: 1-2% improvement with minimal effort
8. **GPU management**: Use mixed precision to fit larger batches
9. **Experiment with losses**: Combined losses often work best
10. **Domain knowledge**: Understand biomedical image characteristics

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch-size` (try 8 or 4)
- Reduce `--img-size` (try 256 or 320)
- Enable `--mixed-precision`
- Use smaller model (efficientnet-b0 or resnet34)

### Poor Convergence
- Increase `--epochs`
- Try different `--loss` function
- Adjust `--lr` (try 5e-5 or 2e-4)
- Check data quality and augmentations

### Low F1/IoU Scores
- Increase model capacity (try efficientnet-b3/b4)
- Add more augmentation
- Enable `--post-process`
- Try ensemble of multiple models
- Tune post-processing `--min-area`

## Citation & References

### Competition
```
Recod.ai & Loyola University Chicago - Scientific Image Forgery Detection
Kaggle Competition, 2025
```

### Key Papers
1. "Benchmarking Scientific Image Forgery Detectors" (Science and Engineering Ethics, 2022)
2. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
3. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (ICML 2019)

### Dataset
- RSIIL (Recod.ai Scientific Image Integrity Library)
- ~39,423 scientific images with synthetic and real forgeries

## Contributing

This is a competition project. After the competition ends, contributions are welcome!

## License

MIT License (after competition conclusion)

## Contact

For questions and issues, please open an issue on the repository.

---

**Good luck with the competition!** 🚀
