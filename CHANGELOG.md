# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-22

### Added

- 🎉 Initial release
- Complete training pipeline with mixed precision support
- 3-Model ensemble (UNet+EffB4, UNet++ResNet101, DeepLabV3+EffB3)
- 5-Fold Cross-Validation
- Progressive resizing (256→384→512)
- Combined loss function (BCE + Dice + Focal)
- 6x Test-Time Augmentation (TTA)
- Post-processing with morphological operations
- Kaggle-ready notebook
- Comprehensive documentation

### Models

- U-Net with EfficientNet-B0/B2/B3/B4 encoders
- U-Net++ with ResNet101 encoder
- DeepLabV3+ with EfficientNet-B3 encoder
- FPN architecture support
- Ensemble model with weighted voting

### Features

- Mixed precision (FP16) training
- Gradient accumulation
- Cosine annealing with warm restarts
- Early stopping
- Model EMA (Exponential Moving Average)
- Memory-efficient data pipeline
- Automatic path detection for Kaggle

### Loss Functions

- Binary Cross-Entropy (BCE)
- Dice Loss
- Focal Loss
- IoU (Jaccard) Loss
- Tversky Loss
- Combined losses

---

## [Unreleased]

### Planned

- [ ] Transformer-based models (SegFormer, Swin-UNet)
- [ ] Self-training with pseudo labels
- [ ] Multi-scale inference
- [ ] Attention visualization
- [ ] ONNX export for deployment

---

## Version History

| Version | Date       | Highlights                         |
| ------- | ---------- | ---------------------------------- |
| 1.0.0   | 2024-12-22 | Initial release with full pipeline |
