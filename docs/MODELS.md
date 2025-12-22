# 🏗️ Model Architectures

Detailed documentation of all supported model architectures.

## U-Net

The classic encoder-decoder architecture for semantic segmentation.

```
Input Image
    │
    ▼
┌─────────────┐
│   Encoder   │  (EfficientNet/ResNet)
│  ─────────  │
│  ↓ ↓ ↓ ↓ ↓  │  (Feature extraction)
└─────────────┘
    │
    ▼
┌─────────────┐
│  Bottleneck │  (Deepest features)
└─────────────┘
    │
    ▼
┌─────────────┐
│   Decoder   │  (Upsampling + Skip connections)
│  ─────────  │
│  ↑ ↑ ↑ ↑ ↑  │
└─────────────┘
    │
    ▼
Output Mask
```

### Supported Encoders

| Encoder         | Parameters | ImageNet Acc | Speed  |
| --------------- | ---------- | ------------ | ------ |
| efficientnet-b0 | 5.3M       | 77.1%        | ⚡⚡⚡ |
| efficientnet-b2 | 9.2M       | 80.1%        | ⚡⚡   |
| efficientnet-b3 | 12M        | 81.6%        | ⚡     |
| efficientnet-b4 | 19M        | 82.9%        | 🐢     |
| resnet50        | 25M        | 76.1%        | ⚡⚡   |
| resnet101       | 44M        | 77.4%        | 🐢     |

---

## U-Net++

Nested U-Net with dense skip connections for better feature aggregation.

```
Encoder    L1     L2     L3     L4    Decoder
  │         │      │      │      │       │
  └────────►├─────►├─────►├─────►├──────►│
            │      │      │      │       │
            └─────►├─────►├─────►├──────►│
                   │      │      │       │
                   └─────►├─────►├──────►│
                          │      │       │
                          └─────►├──────►│
                                 │       │
                                 └──────►│
```

### When to Use

- Higher accuracy requirements
- More complex forgery patterns
- When training time is not a constraint

---

## DeepLabV3+

Uses Atrous Spatial Pyramid Pooling (ASPP) for multi-scale feature extraction.

```
Input
  │
  ▼
┌─────────────────────────────────────┐
│              Encoder                 │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│                ASPP                  │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐       │
│  │r=6 │ │r=12│ │r=18│ │Pool│       │
│  └────┘ └────┘ └────┘ └────┘       │
│         │ Concat │                  │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│             Decoder                  │
└─────────────────────────────────────┘
  │
  ▼
Output Mask
```

### When to Use

- Multi-scale forgery detection
- Varied object sizes
- Dense prediction tasks

---

## FPN (Feature Pyramid Network)

Multi-scale feature fusion for objects of various sizes.

### When to Use

- Forgeries of varying sizes
- Quick inference needed
- Memory-constrained environments

---

## Ensemble

Combines multiple models for maximum accuracy.

```python
Final = 0.40 × UNet_EffB4 + 0.35 × UNet++_ResNet101 + 0.25 × DeepLabV3+_EffB3
```

### Recommended Ensemble

| Model              | Weight | Purpose           |
| ------------------ | ------ | ----------------- |
| UNet + EffB4       | 40%    | Primary predictor |
| UNet++ + ResNet101 | 35%    | Complex patterns  |
| DeepLabV3+ + EffB3 | 25%    | Multi-scale       |
