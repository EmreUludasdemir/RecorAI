# 🎓 Tutorials

This directory contains tutorials and guides for using the Scientific Image Forgery Detection toolkit.

## 📚 Available Tutorials

### 1. Quick Start Guide

Learn the basics in under 10 minutes.

```python
from src.models import create_model
from src.inference import predict

# Load model
model = create_model('unet-efficientnet-b2')
model.load_state_dict(torch.load('models/best_iou.pth'))

# Predict
mask = predict(model, 'test_image.jpg')
```

### 2. Training Your First Model

```bash
# Basic training
python main.py --mode train --epochs 50

# With all optimizations
python main.py --mode train \
    --model unet-efficientnet-b4 \
    --batch-size 16 \
    --epochs 100 \
    --mixed-precision \
    --model-ema
```

### 3. Understanding the Dataset

The dataset contains scientific images with copy-move forgeries:

| Image Type          | Description                 |
| ------------------- | --------------------------- |
| Microscopy          | Cell and tissue images      |
| Western Blots       | Protein expression analysis |
| Gel Electrophoresis | DNA/RNA separation          |

### 4. Model Selection Guide

| Model           | Best For          | GPU Memory |
| --------------- | ----------------- | ---------- |
| EfficientNet-B2 | Quick experiments | 4-6 GB     |
| EfficientNet-B4 | Best accuracy     | 8-12 GB    |
| ResNet50        | Balanced          | 6-8 GB     |

### 5. Post-Processing Techniques

```python
from src.inference import post_process_mask

# Apply morphological operations
cleaned_mask = post_process_mask(
    raw_mask,
    min_area=100,      # Remove small components
    kernel_size=5      # Morphological kernel
)
```

### 6. Ensemble Strategy

```python
from src.models import create_ensemble

# Create weighted ensemble
models = [
    ('unet-efficientnet-b4', 0.40),
    ('unetplusplus-resnet101', 0.35),
    ('deeplabv3-efficientnet-b3', 0.25)
]

ensemble = create_ensemble(models)
```

## 🔗 Additional Resources

- [Kaggle Competition Page](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection)
- [segmentation_models_pytorch Documentation](https://smp.readthedocs.io/)
- [Albumentations Augmentation Guide](https://albumentations.ai/docs/)

## 📧 Need Help?

Open an issue on GitHub or check existing discussions!
