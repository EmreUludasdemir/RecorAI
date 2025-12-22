# 📦 Installation Guide

Complete installation instructions for all platforms.

## 🖥️ System Requirements

### Minimum

- Python 3.8+
- 8 GB RAM
- NVIDIA GPU with 4 GB VRAM (optional but recommended)

### Recommended

- Python 3.10+
- 16 GB RAM
- NVIDIA GPU with 8+ GB VRAM (RTX 3070, A100, V100, etc.)
- CUDA 11.7+

---

## 🐍 Python Installation

### Option 1: pip (Recommended)

```bash
# Clone repository
git clone https://github.com/EmreUludasdemir/RecorAI.git
cd RecorAI

# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Conda

```bash
# Create conda environment
conda create -n recoroi python=3.10
conda activate recoroi

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

---

## 🎮 GPU Setup

### NVIDIA CUDA

1. Install [NVIDIA Driver](https://www.nvidia.com/drivers)
2. Install [CUDA Toolkit 11.7+](https://developer.nvidia.com/cuda-downloads)
3. Install [cuDNN](https://developer.nvidia.com/cudnn)

### Verify GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## ☁️ Cloud Platforms

### Kaggle (Free GPU)

No installation needed! Simply:

1. Create new notebook
2. Enable GPU accelerator
3. Copy `notebooks/kaggle_notebook.py`
4. Run!

### Google Colab

```python
# Install dependencies
!pip install -q segmentation-models-pytorch albumentations timm

# Clone repo
!git clone https://github.com/EmreUludasdemir/RecorAI.git
%cd RecorAI
```

### AWS SageMaker / Azure ML

```bash
# Use provided requirements.txt
pip install -r requirements.txt
```

---

## 🔧 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python main.py --batch-size 8

# Use mixed precision
python main.py --mixed-precision

# Use smaller model
python main.py --model unet-efficientnet-b0
```

### Import Errors

```bash
# Reinstall packages
pip install --upgrade --force-reinstall torch torchvision
pip install --upgrade segmentation-models-pytorch
```

### Albumentations Conflict

```bash
# Install specific version
pip install albumentations==1.4.0
```

---

## ✅ Verify Installation

```python
# Run this to verify everything works
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from src import models, dataset, losses

print("✓ All imports successful!")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ SMP: {smp.__version__}")
print(f"✓ CUDA: {torch.cuda.is_available()}")
```

---

## 📚 Next Steps

1. Read the [Tutorials](TUTORIALS.md)
2. Explore [Model Architectures](MODELS.md)
3. Run your first training:
   ```bash
   python main.py --mode train --epochs 10
   ```
