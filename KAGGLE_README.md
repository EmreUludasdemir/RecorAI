# 🚀 Kaggle'da Çalıştırma Rehberi

Scientific Image Forgery Detection için hazır Kaggle notebook kodu.

## ✅ Hızlı Başlangıç (2 Cell Yöntemi)

### Adım 1: Kaggle Notebook Oluştur

1. https://www.kaggle.com/code adresine git
2. **Create → New Notebook**
3. **Settings → Accelerator → GPU P100** seçimi YAP! (ÇOK ÖNEMLİ!)
4. **Add Data** → Competition dataset'ini ekle

### Adım 2: Cell 1 - Paket Kurulumu

Yeni bir code cell oluştur ve şunu yapıştır:

```python
# CELL 1: Paket Kurulumu
import subprocess, sys

print("📦 Paketler yükleniyor...")

packages = [
    "albumentations==1.3.1",
    "opencv-python-headless==4.8.0.76",
    "timm",
    "segmentation-models-pytorch"
]

for pkg in packages:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

print("\n✅ Kurulum tamamlandı!")
print("\n" + "="*70)
print("⚠️  ŞİMDİ KERNEL'I RESTART EDİN!")
print("="*70)
print("\nSession → Restart Session")
print("\nRestart sonrası Cell 2'yi çalıştırın!\n")
```

**Çalıştır** → Bekle (~2 dk) → **Session → Restart Session**

### Adım 3: Cell 2 - Eğitim Kodu

Kernel restart sonrası, yeni bir cell oluştur:

```python
# CELL 2: Eğitim Kodu
import os, random, warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import timm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, jaccard_score

print("✅ Import başarılı!\n")

# CONFIG
class CFG:
    BASE_PATH = '/kaggle/input/recodai-luc-scientific-image-forgery-detection'
    TRAIN_IMG = f'{BASE_PATH}/train_images'
    TRAIN_MASK = f'{BASE_PATH}/train_masks'
    TEST_IMG = f'{BASE_PATH}/test_images'
    OUT = '/kaggle/working/outputs'
    CKPT = '/kaggle/working/models'

    ENCODER = 'efficientnet-b2'
    BATCH = 16
    EPOCHS = 50  # Test: 5, Gerçek: 50-100
    SIZE = 384
    VAL_SPLIT = 0.2
    DEVICE = 'cuda'
    SEED = 42

# GPU Kontrolü
if not torch.cuda.is_available():
    print("❌ GPU YOK!")
    print("Settings → Accelerator → GPU P100 seçin!")
    raise RuntimeError("GPU gerekli!")

print(f"✅ GPU: {torch.cuda.get_device_name(0)}\n")

# Seed
random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
torch.cuda.manual_seed_all(CFG.SEED)

os.makedirs(CFG.OUT, exist_ok=True)
os.makedirs(CFG.CKPT, exist_ok=True)

# DATASET
class ForgeryDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, tfm=None, mode='train'):
        self.img_dir, self.mask_dir, self.tfm, self.mode = img_dir, mask_dir, tfm, mode
        self.imgs = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.tif')):
                    self.imgs.append(os.path.relpath(os.path.join(root, f), img_dir))
        self.imgs = sorted(self.imgs)
        print(f"{mode.upper()}: {len(self.imgs)} images")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, self.imgs[i])), cv2.COLOR_BGR2RGB)
        if self.mode == 'test':
            if self.tfm: img = self.tfm(image=img)['image']
            return img, os.path.basename(self.imgs[i])
        mask_name = os.path.splitext(os.path.basename(self.imgs[i]))[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, 0) if os.path.exists(mask_path) else np.zeros(img.shape[:2], np.uint8)
        mask = (mask > 127).astype(np.float32)
        if self.tfm:
            aug = self.tfm(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        return img, mask

# TRANSFORMS
train_tfm = A.Compose([
    A.Resize(CFG.SIZE, CFG.SIZE), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5), A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.OneOf([A.GaussNoise(), A.GaussianBlur(), A.MotionBlur()], p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2(),
])

val_tfm = A.Compose([
    A.Resize(CFG.SIZE, CFG.SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2(),
])

# MODEL
model = smp.Unet(encoder_name=CFG.ENCODER, encoder_weights='imagenet',
                 in_channels=3, classes=1, activation=None).to(CFG.DEVICE)

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, pred, tgt):
        sig = torch.sigmoid(pred)
        dice = 1 - (2 * (sig * tgt).sum() + 1) / (sig.sum() + tgt.sum() + 1)
        return 0.5 * self.bce(pred, tgt) + 0.5 * dice

criterion = Loss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG.EPOCHS)
scaler = torch.cuda.amp.GradScaler()

# DATA
full_ds = ForgeryDataset(CFG.TRAIN_IMG, CFG.TRAIN_MASK, train_tfm, 'train')
tr_sz = int((1 - CFG.VAL_SPLIT) * len(full_ds))
tr_ds, val_ds = random_split(full_ds, [tr_sz, len(full_ds) - tr_sz])
val_ds.dataset.tfm = val_tfm

tr_ldr = DataLoader(tr_ds, CFG.BATCH, shuffle=True, num_workers=2, pin_memory=True)
val_ldr = DataLoader(val_ds, CFG.BATCH, shuffle=False, num_workers=2, pin_memory=True)

# TRAINING
print("="*70)
print("EĞİTİM BAŞLIYOR")
print("="*70 + "\n")

best_iou, patience = 0, 0

for epoch in range(CFG.EPOCHS):
    # Train
    model.train()
    tr_loss = 0
    for imgs, masks in tqdm(tr_ldr, desc=f'Epoch {epoch+1}/{CFG.EPOCHS} [Train]'):
        imgs, masks = imgs.to(CFG.DEVICE), masks.to(CFG.DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = criterion(model(imgs), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tr_loss += loss.item()
    tr_loss /= len(tr_ldr)

    # Val
    model.eval()
    val_loss, preds, tgts = 0, [], []
    with torch.no_grad():
        for imgs, masks in tqdm(val_ldr, desc=f'Epoch {epoch+1}/{CFG.EPOCHS} [Val]'):
            imgs, masks = imgs.to(CFG.DEVICE), masks.to(CFG.DEVICE).unsqueeze(1)
            out = model(imgs)
            val_loss += criterion(out, masks).item()
            p = (torch.sigmoid(out) > 0.5).cpu().numpy()
            preds.append(p)
            tgts.append(masks.cpu().numpy())

    val_loss /= len(val_ldr)
    preds = np.concatenate(preds).flatten()
    tgts = np.concatenate(tgts).flatten()
    iou = jaccard_score(tgts, preds, zero_division=0)

    scheduler.step()

    print(f"Epoch {epoch+1}/{CFG.EPOCHS}: Train={tr_loss:.4f}, Val={val_loss:.4f}, IoU={iou:.4f}")

    if iou > best_iou:
        best_iou = iou
        torch.save(model.state_dict(), f"{CFG.CKPT}/best.pth")
        print(f"  ✅ Saved (IoU: {best_iou:.4f})")
        patience = 0
    else:
        patience += 1
        if patience >= 10:
            print("Early stop"); break
    print()

# INFERENCE
print("="*70)
print("TEST TAHMİNLERİ")
print("="*70 + "\n")

model.load_state_dict(torch.load(f"{CFG.CKPT}/best.pth"))
model.eval()

test_ds = ForgeryDataset(CFG.TEST_IMG, tfm=val_tfm, mode='test')
test_ldr = DataLoader(test_ds, CFG.BATCH, shuffle=False, num_workers=2)

preds, names = [], []
with torch.no_grad():
    for imgs, ns in tqdm(test_ldr, desc='Tahmin'):
        out = torch.sigmoid(model(imgs.to(CFG.DEVICE))) > 0.5
        masks = out.squeeze().cpu().numpy()
        if masks.ndim == 2: masks = [masks]
        preds.extend(masks)
        names.extend(ns)

# RLE
def rle_encode(mask):
    p = mask.flatten()
    p = np.concatenate([[0], p, [0]])
    r = np.where(p[1:] != p[:-1])[0] + 1
    r[1::2] -= r[::2]
    return ' '.join(str(x) for x in r)

sub_data = [{'image_id': n, 'rle': rle_encode((p > 0).astype(np.uint8))} for p, n in zip(preds, names)]
sub_path = f"{CFG.OUT}/submission.csv"
pd.DataFrame(sub_data).to_csv(sub_path, index=False)

print("\n" + "="*70)
print("✅ TAMAMLANDI!")
print("="*70)
print(f"📊 En iyi IoU: {best_iou:.4f}")
print(f"💾 {sub_path}")
print("\n📥 Output sekmesinden indir!")
```

**Çalıştır** → Bekle → Submission indir!

---

## ⚙️ AYARLAR

Cell 2'de satır 24-25:

```python
BASE_PATH = '/kaggle/input/DOĞRU-DATASET-ADI'  # Dataset adını kontrol et!
EPOCHS = 5  # Hızlı test için
BATCH = 8   # Memory hatası alırsanız
```

---

## ⏱️ BEKLENEN SÜRELER

- **Paket kurulumu**: ~2-3 dakika
- **Kernel restart**: ~10 saniye
- **5 epoch (test)**: ~15-20 dakika
- **50 epoch (gerçek)**: ~1.5-2 saat
- **100 epoch (en iyi)**: ~3-4 saat

---

## 📥 SUBMISSION İNDİRME

1. **Sol menü** → **Output** sekmesi
2. `outputs/submission.csv` dosyasını bul
3. **Download** butonu
4. Competition sayfasında **Submit Predictions**

---

## ❌ SORUN GİDERME

### "GPU YOK" hatası
- **Settings → Accelerator → GPU P100** seçin!
- Notebook'u restart edin

### "CUDA out of memory"
```python
BATCH = 8  # veya 4
SIZE = 256  # veya 320
```

### "No images found"
```python
# Dataset yolunu kontrol edin:
!ls /kaggle/input/
# Doğru adı BASE_PATH'e yazın
```

### Import hatası
- Cell 1'i çalıştırdınız mı?
- Kernel restart yaptınız mı?
- GPU seçili mi?

---

## 🎯 BAŞARI KONTROL LİSTESİ

- [ ] GPU P100 seçildi
- [ ] Competition dataseti eklendi
- [ ] Cell 1 çalıştırıldı
- [ ] Kernel restart yapıldı
- [ ] Cell 2 çalıştırıldı
- [ ] Eğitim başladı
- [ ] Submission.csv oluştu
- [ ] Dosya indirildi
- [ ] Yarışmaya gönderildi

---

## 📊 PERFORMANS İPUÇLARI

- **Hızlı test**: 5 epoch, 256 size
- **Orta**: 20-30 epoch, 384 size
- **En iyi**: 100+ epoch, 512 size, TTA ekle

---

Başka soru varsa sorabilirsiniz! 🚀
