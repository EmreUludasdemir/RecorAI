# %% [markdown]
# # Recod.ai/LUC — Scientific Image Forgery Detection (Pixel-level Segmentation)
# **Single-file Kaggle notebook** (Python) with:
# - Smart Kaggle path detection
# - Lazy + memory-efficient pipeline (FP16 AMP, grad accumulation, optional grad checkpointing)
# - 3-model ensemble (UNet+EffB4, UNet++ + ResNet101, DeepLabV3+ + EffB3)
# - 5-Fold CV, progressive resizing (256→384→512), cosine warm restarts, early stopping
# - Heavy domain augmentations + 6x TTA
# - Post-processing + RLE submission
#
# ✅ You can run as a Kaggle Notebook (copy-paste into a single notebook .ipynb, or a .py notebook).
# ⚠️ If internet is OFF and packages are missing, you must either enable internet for the run,
# or attach a dataset with wheels and install from there.

# %% [code]
import os, gc, sys, re, math, json, time, random, glob
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# Try imports; if missing, install (may require internet unless already available/cached)
def _pip_install(pkgs: List[str]):
    import subprocess
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

try:
    import cv2
except Exception as e:
    raise RuntimeError("OpenCV (cv2) is required. Please enable it in Kaggle environment.") from e

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    _pip_install(["albumentations"])
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    raise RuntimeError("PyTorch is required in Kaggle GPU runtime.") from e

try:
    import segmentation_models_pytorch as smp
except Exception:
    _pip_install(["segmentation-models-pytorch"])
    import segmentation_models_pytorch as smp

try:
    from sklearn.model_selection import KFold
except Exception:
    _pip_install(["scikit-learn"])
    from sklearn.model_selection import KFold

from tqdm.auto import tqdm


# %% [markdown]
# ## [Cell 1] Imports & Config

# %% [code]
@dataclass
class CFG:
    seed: int = 42
    num_folds: int = 5

    # Time / training budget
    max_epochs_per_stage: Tuple[int,int,int] = (8, 8, 14)  # 256, 384, 512
    early_stop_patience: int = 10
    fold_time_limit_minutes: int = 120  # ~2 hours per fold target (adjust)

    # Image sizes (progressive resizing)
    sizes: Tuple[int,int,int] = (256, 384, 512)

    # Dataloader
    num_workers: int = 2
    pin_memory: bool = True

    # Batch / memory
    batch_size: int = 8              # actual batch
    accumulation_steps: int = 2      # effective batch ~ batch_size * accumulation_steps
    grad_clip: float = 1.0

    # Optim
    lr: float = 2e-4
    weight_decay: float = 1e-4
    t_max_restart: int = 10  # warm restart period (epochs), used via WarmRestarts

    # AMP / performance
    use_amp: bool = True
    enable_grad_checkpointing: bool = True

    # Thresholding & postprocess
    default_thr: float = 0.50
    min_area: int = 100

    # Inference
    tta: bool = True
    tta_modes: Tuple[str,...] = ("original","hflip","vflip","rot90","rot180","rot270")

    # Ensemble weights
    ens_w: Tuple[float,float,float] = (0.40, 0.35, 0.25)  # model1, model2, model3

    # Debug / quick run
    debug: bool = False
    debug_n: int = 300   # samples if debug
    save_dir: str = "/kaggle/working"

CFG = CFG()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # faster
    torch.backends.cudnn.deterministic = False

set_seed(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# %% [markdown]
# ## [Cell 2] Kaggle Path Detection (dynamic)
# Finds competition root, train/test images, masks (if present), train csv, sample_submission.

# %% [code]
def list_dir(path: str, max_items: int = 50):
    items = sorted(glob.glob(os.path.join(path, "*")))
    print(f"[{path}] items={len(items)}")
    for x in items[:max_items]:
        print("  -", os.path.basename(x))
    if len(items) > max_items:
        print("  ...")

def find_comp_root() -> str:
    # User-specified expected path
    preferred = "/kaggle/input/recodai-luc-scientific-image-forgery-detection"
    if os.path.exists(preferred):
        return preferred

    # Otherwise scan /kaggle/input
    base = "/kaggle/input"
    if not os.path.exists(base):
        raise FileNotFoundError("/kaggle/input not found (are you on Kaggle?).")

    candidates = sorted(glob.glob(os.path.join(base, "*")))
    # Heuristic: folder name contains luc/forgery/recod
    key = re.compile(r"(luc|forg|recod|scientific)", re.IGNORECASE)
    scored = []
    for c in candidates:
        name = os.path.basename(c)
        score = 0
        if key.search(name): score += 5
        # presence of csv/images inside
        if glob.glob(os.path.join(c, "**", "*.csv"), recursive=True): score += 2
        if glob.glob(os.path.join(c, "**", "*.png"), recursive=True): score += 2
        if glob.glob(os.path.join(c, "**", "*.jpg"), recursive=True): score += 2
        scored.append((score, c))
    scored.sort(reverse=True, key=lambda x: x[0])
    best = scored[0][1] if scored else base
    print("Auto-selected input root:", best, "score=", scored[0][0] if scored else None)
    return best

COMP_ROOT = find_comp_root()
print("COMP_ROOT =", COMP_ROOT)
list_dir(COMP_ROOT, max_items=80)

def discover_files(root: str) -> Dict[str, str]:
    out = {}

    # CSVs
    csvs = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    csvs = sorted(csvs)

    # Prefer sample_submission
    sample = None
    for c in csvs:
        if "sample_submission" in os.path.basename(c).lower():
            sample = c
            break
    if sample is None and csvs:
        # guess: file with "submission" in name
        for c in csvs:
            if "submission" in os.path.basename(c).lower():
                sample = c
                break
    out["sample_csv"] = sample or ""

    # Try to find train csv: often train.csv or metadata/labels
    train_csv = None
    for c in csvs:
        bn = os.path.basename(c).lower()
        if bn in ("train.csv", "training.csv", "train_labels.csv", "labels.csv"):
            train_csv = c
            break
    if train_csv is None:
        # heuristic: csv with many rows and at least 2 columns
        best = None
        best_rows = -1
        for c in csvs:
            try:
                df0 = pd.read_csv(c, nrows=100)
                if df0.shape[1] >= 2:
                    # read just line count-ish cheaply: not perfect, but ok
                    rows = sum(1 for _ in open(c, "rb"))  # includes header
                    if rows > best_rows:
                        best_rows = rows
                        best = c
            except Exception:
                continue
        train_csv = best
    out["train_csv"] = train_csv or ""

    # Images folders: look for dirs containing many image files
    img_exts = ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp")
    all_imgs = []
    for ext in img_exts:
        all_imgs += glob.glob(os.path.join(root, "**", ext), recursive=True)
    all_imgs = sorted(all_imgs)

    # Group by parent folder
    from collections import Counter
    parent_counts = Counter([os.path.dirname(p) for p in all_imgs])
    common_parents = [p for p,_ in parent_counts.most_common(50)]

    # Heuristics for train/test/mask dirs
    def pick_dir(keys: List[str]) -> str:
        keys = [k.lower() for k in keys]
        for p in common_parents:
            low = os.path.basename(p).lower()
            if any(k in low for k in keys):
                return p
        # fallback: deepest folder with many images
        return common_parents[0] if common_parents else ""

    out["train_img_dir"] = pick_dir(["train_images","train","images_train","training_images","image_train"])
    out["test_img_dir"]  = pick_dir(["test_images","test","images_test","testing_images","image_test"])
    out["mask_dir"]      = pick_dir(["masks","mask","train_masks","labels","gt","groundtruth"])

    return out

paths = discover_files(COMP_ROOT)
print(json.dumps(paths, indent=2))

# Load sample_submission if present (to learn required columns)
sample_sub = None
if paths["sample_csv"] and os.path.exists(paths["sample_csv"]):
    sample_sub = pd.read_csv(paths["sample_csv"])
    print("sample_submission head:")
    print(sample_sub.head())
else:
    print("No sample_submission found by name. We'll infer submission columns later.")

# Load train csv if present
train_df = None
if paths["train_csv"] and os.path.exists(paths["train_csv"]):
    train_df = pd.read_csv(paths["train_csv"])
    print("train_csv =", paths["train_csv"])
    print("train_df shape:", train_df.shape)
    print(train_df.head())
else:
    print("No train csv found. We'll attempt to build from folders (img<->mask name matching).")

# If train_df missing, attempt folder-based pairing
def build_df_from_folders(train_img_dir: str, mask_dir: str) -> pd.DataFrame:
    img_paths = []
    for ext in ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp"):
        img_paths += glob.glob(os.path.join(train_img_dir, ext))
    img_paths = sorted(img_paths)
    if not img_paths:
        raise FileNotFoundError("No training images found in detected train_img_dir.")
    rows = []
    for ip in img_paths:
        fn = os.path.basename(ip)
        stem = os.path.splitext(fn)[0]
        # find mask with same stem
        mp = None
        for ext in (".png",".jpg",".jpeg",".tif",".tiff",".bmp"):
            cand = os.path.join(mask_dir, stem + ext)
            if os.path.exists(cand):
                mp = cand
                break
        rows.append({"image_path": ip, "mask_path": mp or ""})
    return pd.DataFrame(rows)

if train_df is None:
    if paths["train_img_dir"] and paths["mask_dir"] and os.path.exists(paths["train_img_dir"]) and os.path.exists(paths["mask_dir"]):
        train_df = build_df_from_folders(paths["train_img_dir"], paths["mask_dir"])
        print("Built train_df from folders:", train_df.shape)
        print(train_df.head())
    else:
        raise RuntimeError("Could not detect training data layout (train csv or train/mask folders).")

# Identify ID column & image path column, RLE column, etc.
def infer_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = [c.lower() for c in df.columns]
    out = {"id": None, "image": None, "mask": None, "rle": None}

    # id-like
    for c in df.columns:
        cl = c.lower()
        if cl in ("id","image_id","img_id","filename","file_name","image_name"):
            out["id"] = c; break
    if out["id"] is None:
        # fallback: first column
        out["id"] = df.columns[0]

    # image path / filename
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ("image_path","img_path","image","img","file")):
            out["image"] = c
            break

    # mask path
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ("mask_path","mask","label_path","gt_path","groundtruth")):
            out["mask"] = c
            break

    # rle
    for c in df.columns:
        cl = c.lower()
        if "rle" in cl or "encoded" in cl:
            out["rle"] = c
            break

    return out

colmap = infer_columns(train_df)
print("Inferred columns:", colmap)

# Build absolute paths if CSV contains filenames only
def make_abs_path_maybe(p: str, base_dir: str) -> str:
    if not isinstance(p, str): return ""
    if p == "": return ""
    if os.path.isabs(p) and os.path.exists(p): return p
    cand = os.path.join(base_dir, p)
    return cand if os.path.exists(cand) else p

# If image column exists and looks like filename, attach base dir
if colmap["image"] is not None and paths["train_img_dir"]:
    train_df["__img_path__"] = train_df[colmap["image"]].apply(lambda x: make_abs_path_maybe(str(x), paths["train_img_dir"]))
elif "image_path" in train_df.columns:
    train_df["__img_path__"] = train_df["image_path"]
else:
    # try to map from id to file in train_img_dir
    id_col = colmap["id"]
    if paths["train_img_dir"]:
        def id_to_path(x):
            x = str(x)
            for ext in (".png",".jpg",".jpeg",".tif",".tiff",".bmp"):
                cand = os.path.join(paths["train_img_dir"], x + ext)
                if os.path.exists(cand): return cand
            # maybe already has extension
            cand = os.path.join(paths["train_img_dir"], x)
            return cand if os.path.exists(cand) else ""
        train_df["__img_path__"] = train_df[id_col].apply(id_to_path)

# mask path
if colmap["mask"] is not None and paths["mask_dir"]:
    train_df["__mask_path__"] = train_df[colmap["mask"]].apply(lambda x: make_abs_path_maybe(str(x), paths["mask_dir"]))
elif "mask_path" in train_df.columns:
    train_df["__mask_path__"] = train_df["mask_path"]
else:
    train_df["__mask_path__"] = ""

# RLE
if colmap["rle"] is not None:
    train_df["__rle__"] = train_df[colmap["rle"]].astype(str)
else:
    train_df["__rle__"] = ""

# Optionally debug subset
if CFG.debug:
    train_df = train_df.sample(min(CFG.debug_n, len(train_df)), random_state=CFG.seed).reset_index(drop=True)

print("Prepared train_df:", train_df.shape)
print(train_df.head())


# %% [markdown]
# ## [Cell 3] Dataset Class (lazy load + supports mask files OR RLE)

# %% [code]
def read_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    # Handle grayscale
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Handle BGRA
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_mask_bin(path: str, target_shape: Optional[Tuple[int,int]] = None) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if target_shape is not None and (m.shape[0] != target_shape[0] or m.shape[1] != target_shape[1]):
        m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.uint8)
    return m

# RLE (Kaggle standard: run-length on flattened pixels, usually column-major for some comps.
# We will implement both possibilities and auto-detect by matching shape if needed.
def rle_decode(rle: str, shape: Tuple[int,int], order: str = "F") -> np.ndarray:
    if rle is None or rle == "" or rle.lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)
    s = rle.strip().split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    if order.upper() == "F":
        return img.reshape((shape[1], shape[0])).T  # Fortran-like
    else:
        return img.reshape(shape)

def rle_encode(mask: np.ndarray, order: str = "F") -> str:
    # mask: HxW binary {0,1}
    if mask is None:
        return ""
    mask = mask.astype(np.uint8)
    if order.upper() == "F":
        pixels = mask.T.flatten()
    else:
        pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return ""
    return " ".join(str(x) for x in runs)

class ScientificForgeryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, augment=None, mode: str = "train"):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment
        self.mode = mode

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["__img_path__"]
        img = read_image_rgb(img_path)
        h0, w0 = img.shape[:2]

        # Load mask (file or RLE) for train/valid
        mask = None
        if self.mode != "test":
            if isinstance(row.get("__mask_path__", ""), str) and row["__mask_path__"] and os.path.exists(row["__mask_path__"]):
                mask = read_mask_bin(row["__mask_path__"], target_shape=(h0,w0))
            else:
                rle = row.get("__rle__", "")
                # decode with best-guess ordering (F is common for Kaggle segmentation)
                mask = rle_decode(rle, (h0,w0), order="F")

        # Resize via Albumentations (RandomResizedCrop etc.). If no augment, do simple resize + normalize.
        if self.augment is not None:
            if mask is None:
                augmented = self.augment(image=img)
                img_t = augmented["image"]
                return {"image": img_t, "id": str(row[colmap["id"]]) if colmap["id"] in row else str(idx)}
            else:
                augmented = self.augment(image=img, mask=mask)
                img_t = augmented["image"]
                mask_t = augmented["mask"].float().unsqueeze(0)  # 1xHxW
                return {"image": img_t, "mask": mask_t, "id": str(row[colmap["id"]]) if colmap["id"] in row else str(idx)}

        # Fallback basic
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485,0.456,0.406], dtype=np.float32)) / np.array([0.229,0.224,0.225], dtype=np.float32)
        img = torch.from_numpy(img).permute(2,0,1)

        if mask is None:
            return {"image": img, "id": str(row[colmap["id"]]) if colmap["id"] in row else str(idx)}
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return {"image": img, "mask": mask, "id": str(row[colmap["id"]]) if colmap["id"] in row else str(idx)}


# %% [markdown]
# ## [Cell 4] Augmentations (heavy + domain-specific)

# %% [code]
def get_train_aug(img_size: int):
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.8, border_mode=cv2.BORDER_REFLECT_101),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
        ], p=0.4),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        A.CLAHE(p=0.3),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])

def get_valid_aug(img_size: int):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])


# %% [markdown]
# ## [Cell 5] Model Definitions (3 architectures)

# %% [code]
def try_create_model(arch: str, encoder: str):
    # If imagenet weights can't be downloaded (offline), fallback to None
    for weights in ["imagenet", None]:
        try:
            if arch == "unet":
                m = smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1, activation=None)
            elif arch == "unetpp":
                m = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1, activation=None)
            elif arch == "deeplabv3p":
                m = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1, activation=None)
            else:
                raise ValueError("Unknown arch")
            if weights is None:
                print(f"[Model] {arch}+{encoder}: encoder_weights=None (offline-safe)")
            else:
                print(f"[Model] {arch}+{encoder}: encoder_weights='imagenet'")
            return m
        except Exception as e:
            print(f"Failed creating {arch}+{encoder} with weights={weights}: {type(e).__name__}: {e}")
            continue
    raise RuntimeError(f"Could not create model: {arch}+{encoder}")

def enable_grad_checkpointing(model):
    # Best-effort: some timm backbones expose this
    if not CFG.enable_grad_checkpointing:
        return
    try:
        if hasattr(model, "encoder") and hasattr(model.encoder, "set_grad_checkpointing"):
            model.encoder.set_grad_checkpointing(True)
            print("Enabled encoder grad checkpointing.")
        # some models expose .set_grad_checkpointing
        if hasattr(model, "set_grad_checkpointing"):
            model.set_grad_checkpointing(True)
            print("Enabled model grad checkpointing.")
    except Exception as e:
        print("Grad checkpointing not supported / failed:", e)

def build_models():
    # Required by spec
    m1 = try_create_model("unet", "efficientnet-b4")
    m2 = try_create_model("unetpp", "resnet101")
    m3 = try_create_model("deeplabv3p", "efficientnet-b3")
    for m in (m1,m2,m3):
        enable_grad_checkpointing(m)
    return [m1, m2, m3]


# %% [markdown]
# ## [Cell 6] Loss Functions (0.4*BCE + 0.3*Dice + 0.3*Focal) with adaptive weighting

# %% [code]
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = (0,2,3)
        num = 2*(probs*targets).sum(dims)
        den = (probs+targets).sum(dims) + self.eps
        dice = 1 - (num/den)
        return dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p*targets + (1-p)*(1-targets)
        focal = (self.alpha*(1-p_t).pow(self.gamma) * bce)
        return focal.mean()

class AdaptiveComboLoss(nn.Module):
    def __init__(self, init_pos_weight=3.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([init_pos_weight], dtype=torch.float32))
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.momentum = 0.95  # moving avg
    @torch.no_grad()
    def update_pos_weight(self, targets):
        # targets: (B,1,H,W)
        # Estimate foreground ratio and update pos_weight
        fg = targets.float().mean().clamp(1e-6, 1-1e-6).item()
        new_pw = (1 - fg) / fg
        new_pw = float(np.clip(new_pw, 1.0, 30.0))
        cur = self.pos_weight.item()
        updated = self.momentum*cur + (1-self.momentum)*new_pw
        self.pos_weight[:] = updated

    def forward(self, logits, targets):
        self.update_pos_weight(targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=self.pos_weight.to(logits.device))
        dice = self.dice(logits, targets)
        focal = self.focal(logits, targets)
        return 0.4*bce + 0.3*dice + 0.3*focal


# %% [markdown]
# ## [Cell 7] Training Loop (AMP FP16 + cosine warm restarts + early stopping)

# %% [code]
@torch.no_grad()
def compute_metrics_from_logits(logits, targets, thr=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    targets = targets.float()

    tp = (preds*targets).sum().item()
    fp = (preds*(1-targets)).sum().item()
    fn = ((1-preds)*targets).sum().item()

    f1 = (2*tp) / (2*tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return f1, iou

def train_one_epoch(model, loader, optimizer, scaler, loss_fn, scheduler=None):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, leave=False)
    for step, batch in enumerate(pbar):
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=CFG.use_amp):
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss = loss / CFG.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % CFG.accumulation_steps == 0:
            if CFG.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * CFG.accumulation_steps
        pbar.set_postfix(loss=running_loss/(step+1))

    return running_loss / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, thr=0.5):
    model.eval()
    losses = []
    f1s, ious = [], []
    loss_fn_val = AdaptiveComboLoss(init_pos_weight=3.0).to(device)  # separate for eval loss display

    for batch in tqdm(loader, leave=False):
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=CFG.use_amp):
            logits = model(imgs)
            loss = loss_fn_val(logits, masks)
        f1, iou = compute_metrics_from_logits(logits, masks, thr=thr)
        losses.append(loss.item())
        f1s.append(f1)
        ious.append(iou)

    return float(np.mean(losses)), float(np.mean(f1s)), float(np.mean(ious))

def cleanup_cuda():
    torch.cuda.empty_cache()
    gc.collect()

def fit_model_for_fold(model, fold:int, train_idx, val_idx, img_size:int, stage_name:str,
                       df: pd.DataFrame, best_path: str, start_time: float):
    train_ds = ScientificForgeryDataset(df.iloc[train_idx], img_size=img_size, augment=get_train_aug(img_size), mode="train")
    val_ds   = ScientificForgeryDataset(df.iloc[val_idx],   img_size=img_size, augment=get_valid_aug(img_size), mode="valid")

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=CFG.pin_memory, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=max(1, CFG.batch_size//2), shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=CFG.pin_memory, drop_last=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    # Cosine annealing warm restarts per update step (batch-level stepping)
    steps_per_epoch = max(1, len(train_loader)//CFG.accumulation_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, CFG.t_max_restart*steps_per_epoch), T_mult=1, eta_min=1e-6
    )

    scaler = torch.cuda.amp.GradScaler(enabled=CFG.use_amp)
    loss_fn = AdaptiveComboLoss(init_pos_weight=3.0).to(device)

    best_f1 = -1.0
    best_thr = CFG.default_thr
    bad_epochs = 0

    # stage epochs
    stage_epochs = {
        "256": CFG.max_epochs_per_stage[0],
        "384": CFG.max_epochs_per_stage[1],
        "512": CFG.max_epochs_per_stage[2],
    }[stage_name]

    for epoch in range(stage_epochs):
        # Time guard per fold
        elapsed_min = (time.time() - start_time) / 60.0
        if elapsed_min > CFG.fold_time_limit_minutes:
            print(f"[Fold {fold}] Time limit reached ({elapsed_min:.1f} min). Stopping stage.")
            break

        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, scheduler=scheduler)

        # Try a few thresholds quickly (optional)
        thr_candidates = [0.35, 0.45, 0.50, 0.55, 0.60]
        best_epoch_f1, best_epoch_iou, best_epoch_thr, best_epoch_loss = -1, -1, None, None
        for thr in thr_candidates:
            va_loss, va_f1, va_iou = validate(model, val_loader, thr=thr)
            if va_f1 > best_epoch_f1:
                best_epoch_f1, best_epoch_iou, best_epoch_thr, best_epoch_loss = va_f1, va_iou, thr, va_loss

        print(f"[Fold {fold} | {stage_name}] Epoch {epoch+1}/{stage_epochs} "
              f"train_loss={tr_loss:.4f} val_loss={best_epoch_loss:.4f} val_f1={best_epoch_f1:.4f} val_iou={best_epoch_iou:.4f} thr={best_epoch_thr}")

        if best_epoch_f1 > best_f1:
            best_f1 = best_epoch_f1
            best_thr = best_epoch_thr
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "best_thr": best_thr}, best_path)
            print("  ✔ saved:", best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= CFG.early_stop_patience:
                print("  ⏹ early stopping.")
                break

        cleanup_cuda()

    # Load best for next stage
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        best_thr = float(ckpt.get("best_thr", CFG.default_thr))

    return model, best_thr


# %% [markdown]
# ## [Cell 8] Cross-Validation Pipeline (5-fold + progressive resizing)

# %% [code]
# Build folds
kf = KFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.seed)
indices = np.arange(len(train_df))

# 3 model specs as requested
MODEL_SPECS = [
    ("model1_unet_effb4",  "unet",      "efficientnet-b4"),
    ("model2_unetpp_r101", "unetpp",    "resnet101"),
    ("model3_dlv3p_effb3", "deeplabv3p","efficientnet-b3"),
]

# You can toggle training/inference:
DO_TRAIN = True

# Where to save per-fold best weights per model+fold+stage (final stage used for inference)
os.makedirs(CFG.save_dir, exist_ok=True)

fold_thresholds = {name: {} for (name,_,_) in MODEL_SPECS}

if DO_TRAIN:
    for model_name, arch, encoder in MODEL_SPECS:
        print("\n" + "="*90)
        print("Training:", model_name, f"({arch}+{encoder})")
        print("="*90)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(indices)):
            print("\n" + "-"*70)
            print(f"{model_name} | Fold {fold}/{CFG.num_folds-1}")
            print("-"*70)

            start_time = time.time()
            # Create model fresh per fold
            model = try_create_model(arch, encoder)
            enable_grad_checkpointing(model)

            best_thr = CFG.default_thr

            # Progressive resizing: 256 -> 384 -> 512
            for stage_size, stage_name in zip(CFG.sizes, ("256","384","512")):
                best_path = os.path.join(CFG.save_dir, f"best_{model_name}_fold{fold}_stage{stage_name}.pth")
                # If continuing, load previous stage best into model
                if stage_name != "256":
                    prev_candidates = sorted(glob.glob(os.path.join(CFG.save_dir, f"best_{model_name}_fold{fold}_stage*.pth")))
                    if prev_candidates:
                        ckpt = torch.load(prev_candidates[-1], map_location="cpu")
                        model.load_state_dict(ckpt["model"])
                        best_thr = float(ckpt.get("best_thr", best_thr))
                        print("Loaded previous stage:", prev_candidates[-1], "thr=", best_thr)

                model, best_thr = fit_model_for_fold(
                    model=model, fold=fold, train_idx=tr_idx, val_idx=va_idx,
                    img_size=stage_size, stage_name=stage_name, df=train_df,
                    best_path=best_path, start_time=start_time
                )

            fold_thresholds[model_name][fold] = best_thr

            # Cleanup per fold
            del model
            cleanup_cuda()

    # Save thresholds
    with open(os.path.join(CFG.save_dir, "fold_thresholds.json"), "w") as f:
        json.dump(fold_thresholds, f, indent=2)
    print("Saved fold_thresholds.json")

else:
    # Load thresholds if exist
    thr_path = os.path.join(CFG.save_dir, "fold_thresholds.json")
    if os.path.exists(thr_path):
        fold_thresholds = json.load(open(thr_path, "r"))
        print("Loaded thresholds:", thr_path)


# %% [markdown]
# ## [Cell 9] Inference + 6x TTA

# %% [code]
# Build test dataframe from folders or sample_submission
def build_test_df(paths: Dict[str,str], sample_sub: Optional[pd.DataFrame]) -> pd.DataFrame:
    # If sample submission exists, use its id column and map to files
    if sample_sub is not None and len(sample_sub) > 0:
        id_col = sample_sub.columns[0]
        df = sample_sub[[id_col]].copy()
        df.rename(columns={id_col: "__id__"}, inplace=True)
        if paths["test_img_dir"]:
            def id_to_path(x):
                x = str(x)
                # try with and without extension
                for ext in (".png",".jpg",".jpeg",".tif",".tiff",".bmp"):
                    cand = os.path.join(paths["test_img_dir"], x + ext)
                    if os.path.exists(cand): return cand
                cand = os.path.join(paths["test_img_dir"], x)
                return cand if os.path.exists(cand) else ""
            df["__img_path__"] = df["__id__"].apply(id_to_path)
        else:
            df["__img_path__"] = df["__id__"].astype(str)
        return df

    # Otherwise scan test folder
    test_dir = paths["test_img_dir"]
    if not test_dir or not os.path.exists(test_dir):
        # fallback: guess some folder in COMP_ROOT
        test_dir = paths["train_img_dir"]
        print("⚠️ test_img_dir not found; falling back to train_img_dir scanning for 'test'-like images.")
    img_paths = []
    for ext in ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp"):
        img_paths += glob.glob(os.path.join(test_dir, ext))
    img_paths = sorted(img_paths)
    df = pd.DataFrame({"__img_path__": img_paths})
    df["__id__"] = df["__img_path__"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    return df

test_df = build_test_df(paths, sample_sub)
if CFG.debug:
    test_df = test_df.head(100).copy()
print("test_df:", test_df.shape)
print(test_df.head())

# Build tta helpers
def apply_tta(img: torch.Tensor, mode: str) -> torch.Tensor:
    # img: BxCxHxW
    if mode == "original":
        return img
    if mode == "hflip":
        return torch.flip(img, dims=[3])
    if mode == "vflip":
        return torch.flip(img, dims=[2])
    if mode == "rot90":
        return torch.rot90(img, k=1, dims=[2,3])
    if mode == "rot180":
        return torch.rot90(img, k=2, dims=[2,3])
    if mode == "rot270":
        return torch.rot90(img, k=3, dims=[2,3])
    raise ValueError("Unknown TTA mode")

def undo_tta(mask: torch.Tensor, mode: str) -> torch.Tensor:
    # mask: Bx1xHxW
    if mode == "original":
        return mask
    if mode == "hflip":
        return torch.flip(mask, dims=[3])
    if mode == "vflip":
        return torch.flip(mask, dims=[2])
    if mode == "rot90":
        return torch.rot90(mask, k=3, dims=[2,3])
    if mode == "rot180":
        return torch.rot90(mask, k=2, dims=[2,3])
    if mode == "rot270":
        return torch.rot90(mask, k=1, dims=[2,3])
    raise ValueError("Unknown TTA mode")

@torch.no_grad()
def predict_model(model, loader, tta=True, tta_modes=CFG.tta_modes):
    model.eval()
    preds = []
    ids = []
    for batch in tqdm(loader, leave=False):
        imgs = batch["image"].to(device, non_blocking=True)
        batch_ids = batch["id"]

        if not tta:
            with torch.cuda.amp.autocast(enabled=CFG.use_amp):
                logits = model(imgs)
                prob = torch.sigmoid(logits).float()
        else:
            prob_acc = 0.0
            for m in tta_modes:
                x = apply_tta(imgs, m)
                with torch.cuda.amp.autocast(enabled=CFG.use_amp):
                    logits = model(x)
                    p = torch.sigmoid(logits).float()
                p = undo_tta(p, m)
                prob_acc = prob_acc + p
            prob = prob_acc / float(len(tta_modes))

        preds.append(prob.detach().cpu())
        ids.extend(list(batch_ids))
    preds = torch.cat(preds, dim=0)  # Nx1xHxW
    return ids, preds

# Test loader at final size (512)
infer_size = CFG.sizes[-1]
test_ds = ScientificForgeryDataset(
    df=test_df.assign(**{colmap["id"] if colmap["id"] in test_df.columns else "__id__": test_df["__id__"]}),
    img_size=infer_size,
    augment=get_valid_aug(infer_size),
    mode="test"
)
# Patch dataset id access: it uses colmap["id"] if in row else idx. We'll ensure 'id' column exists.
if colmap["id"] not in test_ds.df.columns:
    test_ds.df[colmap["id"]] = test_ds.df["__id__"]

test_loader = DataLoader(test_ds, batch_size=max(1, CFG.batch_size//2), shuffle=False,
                         num_workers=CFG.num_workers, pin_memory=CFG.pin_memory, drop_last=False)


# %% [markdown]
# ## [Cell 10] Post-processing (morph close + CC filtering)

# %% [code]
def post_process(prob: np.ndarray, thr: float = 0.5, min_area: int = 100) -> np.ndarray:
    # prob: HxW float [0,1]
    mask = (prob > thr).astype(np.uint8)

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Connected component filtering
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1
    return out


# %% [markdown]
# ## [Cell 11] Ensemble (3 models × 5 folds) with weights (0.40 / 0.35 / 0.25)

# %% [code]
def load_best_final_stage(model_name: str, fold: int) -> str:
    # Prefer final stage 512
    patt = os.path.join(CFG.save_dir, f"best_{model_name}_fold{fold}_stage512.pth")
    if os.path.exists(patt):
        return patt
    # fallback: any best_... for fold, take latest
    cands = sorted(glob.glob(os.path.join(CFG.save_dir, f"best_{model_name}_fold{fold}_stage*.pth")))
    return cands[-1] if cands else ""

@torch.no_grad()
def infer_all_models_ensemble():
    # returns: dict id -> prob(HxW)
    final_probs = None
    final_ids = None

    # For memory safety: accumulate in float32 on CPU per chunk
    # We'll build N x H x W arrays on CPU (can be big). If too big, you can stream-save per image.
    for (spec_i, (model_name, arch, encoder)) in enumerate(MODEL_SPECS):
        w_model = CFG.ens_w[spec_i]
        print("\nEnsembling:", model_name, "weight=", w_model)

        # average over folds
        fold_probs_sum = None

        for fold in range(CFG.num_folds):
            ckpt_path = load_best_final_stage(model_name, fold)
            if not ckpt_path:
                print(f"⚠️ Missing checkpoint for {model_name} fold {fold}, skipping.")
                continue

            model = try_create_model(arch, encoder).to(device)
            enable_grad_checkpointing(model)

            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            thr = float(ckpt.get("best_thr", CFG.default_thr))
            print(f"  Fold {fold}: ckpt={os.path.basename(ckpt_path)} thr={thr}")

            ids, probs = predict_model(model, test_loader, tta=CFG.tta, tta_modes=CFG.tta_modes)

            # Ensure consistent order
            if final_ids is None:
                final_ids = ids
            else:
                assert final_ids == ids, "ID order mismatch across models/folds."

            # probs: Nx1xHxW
            probs = probs[:,0]  # NxHxW

            if fold_probs_sum is None:
                fold_probs_sum = probs
            else:
                fold_probs_sum += probs

            del model
            cleanup_cuda()

        if fold_probs_sum is None:
            continue

        fold_probs_avg = fold_probs_sum / float(CFG.num_folds)

        if final_probs is None:
            final_probs = w_model * fold_probs_avg
        else:
            final_probs += w_model * fold_probs_avg

    return final_ids, final_probs  # ids list, probs tensor NxHxW

ids, probs = infer_all_models_ensemble()
print("Ensemble probs:", probs.shape)


# %% [markdown]
# ## [Cell 12] Submission Generation (RLE)

# %% [code]
# Determine submission columns (from sample_submission if available)
if sample_sub is not None:
    sub_id_col = sample_sub.columns[0]
    sub_rle_col = sample_sub.columns[1] if sample_sub.shape[1] > 1 else "rle"
else:
    sub_id_col, sub_rle_col = "id", "rle"

# Postprocess + RLE encode
rles = []
for i in tqdm(range(len(ids))):
    prob = probs[i].numpy()  # HxW at infer_size
    # Use a global threshold; if you want per-fold thresholding, keep a tuned thr (already saved per fold), but ensemble mixes folds.
    thr = CFG.default_thr
    mask = post_process(prob, thr=thr, min_area=CFG.min_area)
    rles.append(rle_encode(mask, order="F"))

submission = pd.DataFrame({sub_id_col: ids, sub_rle_col: rles})
sub_path = os.path.join(CFG.save_dir, "submission.csv")
submission.to_csv(sub_path, index=False)
print("Saved:", sub_path)
print(submission.head())


# %% [markdown]
# ## [Cell 13] Cleanup & Memory

# %% [code]
del probs
cleanup_cuda()
print("Done.")
