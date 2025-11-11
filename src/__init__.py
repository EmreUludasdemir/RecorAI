"""
Scientific Image Forgery Detection Package

A comprehensive solution for detecting and segmenting copy-move forgeries
in biomedical research images.
"""

__version__ = "1.0.0"
__author__ = "RecorAI Team"

from .dataset import (
    ScientificForgeryDataset,
    get_train_transforms,
    get_val_transforms,
    get_tta_transforms,
    SyntheticForgeryGenerator
)

from .models import (
    get_model,
    create_unet_model,
    create_unetplusplus_model,
    create_fpn_model,
    create_deeplabv3_model,
    create_ensemble,
    HybridForgeryDetector,
    EnsembleModel
)

from .losses import (
    get_loss,
    DiceLoss,
    IoULoss,
    FocalLoss,
    TverskyLoss,
    CombinedLoss,
    BCEDiceIoULoss,
    FocalDiceLoss
)

from .train import (
    Trainer,
    train_model
)

from .inference import (
    predict_single,
    predict_batch,
    predict_with_tta,
    post_process_mask,
    batch_post_process,
    run_inference,
    create_submission_csv,
    create_submission_rle
)

from .utils import (
    set_seed,
    get_device,
    compute_metrics,
    print_metrics,
    count_parameters,
    print_model_summary,
    split_dataset,
    create_kfold_splits,
    plot_training_history
)

__all__ = [
    # Dataset
    'ScientificForgeryDataset',
    'get_train_transforms',
    'get_val_transforms',
    'get_tta_transforms',
    'SyntheticForgeryGenerator',

    # Models
    'get_model',
    'create_unet_model',
    'create_unetplusplus_model',
    'create_fpn_model',
    'create_deeplabv3_model',
    'create_ensemble',
    'HybridForgeryDetector',
    'EnsembleModel',

    # Losses
    'get_loss',
    'DiceLoss',
    'IoULoss',
    'FocalLoss',
    'TverskyLoss',
    'CombinedLoss',
    'BCEDiceIoULoss',
    'FocalDiceLoss',

    # Training
    'Trainer',
    'train_model',

    # Inference
    'predict_single',
    'predict_batch',
    'predict_with_tta',
    'post_process_mask',
    'batch_post_process',
    'run_inference',
    'create_submission_csv',
    'create_submission_rle',

    # Utils
    'set_seed',
    'get_device',
    'compute_metrics',
    'print_metrics',
    'count_parameters',
    'print_model_summary',
    'split_dataset',
    'create_kfold_splits',
    'plot_training_history',
]
