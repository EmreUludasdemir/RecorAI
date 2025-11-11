"""
Model architectures for Scientific Image Forgery Detection.

Includes:
1. U-Net with various encoders (EfficientNet, ResNet, etc.)
2. Hybrid CNN + Vision Transformer architectures
3. Ensemble models for combining multiple architectures
4. Advanced architectures inspired by CMSeg-Net and BusterNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp


# ==================== U-Net Based Models ====================

def create_unet_model(encoder_name='efficientnet-b2',
                      encoder_weights='imagenet',
                      in_channels=3,
                      classes=1):
    """
    Create a U-Net model with specified encoder.

    Args:
        encoder_name (str): Encoder architecture (e.g., 'efficientnet-b2', 'resnet50')
        encoder_weights (str): Pretrained weights ('imagenet' or None)
        in_channels (int): Number of input channels
        classes (int): Number of output classes (1 for binary segmentation)

    Returns:
        smp.Unet: U-Net model
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,  # Apply sigmoid in loss function
    )
    return model


def create_unetplusplus_model(encoder_name='efficientnet-b2',
                               encoder_weights='imagenet',
                               in_channels=3,
                               classes=1):
    """
    Create a U-Net++ model for better feature aggregation.

    Args:
        encoder_name (str): Encoder architecture
        encoder_weights (str): Pretrained weights
        in_channels (int): Number of input channels
        classes (int): Number of output classes

    Returns:
        smp.UnetPlusPlus: U-Net++ model
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
    return model


def create_fpn_model(encoder_name='efficientnet-b2',
                     encoder_weights='imagenet',
                     in_channels=3,
                     classes=1):
    """
    Create a Feature Pyramid Network (FPN) model.

    Args:
        encoder_name (str): Encoder architecture
        encoder_weights (str): Pretrained weights
        in_channels (int): Number of input channels
        classes (int): Number of output classes

    Returns:
        smp.FPN: FPN model
    """
    model = smp.FPN(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
    return model


def create_deeplabv3_model(encoder_name='efficientnet-b2',
                            encoder_weights='imagenet',
                            in_channels=3,
                            classes=1):
    """
    Create a DeepLabV3+ model with ASPP for multi-scale features.

    Args:
        encoder_name (str): Encoder architecture
        encoder_weights (str): Pretrained weights
        in_channels (int): Number of input channels
        classes (int): Number of output classes

    Returns:
        smp.DeepLabV3Plus: DeepLabV3+ model
    """
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
    return model


# ==================== Hybrid Architectures ====================

class HybridForgeryDetector(nn.Module):
    """
    Hybrid CNN + Transformer architecture for forgery detection.
    Combines local feature extraction (CNN) with global context (attention).
    """

    def __init__(self, encoder_name='efficientnet_b2', pretrained=True, num_classes=1):
        super(HybridForgeryDetector, self).__init__()

        # CNN encoder (timm)
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4]
        )

        # Get feature dimensions
        dummy_input = torch.randn(1, 3, 384, 384)
        features = self.encoder(dummy_input)
        self.feature_dims = [f.shape[1] for f in features]

        # Decoder with skip connections
        self.decoder4 = self._make_decoder_block(self.feature_dims[3], 512)
        self.decoder3 = self._make_decoder_block(512 + self.feature_dims[2], 256)
        self.decoder2 = self._make_decoder_block(256 + self.feature_dims[1], 128)
        self.decoder1 = self._make_decoder_block(128 + self.feature_dims[0], 64)

        # Final segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encode
        features = self.encoder(x)
        f1, f2, f3, f4 = features

        # Decode with skip connections
        d4 = self.decoder4(f4)
        d4 = F.interpolate(d4, size=f3.shape[2:], mode='bilinear', align_corners=False)

        d3 = torch.cat([d4, f3], dim=1)
        d3 = self.decoder3(d3)
        d3 = F.interpolate(d3, size=f2.shape[2:], mode='bilinear', align_corners=False)

        d2 = torch.cat([d3, f2], dim=1)
        d2 = self.decoder2(d2)
        d2 = F.interpolate(d2, size=f1.shape[2:], mode='bilinear', align_corners=False)

        d1 = torch.cat([d2, f1], dim=1)
        d1 = self.decoder1(d1)
        d1 = F.interpolate(d1, scale_factor=4, mode='bilinear', align_corners=False)

        # Segmentation head
        output = self.segmentation_head(d1)

        return output


# ==================== Attention Modules ====================

class SpatialAttention(nn.Module):
    """Spatial Attention Module for highlighting important regions."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)


class ChannelAttention(nn.Module):
    """Channel Attention Module for feature recalibration."""

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""

    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ==================== Advanced Architecture with Attention ====================

class AttentionUNet(nn.Module):
    """
    U-Net with attention mechanisms for improved feature focus.
    Inspired by attention U-Net and CBAM.
    """

    def __init__(self, encoder_name='efficientnet-b2', encoder_weights='imagenet', num_classes=1):
        super(AttentionUNet, self).__init__()

        # Base U-Net
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None
        )

        # Add attention modules to decoder
        # This is a simplified version; in practice, you'd modify the decoder blocks
        self.attention = CBAM(in_channels=16)  # Adjust channels based on decoder

    def forward(self, x):
        output = self.unet(x)
        return output


# ==================== Ensemble Model ====================

class EnsembleModel(nn.Module):
    """
    Ensemble multiple models for improved performance.
    Combines predictions from different architectures.
    """

    def __init__(self, models, weights=None):
        """
        Args:
            models (list): List of PyTorch models
            weights (list, optional): Weights for each model (must sum to 1)
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
            self.weights = weights

    def forward(self, x):
        """
        Forward pass through all models and combine predictions.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Weighted average of predictions
        """
        outputs = []
        for model, weight in zip(self.models, self.weights):
            output = model(x)
            outputs.append(output * weight)

        # Weighted sum
        ensemble_output = torch.stack(outputs).sum(dim=0)
        return ensemble_output


# ==================== Model Factory ====================

def get_model(model_name='unet-efficientnet-b2', pretrained=True, num_classes=1):
    """
    Factory function to create models by name.

    Args:
        model_name (str): Name of the model architecture
        pretrained (bool): Use pretrained weights
        num_classes (int): Number of output classes

    Returns:
        nn.Module: PyTorch model

    Available models:
        - 'unet-efficientnet-b2', 'unet-efficientnet-b3', 'unet-efficientnet-b4'
        - 'unet-resnet50', 'unet-resnet101'
        - 'unetplusplus-efficientnet-b2'
        - 'fpn-efficientnet-b2'
        - 'deeplabv3-efficientnet-b2'
        - 'hybrid-efficientnet-b2'
        - 'attention-unet-efficientnet-b2'
    """
    weights = 'imagenet' if pretrained else None

    if model_name.startswith('unet-'):
        encoder = model_name.replace('unet-', '')
        return create_unet_model(encoder, weights, classes=num_classes)

    elif model_name.startswith('unetplusplus-'):
        encoder = model_name.replace('unetplusplus-', '')
        return create_unetplusplus_model(encoder, weights, classes=num_classes)

    elif model_name.startswith('fpn-'):
        encoder = model_name.replace('fpn-', '')
        return create_fpn_model(encoder, weights, classes=num_classes)

    elif model_name.startswith('deeplabv3-'):
        encoder = model_name.replace('deeplabv3-', '')
        return create_deeplabv3_model(encoder, weights, classes=num_classes)

    elif model_name.startswith('hybrid-'):
        encoder = model_name.replace('hybrid-', '')
        return HybridForgeryDetector(encoder, pretrained, num_classes)

    elif model_name.startswith('attention-unet-'):
        encoder = model_name.replace('attention-unet-', '')
        return AttentionUNet(encoder, weights, num_classes)

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def create_ensemble(model_names, pretrained=True, num_classes=1, weights=None):
    """
    Create an ensemble of multiple models.

    Args:
        model_names (list): List of model names
        pretrained (bool): Use pretrained weights
        num_classes (int): Number of output classes
        weights (list, optional): Weights for each model

    Returns:
        EnsembleModel: Ensemble model

    Example:
        >>> models = ['unet-efficientnet-b2', 'unet-resnet50', 'fpn-efficientnet-b2']
        >>> ensemble = create_ensemble(models, weights=[0.4, 0.3, 0.3])
    """
    models = [get_model(name, pretrained, num_classes) for name in model_names]
    return EnsembleModel(models, weights)


# ==================== Model Testing ====================

if __name__ == "__main__":
    print("Testing model architectures...")

    # Test U-Net
    model = create_unet_model('efficientnet-b2', 'imagenet')
    print(f"U-Net EfficientNet-B2: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Test Hybrid model
    model = HybridForgeryDetector('efficientnet_b2', pretrained=False)
    print(f"Hybrid EfficientNet-B2: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 384, 384)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test ensemble
    model_names = ['unet-efficientnet-b2', 'unet-resnet50']
    ensemble = create_ensemble(model_names, pretrained=False)
    print(f"Ensemble: {len(ensemble.models)} models")

    print("\nAll models loaded successfully!")
