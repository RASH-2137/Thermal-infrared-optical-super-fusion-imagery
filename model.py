"""
Thermal Infrared Super-Resolution Model

This module implements a fusion-based super-resolution model that combines:
- Low-resolution thermal images (preserving temperature fidelity)
- High-resolution optical images (providing spatial details)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from typing import Tuple, Optional


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and ReLU activation"""
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    """Channel attention module for feature refinement"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = avg_out + max_out
        attention = attention.view(b, c, 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention module for spatial feature refinement"""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


class FusionModule(nn.Module):
    """Attention-based fusion module for thermal and optical features"""
    
    def __init__(self, thermal_channels: int, optical_channels: int, 
                 fusion_channels: int = 64):
        super(FusionModule, self).__init__()
        self.thermal_conv = nn.Conv2d(thermal_channels, fusion_channels, 1)
        self.optical_conv = nn.Conv2d(optical_channels, fusion_channels, 1)
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(fusion_channels * 2)
        self.spatial_attention = SpatialAttention()
        
        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_channels * 2, fusion_channels, 3, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, thermal_feat: torch.Tensor, optical_feat: torch.Tensor) -> torch.Tensor:
        # Project to same channel dimension
        thermal_proj = self.thermal_conv(thermal_feat)
        optical_proj = self.optical_conv(optical_feat)
        
        # Align spatial dimensions before concatenation
        # Resize optical features to match thermal features spatial size
        if thermal_proj.shape[2:] != optical_proj.shape[2:]:
            optical_proj = F.interpolate(
                optical_proj, 
                size=thermal_proj.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Concatenate features
        fused = torch.cat([thermal_proj, optical_proj], dim=1)
        
        # Apply attention
        fused = self.channel_attention(fused)
        fused = self.spatial_attention(fused)
        
        # Final fusion
        fused = self.fusion_conv(fused)
        
        return fused


class ThermalEncoder(nn.Module):
    """Encoder for thermal images - preserves temperature fidelity"""
    
    def __init__(self, input_channels: int = 1, base_channels: int = 64):
        super(ThermalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, base_channels, 7, padding=3)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Downsampling blocks
        self.down1 = self._make_down_block(base_channels, base_channels * 2)
        self.down2 = self._make_down_block(base_channels * 2, base_channels * 4)
        self.down3 = self._make_down_block(base_channels * 4, base_channels * 8)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8)
        )
    
    def _make_down_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        x = self.res_blocks(x)
        
        return x


class OpticalEncoder(nn.Module):
    """Encoder for optical images - extracts spatial details"""
    
    def __init__(self, input_channels: int = 3, base_channels: int = 64):
        super(OpticalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, base_channels, 7, padding=3)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Downsampling blocks
        self.down1 = self._make_down_block(base_channels, base_channels * 2)
        self.down2 = self._make_down_block(base_channels * 2, base_channels * 4)
        self.down3 = self._make_down_block(base_channels * 4, base_channels * 8)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8)
        )
    
    def _make_down_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        x = self.res_blocks(x)
        
        return x


class SuperResolutionDecoder(nn.Module):
    """Decoder for super-resolution reconstruction"""
    
    def __init__(self, input_channels: int = 64, scale_factor: int = 4):
        super(SuperResolutionDecoder, self).__init__()
        self.scale_factor = scale_factor
        
        # Upsampling blocks
        self.up1 = self._make_up_block(input_channels, input_channels * 2)
        self.up2 = self._make_up_block(input_channels * 2, input_channels)
        self.up3 = self._make_up_block(input_channels, input_channels // 2)
        
        # Final reconstruction
        self.final_conv = nn.Sequential(
            nn.Conv2d(input_channels // 2, input_channels // 4, 3, padding=1),
            nn.BatchNorm2d(input_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 4, 1, 3, padding=1),
            nn.Sigmoid()  # Output in range [0, 1] to match thermal normalization
        )
        
        # Pixel shuffle for upsampling
        self.pixel_shuffle = nn.PixelShuffle(2)
    
    def _make_up_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )
    
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        
        x = self.final_conv(x)
        
        # Additional upsampling if needed
        if self.scale_factor > 4:
            x = F.interpolate(x, scale_factor=self.scale_factor // 4, mode='bilinear', align_corners=False)
        
        return x


class ThermalSuperResolutionNet(nn.Module):
    """
    Main thermal super-resolution network
    
    Architecture:
    1. Dual-branch encoder (thermal + optical)
    2. Attention-based fusion module
    3. Super-resolution decoder
    """
    
    def __init__(self, thermal_channels: int = 1, optical_channels: int = 3, 
                 base_channels: int = 64, scale_factor: int = 4):
        super(ThermalSuperResolutionNet, self).__init__()
        self.scale_factor = scale_factor
        
        # Encoders
        self.thermal_encoder = ThermalEncoder(thermal_channels, base_channels)
        self.optical_encoder = OpticalEncoder(optical_channels, base_channels)
        
        # Fusion module
        self.fusion = FusionModule(base_channels * 8, base_channels * 8, base_channels * 8)
        
        # Decoder
        self.decoder = SuperResolutionDecoder(base_channels * 8, scale_factor)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, lr_thermal: torch.Tensor, hr_optical: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            lr_thermal: Low-resolution thermal image (B, 1, H, W)
            hr_optical: High-resolution optical image (B, 3, H, W)
            
        Returns:
            sr_thermal: Super-resolved thermal image (B, 1, H*scale, W*scale)
        """
        # Encode features
        thermal_feat = self.thermal_encoder(lr_thermal)
        optical_feat = self.optical_encoder(hr_optical)
        
        # Fuse features
        fused_feat = self.fusion(thermal_feat, optical_feat)
        
        # Decode to super-resolution
        sr_thermal = self.decoder(fused_feat)
        
        return sr_thermal


class ThermalLoss(nn.Module):
    """Combined loss function for thermal super-resolution"""
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.1):
        super(ThermalLoss, self).__init__()
        self.alpha = alpha  # L1 loss weight
        self.beta = beta    # SSIM loss weight
        self.gamma = gamma  # Edge loss weight
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def ssim_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                  window_size: int = 11) -> torch.Tensor:
        """Compute SSIM loss"""
        def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
            coords = torch.arange(size, dtype=torch.float32)
            coords = coords - size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g = g / g.sum()
            return g.view(1, 1, 1, -1) * g.view(1, 1, -1, 1)
        
        window = gaussian_window(window_size).to(pred.device)
        
        mu1 = F.conv2d(pred, window, padding=window_size//2, groups=1)
        mu2 = F.conv2d(target, window, padding=window_size//2, groups=1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=1) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=1) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        eps = 1e-6  # Numerical stability epsilon
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)
        
        return 1 - ssim_map.mean()
    
    def edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute edge loss using Sobel operator"""
        def sobel_edge(img: torch.Tensor) -> torch.Tensor:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                 dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                 dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
            
            edge_x = F.conv2d(img, sobel_x, padding=1)
            edge_y = F.conv2d(img, sobel_y, padding=1)
            edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
            return edge
        
        pred_edge = sobel_edge(pred)
        target_edge = sobel_edge(target)
        
        return self.l1_loss(pred_edge, target_edge)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute combined loss with NaN/Inf protection
        
        Args:
            pred: Predicted thermal image
            target: Ground truth thermal image
            
        Returns:
            Dictionary containing individual and total losses
        """
        # L1 loss (thermal fidelity) with NaN protection
        l1 = self.l1_loss(pred, target)
        if torch.isnan(l1) or torch.isinf(l1):
            l1 = torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # SSIM loss (structural similarity) with NaN protection
        try:
            ssim = self.ssim_loss(pred, target)
            if torch.isnan(ssim) or torch.isinf(ssim):
                ssim = torch.tensor(0.0, device=pred.device, requires_grad=True)
        except:
            ssim = torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Edge loss (spatial details) with NaN protection
        try:
            edge = self.edge_loss(pred, target)
            if torch.isnan(edge) or torch.isinf(edge):
                edge = torch.tensor(0.0, device=pred.device, requires_grad=True)
        except:
            edge = torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Total loss with NaN protection
        total = self.alpha * l1 + self.beta * ssim + self.gamma * edge
        if torch.isnan(total) or torch.isinf(total):
            total = torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return {
            'total': total,
            'l1': l1,
            'ssim': ssim,
            'edge': edge
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing Thermal Super-Resolution Model...")
    
    # Create model
    model = ThermalSuperResolutionNet(scale_factor=4)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    lr_thermal = torch.randn(batch_size, 1, 64, 64)
    hr_optical = torch.randn(batch_size, 3, 256, 256)
    
    with torch.no_grad():
        sr_thermal = model(lr_thermal, hr_optical)
        print(f"Input LR thermal shape: {lr_thermal.shape}")
        print(f"Input HR optical shape: {hr_optical.shape}")
        print(f"Output SR thermal shape: {sr_thermal.shape}")
    
    # Test loss function
    loss_fn = ThermalLoss()
    target = torch.randn(batch_size, 1, 256, 256)
    losses = loss_fn(sr_thermal, target)
    
    print(f"Losses: {losses}")
    print("Model test completed!")
