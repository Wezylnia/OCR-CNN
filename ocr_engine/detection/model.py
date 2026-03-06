"""
DBNet (Differentiable Binarization Network) metin tespit modeli
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Optional, Tuple


class ConvBnRelu(nn.Module):
    """Conv + BatchNorm + ReLU blogu"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResNetBackbone(nn.Module):
    """ResNet backbone - feature extraction"""
    
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        
        if backbone_name == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.out_channels = [64, 128, 256, 512]
        elif backbone_name == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.out_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Desteklenmeyen backbone: {backbone_name}")
        
        # ResNet katmanlarini ayir
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Giris tensoru [B, 3, H, W]
            
        Returns:
            4 seviye feature map listesi
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c1 = self.layer1(x)   # 1/4
        c2 = self.layer2(c1)  # 1/8
        c3 = self.layer3(c2)  # 1/16
        c4 = self.layer4(c3)  # 1/32
        
        return [c1, c2, c3, c4]


class FPN(nn.Module):
    """Feature Pyramid Network - multi-scale feature fusion"""
    
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        
        # Lateral connections (1x1 conv)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, kernel_size=1)
            for ch in in_channels
        ])
        
        # Smooth layers (3x3 conv)
        self.smooth_convs = nn.ModuleList([
            ConvBnRelu(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])
        
        self.out_channels = out_channels
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: Backbone feature maps [c1, c2, c3, c4]
            
        Returns:
            Fused feature map
        """
        # Lateral connections
        laterals = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, features)
        ]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
        
        # Smooth
        outputs = [
            smooth_conv(lateral)
            for smooth_conv, lateral in zip(self.smooth_convs, laterals)
        ]
        
        # Hepsini ayni boyuta getir ve birlestir
        target_h, target_w = outputs[0].shape[2:]
        fused = outputs[0]
        for out in outputs[1:]:
            fused = fused + F.interpolate(
                out,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )
        
        return fused


class DBHead(nn.Module):
    """
    Differentiable Binarization Head
    
    Outputs:
        - probability map (text regions)
        - threshold map (adaptive threshold)
        - binary map (differentiable binarization)
    """
    
    def __init__(self, in_channels: int, k: int = 50):
        super().__init__()
        self.k = k  # Binarization aggressiveness
        
        # Probability map branch
        self.prob_conv = nn.Sequential(
            ConvBnRelu(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        
        # Threshold map branch
        self.thresh_conv = nn.Sequential(
            ConvBnRelu(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Feature map [B, C, H, W]
            return_maps: Tum map'leri dondur
            
        Returns:
            Dictionary: probability, threshold, binary maps
        """
        prob_map = self.prob_conv(x)
        thresh_map = self.thresh_conv(x)
        
        # Differentiable binarization
        # binary = 1 / (1 + exp(-k * (prob - thresh)))
        binary_map = torch.reciprocal(
            1 + torch.exp(-self.k * (prob_map - thresh_map))
        )
        
        outputs = {
            'prob_map': prob_map,
            'binary_map': binary_map
        }
        
        if return_maps:
            outputs['thresh_map'] = thresh_map
        
        return outputs


class DBNet(nn.Module):
    """
    DBNet - Differentiable Binarization Network
    
    Metin bolge tespiti icin end-to-end model.
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        fpn_channels: int = 256,
        k: int = 50
    ):
        """
        Args:
            backbone: Backbone adi (resnet18 veya resnet50)
            pretrained: Pre-trained agirliklar kullanilsin mi
            fpn_channels: FPN cikis kanal sayisi
            k: Binarization aggressiveness
        """
        super().__init__()
        
        # Backbone
        self.backbone = ResNetBackbone(backbone, pretrained)
        
        # Neck (FPN)
        self.fpn = FPN(self.backbone.out_channels, fpn_channels)
        
        # Head
        self.head = DBHead(fpn_channels, k)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Agirliklari initialize et (backbone haric)"""
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.fpn.apply(_init)
        self.head.apply(_init)
    
    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Giris gorseli [B, 3, H, W]
            return_maps: Threshold map'i de dondur
            
        Returns:
            Dictionary: probability, binary, (optional: threshold) maps
        """
        # Feature extraction
        features = self.backbone(x)
        
        # Feature fusion
        fused = self.fpn(features)
        
        # Detection head
        outputs = self.head(fused, return_maps)
        
        return outputs
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.3
    ) -> torch.Tensor:
        """
        Inference icin basitlestirilmis metod
        
        Args:
            x: Giris gorseli [B, 3, H, W]
            threshold: Probability threshold
            
        Returns:
            Binary prediction [B, 1, H, W]
        """
        self.eval()
        outputs = self.forward(x, return_maps=False)
        prob_map = outputs['prob_map']
        return (prob_map > threshold).float()


class DBLoss(nn.Module):
    """
    DBNet loss function
    
    L = L_prob + alpha * L_binary + beta * L_thresh
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 10.0,
        ohem_ratio: int = 3,
        eps: float = 1e-6
    ):
        """
        Args:
            alpha: Binary loss weight
            beta: Threshold loss weight
            ohem_ratio: OHEM negative/positive ratio
            eps: Numerical stability
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ohem_ratio = ohem_ratio
        self.eps = eps
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_prob: torch.Tensor,
        gt_thresh: torch.Tensor,
        gt_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model cikislari (prob_map, binary_map, thresh_map)
            gt_prob: Ground truth probability map
            gt_thresh: Ground truth threshold map
            gt_mask: Valid region mask
            
        Returns:
            Dictionary: total_loss, prob_loss, binary_loss, thresh_loss
        """
        prob_map = outputs['prob_map']
        binary_map = outputs['binary_map']
        thresh_map = outputs.get('thresh_map')
        
        # Probability loss (BCE with OHEM)
        prob_loss = self._balanced_bce_loss(prob_map, gt_prob, gt_mask)
        
        # Binary loss (Dice loss)
        binary_loss = self._dice_loss(binary_map, gt_prob, gt_mask)
        
        # Threshold loss (L1)
        if thresh_map is not None:
            thresh_loss = self._l1_loss(thresh_map, gt_thresh, gt_prob)
        else:
            thresh_loss = torch.tensor(0.0, device=prob_map.device)
        
        # Total loss
        total_loss = prob_loss + self.alpha * binary_loss + self.beta * thresh_loss
        
        return {
            'total_loss': total_loss,
            'prob_loss': prob_loss,
            'binary_loss': binary_loss,
            'thresh_loss': thresh_loss
        }
    
    def _balanced_bce_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Balanced BCE loss with OHEM"""
        positive = (gt > 0.5).float() * mask
        negative = (gt <= 0.5).float() * mask
        
        positive_count = int(positive.sum())
        negative_count = min(
            int(negative.sum()),
            positive_count * self.ohem_ratio
        )
        
        # BCE loss
        loss = F.binary_cross_entropy(pred, gt, reduction='none')
        
        # Positive loss
        positive_loss = (loss * positive).sum()
        
        # Negative loss (hard negative mining)
        negative_loss = loss * negative
        negative_loss, _ = negative_loss.view(-1).topk(negative_count)
        negative_loss = negative_loss.sum()
        
        total_count = positive_count + negative_count
        if total_count > 0:
            return (positive_loss + negative_loss) / total_count
        return torch.tensor(0.0, device=pred.device)
    
    def _dice_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Dice loss"""
        pred = pred * mask
        gt = gt * mask
        
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum()
        
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice
    
    def _l1_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """L1 loss for threshold map"""
        mask = (mask > 0.5).float()
        loss = torch.abs(pred - gt) * mask
        return loss.sum() / (mask.sum() + self.eps)


def build_dbnet(
    backbone: str = "resnet18",
    pretrained: bool = True,
    weights_path: Optional[str] = None
) -> DBNet:
    """
    DBNet model olustur
    
    Args:
        backbone: Backbone adi
        pretrained: Backbone icin pre-trained agirliklar
        weights_path: Model agirlik dosyasi
        
    Returns:
        DBNet model
    """
    model = DBNet(backbone=backbone, pretrained=pretrained)
    
    if weights_path:
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
    
    return model
