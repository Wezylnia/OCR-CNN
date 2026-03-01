"""
CRNN (CNN + RNN + CTC) metin tanima modeli
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM blogu"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.0
    ):
        super().__init__()
        # dropout nn.LSTM'e VERILMEZ (num_layers=1 olduğunda PyTorch uyarı üretir).
        # Yerine ayrı nn.Dropout katmanı kullanılır — davranış aynıdır.
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_size]
            
        Returns:
            [batch, seq_len, output_size]
        """
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        output = self.dropout(output)
        output = self.linear(output)
        return output


class VGGEncoder(nn.Module):
    """VGG-style CNN encoder for feature extraction"""
    
    def __init__(self, input_channels: int = 1):
        super().__init__()
        
        # Kanal sayilari: 64 -> 128 -> 256 -> 256 -> 512 -> 512 -> 512
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # H/2
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # H/4
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d((2, 1), (2, 1))  # H/8, W ayni
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d((2, 1), (2, 1))  # H/16, W ayni
        
        self.conv5 = nn.Conv2d(512, 512, kernel_size=2, padding=0)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.output_channels = 512
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]
            
        Returns:
            [batch, channels, 1, width'] - height 1'e dusurulur
        """
        # Conv block 1
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Conv block 2
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Conv block 3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.pool3(x)
        
        # Conv block 4
        x = self.relu(self.bn4(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        
        # Conv block 5
        x = self.relu(self.bn5(self.conv5(x)))
        
        return x


class MobileNetV3Encoder(nn.Module):
    """
    MobileNetV3-small tabanli encoder — VGGEncoder drop-in yedegi.

    VGG'ye kiyasla ~%%15 daha hizli, benzer dogruluk.
    Ilk 4 MobileNet blogu kullanilir (3x stride-2 => H/8, W/8),
    ardindan Conv(H=4, W=1) ile height kolapse edilir.

    Gereksinim: torchvision
    """

    def __init__(self, input_channels: int = 1, pretrained: bool = True):
        super().__init__()

        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
        except (ImportError, AttributeError):
            try:
                import torchvision.models as _tvm
                backbone = _tvm.mobilenet_v3_small(pretrained=pretrained)
            except TypeError:
                import torchvision.models as _tvm
                backbone = _tvm.mobilenet_v3_small()

        feat_list = list(backbone.features.children())

        # Ilk konvolusyonu 1-kanal girdi icin uyarla
        first_block = feat_list[0]   # Conv2dNormActivation
        old_conv = first_block[0]    # Conv2d(3, 16, 3, stride=2, padding=1)
        new_conv = nn.Conv2d(
            input_channels, 16,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        first_block[0] = new_conv

        # feat_list[0..3]: 3x stride-2 => H/8=4, W/8 (H=32 girdi varsayimi)
        # [0] Conv2dNormActivation stride=(2,2) => [B, 16, 16, W/2]
        # [1] InvertedResidual      stride=(2,2) => [B, 16,  8, W/4]
        # [2] InvertedResidual      stride=(2,2) => [B, 24,  4, W/8]
        # [3] InvertedResidual      stride=(1,1) => [B, 24,  4, W/8]
        self.features = nn.Sequential(*feat_list[:4])

        # H=4 -> 1 kolaps + kanal projeksiyonu (512 = VGGEncoder ile ayni)
        self.bridge = nn.Sequential(
            nn.Conv2d(24, 512, kernel_size=(4, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.Hardswish(inplace=True),
        )

        self.output_channels = 512  # VGGEncoder ile ayni -> CRNN dogrudan uyumlu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, 32, W]  — yukseklik 32 olmali

        Returns:
            [B, 512, 1, W/8]
        """
        x = self.features(x)  # [B, 24, 4, W/8]
        x = self.bridge(x)    # [B, 512, 1, W/8]
        return x


class CRNN(nn.Module):
    """
    CRNN - Convolutional Recurrent Neural Network
    
    Metin tanima icin CNN + BiLSTM + CTC mimarisi.
    """
    
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        encoder_type: str = "vgg"
    ):
        """
        Args:
            num_classes: Karakter sayisi (vocab size + blank)
            input_channels: Giris kanal sayisi (1=grayscale, 3=RGB)
            hidden_size: LSTM hidden size
            num_layers: LSTM katman sayisi
            dropout: Dropout orani
            encoder_type: CNN encoder tipi (vgg veya resnet)
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # CNN Encoder
        if encoder_type == "vgg":
            self.encoder = VGGEncoder(input_channels)
        elif encoder_type == "mobilenetv3":
            self.encoder = MobileNetV3Encoder(input_channels)
        else:
            raise ValueError(f"Bilinmeyen encoder tipi: {encoder_type}")
        
        encoder_output = self.encoder.output_channels
        
        # Sequence modeling (BiLSTM)
        self.sequence = nn.Sequential()
        
        for i in range(num_layers):
            input_dim = encoder_output if i == 0 else hidden_size
            self.sequence.add_module(
                f'bilstm_{i}',
                BidirectionalLSTM(
                    input_dim,
                    hidden_size,
                    hidden_size,
                    dropout if i < num_layers - 1 else 0
                )
            )
        
        # Output layer
        self.output = nn.Linear(hidden_size, num_classes)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Agirliklari initialize et"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Giris gorseli [batch, channels, height, width]
               height = 32 olmali
               
        Returns:
            Log probabilities [seq_len, batch, num_classes]
        """
        # CNN feature extraction
        # [batch, channels, height, width] -> [batch, 512, 1, width']
        conv_out = self.encoder(x)
        
        # Reshape for RNN
        # [batch, 512, 1, width'] -> [batch, width', 512]
        batch, channels, height, width = conv_out.size()
        assert height == 1, (
            f"CRNN encoder ciktisi height=1 olmali, ancak {height} geldi. "
            f"Giris gorselinin yuksekligi encoder'in beklentisiyle uyusmuyor. "
            f"Input shape: {x.shape}"
        )
        
        conv_out = conv_out.squeeze(2)  # [batch, channels, width]
        conv_out = conv_out.permute(0, 2, 1)  # [batch, width, channels]
        
        # Sequence modeling
        # [batch, width, channels] -> [batch, width, hidden_size]
        rnn_out = self.sequence(conv_out)
        
        # Output layer
        # [batch, width, hidden_size] -> [batch, width, num_classes]
        output = self.output(rnn_out)
        
        # CTC icin: [seq_len, batch, num_classes]
        output = output.permute(1, 0, 2)
        
        # Log softmax
        output = F.log_softmax(output, dim=2)
        
        return output
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference icin
        
        Args:
            x: Giris gorseli [batch, channels, height, width]
            
        Returns:
            Log probabilities [seq_len, batch, num_classes]
        """
        self.eval()
        return self.forward(x)
    
    def get_sequence_length(self, input_width: int) -> int:
        """Encoder tipine gore cikis sequence uzunlugunu hesapla.

        VGG:          Pool1 W/2, Pool2 W/2, Conv5 W-1  -> (W//4) - 1
        MobileNetV3:  3x stride-2 in width             -> W//8
        """
        if isinstance(self.encoder, MobileNetV3Encoder):
            return input_width // 8
        # VGG encoder (varsayilan)
        return (input_width // 4) - 1


class CRNNLoss(nn.Module):
    """CTC Loss wrapper"""
    
    def __init__(self, blank_idx: int = 0):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            log_probs: [seq_len, batch, num_classes]
            targets: [sum(target_lengths)] veya [batch, max_target_len]
            input_lengths: [batch]
            target_lengths: [batch]
            
        Returns:
            CTC loss
        """
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


def build_crnn(
    num_classes: int,
    input_channels: int = 1,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    encoder_type: str = "vgg",
    weights_path: Optional[str] = None
) -> CRNN:
    """
    CRNN model olustur
    
    Args:
        num_classes: Karakter sayisi
        input_channels: Giris kanal sayisi
        hidden_size: LSTM hidden size
        num_layers: LSTM katman sayisi
        dropout: Dropout orani
        encoder_type: CNN encoder tipi
        weights_path: Model agirlik dosyasi
        
    Returns:
        CRNN model
    """
    model = CRNN(
        num_classes=num_classes,
        input_channels=input_channels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        encoder_type=encoder_type
    )
    
    if weights_path:
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
    
    return model
