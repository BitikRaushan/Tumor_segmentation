"""
Complete Segmentation Network: CRes2Net + MDA-SGT + Multi-Scale Skip Decoder
Includes Monte Carlo Dropout for Epistemic Uncertainty Estimation
"""
import torch
import torch.nn as nn
from models.cres2net import CRes2NetBlock
from models.attention import MDASGTBottleneck

class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Verify and align shapes if odd dimension rounding occurs
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class BR_UA_MDASGTNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1, mc_dropout_rate: float = 0.2):
        super().__init__()
        # Stem: [B, 1, 224, 224] -> [B, 64, 224, 224]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: [B, 64, 224, 224]
        self.enc1 = CRes2NetBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2) # -> 112 x 112

        # Stage 2: [B, 128, 112, 112]
        self.enc2 = CRes2NetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2) # -> 56 x 56

        # Stage 3: [B, 256, 56, 56]
        self.enc3 = CRes2NetBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2) # -> 28 x 28

        # Stage 4: [B, 512, 28, 28]
        self.enc4 = CRes2NetBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2) # -> 14 x 14

        # Attention Bottleneck: [B, 512, 14, 14]
        self.bottleneck = MDASGTBottleneck(channels=512, embed_dim=768)

        # Decoder Stages
        # dec4: upsamples 14x14 -> 28x28, concatenates e4 (512 ch) -> outputs 256 ch
        self.dec4 = DecoderBlock(in_ch=512, skip_ch=512, out_ch=256)
        
        # dec3: upsamples 28x28 -> 56x56, concatenates e3 (256 ch) -> outputs 128 ch
        self.dec3 = DecoderBlock(in_ch=256, skip_ch=256, out_ch=128)
        
        # dec2: upsamples 56x56 -> 112x112, concatenates e2 (128 ch) -> outputs 64 ch
        self.dec2 = DecoderBlock(in_ch=128, skip_ch=128, out_ch=64)
        
        # dec1: upsamples 112x112 -> 224x224, concatenates e1 (64 ch) -> outputs 32 ch
        self.dec1 = DecoderBlock(in_ch=64, skip_ch=64, out_ch=32)

        # Monte Carlo Dropout for Epistemic Uncertainty Estimation
        self.mc_dropout = nn.Dropout2d(p=mc_dropout_rate)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.stem(x)         # [B, 64, 224, 224]
        e1 = self.enc1(x0)         # [B, 64, 224, 224] (skip 1)
        
        x1 = self.pool1(e1)        # [B, 64, 112, 112]
        e2 = self.enc2(x1)         # [B, 128, 112, 112] (skip 2)
        
        x2 = self.pool2(e2)        # [B, 128, 56, 56]
        e3 = self.enc3(x2)         # [B, 256, 56, 56] (skip 3)
        
        x3 = self.pool3(e3)        # [B, 256, 28, 28]
        e4 = self.enc4(x3)         # [B, 512, 28, 28] (skip 4)
        
        x4 = self.pool4(e4)        # [B, 512, 14, 14]

        # Bottleneck
        b = self.bottleneck(x4)    # [B, 512, 14, 14]

        # Decoder with properly paired multi-scale skips
        d4 = self.dec4(b, e4)      # 14 -> 28 (pairs with e4)
        d3 = self.dec3(d4, e3)     # 28 -> 56 (pairs with e3)
        d2 = self.dec2(d3, e2)     # 56 -> 112 (pairs with e2)
        d1 = self.dec1(d2, e1)     # 112 -> 224 (pairs with e1)

        # Dropout & Output Head
        d1 = self.mc_dropout(d1)
        logits = self.final_conv(d1) # [B, 1, 224, 224]
        return logits

    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 20):
        self.train()  # Keep dropout active
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                prob = torch.sigmoid(self.forward(x))
                preds.append(prob.unsqueeze(0))
        
        preds = torch.cat(preds, dim=0)
        mean_prob = torch.mean(preds, dim=0)
        
        eps = 1e-8
        entropy = - (mean_prob * torch.log(mean_prob + eps) + 
                     (1.0 - mean_prob) * torch.log(1.0 - mean_prob + eps))
        return mean_prob, entropy