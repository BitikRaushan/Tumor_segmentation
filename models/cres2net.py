"""
CRes2Net: Convolutional Res2Net Module with Squeeze-and-Excitation (SE) Attention
Reconstructed from BSPC 2025 Paper (Figure 2 & Section 3.2)
"""
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for dynamic channel recalibration."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_ch = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_ch, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CRes2NetBlock(nn.Module):
    """
    Hierarchical multi-scale Res2Net block combined with SE recalibration.
    Partitions channels into 4 groups and connects them in a cascade.
    """
    def __init__(self, in_channels: int, out_channels: int, scale: int = 4, stride: int = 1):
        super().__init__()
        self.scale = scale
        self.out_channels = out_channels
        self.width = out_channels // scale
        
        # 1x1 Conv to adjust initial channel capacity
        self.conv1 = nn.Conv2d(in_channels, self.width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width * scale)
        self.relu = nn.ReLU(inplace=True)

        # 3x3 Convolutions for hierarchical scales (scale - 1 = 3 convolutions)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.width, self.width, kernel_size=3, stride=stride, padding=1, bias=False)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(self.width) for _ in range(scale - 1)
        ])

        # 1x1 Conv for feature fusion
        self.conv3 = nn.Conv2d(self.width * scale, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Squeeze-and-Excitation integration
        self.se = SEBlock(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        
        # Split feature map into 'scale' partitions along channel dimension
        chunks = torch.chunk(out, self.scale, dim=1)
        y = []
        for i in range(self.scale):
            if i == 0:
                y.append(chunks[i])
            elif i == 1:
                y.append(self.relu(self.bns[i - 1](self.convs[i - 1](chunks[i]))))
            else:
                # Hierarchical cascade: add output of previous group
                t = chunks[i] + y[i - 1]
                y.append(self.relu(self.bns[i - 1](self.convs[i - 1](t))))

        out = torch.cat(y, dim=1)
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out = self.relu(out + residual)
        return out