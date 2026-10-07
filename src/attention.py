"""Modul attention yang dipasang setelah backbone MobileNetV3.

Tiga varian pembanding di BAB 4: SE, CBAM, dan Coordinate Attention.
Varian baseline memakai nn.Identity, artinya tanpa modul tambahan.

Rujukan rumus ada di BAB 3 laporan: 3.2 untuk SE, 3.3 untuk CBAM,
3.4 sampai 3.6 untuk Coordinate Attention.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation.

    Tiap channel diberi bobot berdasarkan rata-rata global seluruh
    feature map, jadi channel yang informatif dikuatkan.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.excite(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * s


class ChannelAttention(nn.Module):
    """Bagian channel dari CBAM: gabungan average-pool dan max-pool."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        b, c, _, _ = x.shape
        a = self.mlp(self.avg_pool(x).view(b, c)) + self.mlp(self.max_pool(x).view(b, c))
        return torch.sigmoid(a).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Bagian spasial dari CBAM: bobot untuk tiap posisi piksel."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Channel attention dulu, hasilnya dilanjut spatial attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        return x * self.sa(x)


class CoordAttention(nn.Module):
    """Coordinate Attention.

    Pooling dipisah per sumbu horizontal dan vertikal supaya informasi
    posisi tidak hilang seperti pada global average pooling biasa.
    """

    def __init__(self, channels, reduction=32):
        super().__init__()
        mid = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w
