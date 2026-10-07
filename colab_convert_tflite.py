"""Konversi checkpoint PyTorch ke TFLite — dijalankan di Google Colab.

Kenapa di Colab: ai-edge-torch tidak punya paket Windows, jadi konversi
tidak bisa jalan di laptop. Di Colab (Linux) instalasinya normal.

Langkah:
1. Buka https://colab.research.google.com, buat notebook baru.
2. Di satu sel, jalankan:  !pip install -q litert-torch
3. Upload DUA file ke Colab via panel Files (ikon folder di kiri):
     - file ini (colab_convert_tflite.py)
     - best_phase2.pt  (dari outputs/runs/ca/seed123/ di laptop)
4. Jalankan sel berisi:  !python colab_convert_tflite.py
5. File model_ca.tflite otomatis terdownload. Catat ukurannya
   untuk laporan, lalu simpan berdampingan dengan benchmark.json.

Arsitektur di bawah ini disalin persis dari src/attention.py dan
src/models.py di proyek, jadi bobot checkpoint cocok tanpa ubahan.
"""

import torch
import torch.nn as nn

CKPT_PATH = "best_phase2.pt"
OUT_PATH = "model_ca.tflite"


class CoordAttention(nn.Module):
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


def build_ca_model(num_classes=2):
    from torchvision.models import mobilenet_v3_small

    backbone = mobilenet_v3_small(weights=None).features
    head = nn.Sequential(
        CoordAttention(576),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Dropout(0.2),
        nn.Linear(576, num_classes),
    )
    return nn.Sequential(backbone, head)


def main():
    import litert_torch

    model = build_ca_model()
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu", weights_only=True))
    model.eval()
    print(f"checkpoint {CKPT_PATH} dimuat.")

    sample = (torch.randn(1, 3, 224, 224),)
    edge_model = litert_torch.convert(model, sample)
    edge_model.export(OUT_PATH)

    import os

    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"{OUT_PATH} tersimpan, ukuran {size_mb:.2f} MB.")

    try:
        from google.colab import files

        files.download(OUT_PATH)
        print("file otomatis terdownload ke laptopmu.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
