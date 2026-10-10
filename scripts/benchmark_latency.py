"""Ukur jumlah parameter dan waktu inferensi per citra di CPU untuk tiap varian.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import LeafDataset, eval_transform
from src.models import build_model

VARIANTS = ["baseline", "se", "cbam", "ca"]


def cari_checkpoint_terbaik(variant):
    # ambil seed dengan f1 tertinggi menurut test_metrics.json
    terbaik, f1_terbaik = None, -1.0
    for rundir in (ROOT / "outputs" / "runs" / variant).glob("seed*"):
        mf, ckpt = rundir / "test_metrics.json", rundir / "best_phase2.pt"
        if not (mf.is_file() and ckpt.is_file()):
            continue
        f1 = json.loads(mf.read_text(encoding="utf-8")).get("f1", 0)
        if f1 > f1_terbaik:
            terbaik, f1_terbaik = ckpt, f1
    return terbaik


def satu_contoh_uji(manifest_csv, image_size):
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                ds = LeafDataset([(row["path"], int(row["label"]))], eval_transform(image_size))
                x, _ = ds[0]
                return x.unsqueeze(0)
    raise SystemExit("tidak ada baris test di manifest.csv")


@torch.no_grad()
def ukur_latency(model, contoh, n_ukur=100):
    model.eval()
    for _ in range(10):  # pemanasan, hasilnya dibuang
        model(contoh)
    t0 = time.perf_counter()
    for _ in range(n_ukur):
        model(contoh)
    return (time.perf_counter() - t0) / n_ukur * 1000


def main():
    ap = argparse.ArgumentParser(description="Benchmark jumlah parameter dan latency CPU.")
    ap.add_argument("--n", type=int, default=100, help="jumlah inferensi yang diukur")
    args = ap.parse_args()

    manifest = ROOT / "outputs" / "splits" / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit("manifest.csv tidak ketemu, jalankan prepare_splits.py dulu")

    contoh = satu_contoh_uji(manifest, image_size=224)
    hasil = {}
    print("Benchmark jalan di CPU, satu citra per inferensi.")
    for variant in VARIANTS:
        ckpt = cari_checkpoint_terbaik(variant)
        if ckpt is None:
            print(f"{variant}: checkpoint tidak ketemu, dilewati")
            continue
        model, _ = build_model(variant, pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        params = sum(p.numel() for p in model.parameters())
        latency = ukur_latency(model, contoh, n_ukur=args.n)
        size_mb = ckpt.stat().st_size / 1024 / 1024
        hasil[variant] = {
            "params": params,
            "size_mb": round(size_mb, 2),
            "latency_ms": round(latency, 2),
            "checkpoint": ckpt.name,
        }
        print(f"{variant}: {params / 1e6:.2f} juta parameter, {size_mb:.2f} MB, {latency:.2f} ms per citra")

    outdir = ROOT / "outputs" / "summary"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "latency.json").write_text(json.dumps(hasil, indent=2), encoding="utf-8")
    print("hasil tersimpan di outputs/summary/latency.json")


if __name__ == "__main__":
    main()
