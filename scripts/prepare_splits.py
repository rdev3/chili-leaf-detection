#!/usr/bin/env python3
"""
prepare_splits.py -- Membuat split train/val/test dari dataset yang sudah
lolos verify_dataset.py.

Membaca laporan JSON verifikasi, mengumpulkan citra VALID per kelas
(file corrupt dilewati otomatis), lalu split 70:15:15 secara stratified
dengan split_seed TETAP dari configs/experiment.yaml.

Split memakai split_seed yang tetap dan berbeda dari training seeds.
Split dibuat sekali lalu dipakai untuk semua 12 training supaya
perbandingan antar varian adil. Yang boleh beda antar run hanya
inisialisasi bobot dan urutan batch, bukan komposisi datanya.

Cara pakai (dari root proyek):
    python scripts/prepare_splits.py --report outputs/dataset_report.json

Keluaran: outputs/splits/manifest.csv  (kolom: path,label,split)
          outputs/splits/label_map.json
Label: bacterial_spot = 1 (kelas positif / penyakit), healthy = 0.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# bacterial_spot = 1 karena itu kelas penyakit yang dideteksi
LABEL_MAP = {"bacterial_spot": 1, "healthy": 0}


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_valid_images(report: dict) -> tuple[list, list]:
    """Kumpulkan path dan label citra yang valid dari tiap kelas."""
    paths, labels = [], []
    for cls_name, label in LABEL_MAP.items():
        info = report["kelas"].get(cls_name)
        if not info or info.get("status") != "ok":
            print(f"FATAL: kelas '{cls_name}' tidak berstatus ok di laporan.")
            sys.exit(1)
        folder = Path(info["folder"])
        corrupt = {c["file"] for c in info.get("corrupt", [])}
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS and f.name not in corrupt:
                paths.append(str(f.resolve()))
                labels.append(label)
    return paths, labels


def main() -> int:
    ap = argparse.ArgumentParser(description="Buat split train/val/test stratified.")
    ap.add_argument("--report", required=True, help="Path dataset_report.json dari verify_dataset.py.")
    ap.add_argument("--outdir", default=None, help="Folder keluaran (default: outputs/splits).")
    args = ap.parse_args()

    cfg = load_config()
    split_cfg = cfg["data"]["split"]
    split_seed = cfg["data"]["split_seed"]
    test_ratio = split_cfg["test"]
    val_ratio = split_cfg["val"] / (1.0 - test_ratio)  # proporsi val dari sisa setelah test dipisah

    report_path = Path(args.report)
    if not report_path.is_file():
        print(f"FATAL: laporan tidak ketemu: {report_path}")
        print("Jalankan dulu: python scripts/verify_dataset.py --data <path dataset>")
        return 1
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    if report.get("kesimpulan") != "OK":
        print("FATAL: laporan verifikasi berstatus BERMASALAH. Perbaiki dulu, baru split.")
        return 1

    paths, labels = collect_valid_images(report)
    if not paths:
        print("FATAL: nol citra valid.")
        return 1

    # Pisahkan test dulu, lalu val dari sisanya. Keduanya stratified.
    p_trainval, p_test, y_trainval, y_test = train_test_split(
        paths, labels, test_size=test_ratio, random_state=split_seed, stratify=labels
    )
    p_train, p_val, y_train, y_val = train_test_split(
        p_trainval, y_trainval, test_size=val_ratio, random_state=split_seed, stratify=y_trainval
    )

    outdir = Path(args.outdir) if args.outdir else Path(__file__).resolve().parents[1] / "outputs" / "splits"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = (
        [(p, y, "train") for p, y in zip(p_train, y_train)]
        + [(p, y, "val") for p, y in zip(p_val, y_val)]
        + [(p, y, "test") for p, y in zip(p_test, y_test)]
    )
    manifest = outdir / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        w.writerows(rows)
    with open(outdir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"bacterial_spot": 1, "healthy": 0}, f, indent=2)

    print("=== Split dataset ===")
    print(f"split_seed: {split_seed}")
    for split in ("train", "val", "test"):
        n = sum(1 for _, _, s in rows if s == split)
        n_pos = sum(1 for _, y, s in rows if s == split and y == 1)
        print(f"  {split:5s}: {n:4d} citra  (bacterial_spot={n_pos}, healthy={n - n_pos})")
    print(f"Total: {len(rows)} citra")
    print(f"Manifest: {manifest}")
    print("Split selesai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
