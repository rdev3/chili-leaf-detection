#!/usr/bin/env python3
"""Ringkasan 12 run training: tabel perbandingan + confusion matrix.

Dibaca dari outputs/runs/<varian>/seed<seed>/test_metrics.json dan
checkpoint best_phase2.pt tiap run. Run yang belum ada hasilnya
dilewati dan dilaporkan apa adanya.

Contoh pakai dari root proyek:
    python scripts/evaluate.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import LeafDataset, eval_transform
from src.models import build_model

METRICS = ["accuracy", "precision", "recall", "f1"]
LABELS = [0, 1]
LABEL_NAMES = ["healthy", "bacterial_spot"]


def read_run_metrics(rundir):
    path = rundir / "test_metrics.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_rows(manifest_csv):
    rows = []
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                rows.append((row["path"], int(row["label"])))
    return rows


@torch.no_grad()
def predict_test(variant, ckpt, rows, image_size, batch_size, device):
    model, _ = build_model(variant, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    loader = DataLoader(
        LeafDataset(rows, eval_transform(image_size)), batch_size=batch_size
    )
    preds, labels = [], []
    for x, y in loader:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
        labels += y.tolist()
    return labels, preds


def mean_std(values):
    a = np.array(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(description="Ringkasan hasil training.")
    ap.add_argument("--outdir", default="outputs/summary", help="folder keluaran")
    args = ap.parse_args()

    with open(ROOT / "configs" / "experiment.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    variants = cfg["variants"]
    seeds = cfg["training"]["seeds"]
    image_size = cfg["data"]["image_size"]
    batch_size = cfg["training"]["batch_size"]

    manifest = ROOT / "outputs" / "splits" / "manifest.csv"
    if not manifest.is_file():
        print("manifest.csv tidak ketemu, jalankan prepare_splits.py dulu")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Kumpulkan metrik tiap run yang sudah selesai.
    runs = {}
    missing = []
    for variant in variants:
        for seed in seeds:
            rundir = ROOT / "outputs" / "runs" / variant / f"seed{seed}"
            m = read_run_metrics(rundir)
            if m is None:
                missing.append(f"{variant}/seed{seed}")
            else:
                runs.setdefault(variant, []).append((seed, m, rundir))

    total = len(variants) * len(seeds)
    print(f"{total - len(missing)}/{total} run ditemukan")
    for name in missing:
        print(f"  belum ada: {name}")

    # Tabel perbandingan: mean +- std tiap varian.
    table = []
    for variant in variants:
        if variant not in runs:
            continue
        row = {"variant": variant, "n_runs": len(runs[variant])}
        for metric in METRICS:
            mean, std = mean_std([m[metric] for _, m, _ in runs[variant]])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        table.append(row)

    with open(outdir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["variant", "n_runs"]
        for metric in METRICS:
            header += [f"{metric}_mean", f"{metric}_std"]
        w.writerow(header)
        for row in table:
            cells = [row["variant"], row["n_runs"]]
            for m in METRICS:
                cells.append(f"{row[f'{m}_mean']:.6f}")
                cells.append(f"{row[f'{m}_std']:.6f}")
            w.writerow(cells)

    lines = [
        "| Varian | Akurasi | Presisi | Recall | F1 |",
        "|---|---|---|---|---|",
    ]
    for row in table:
        cells = [row["variant"]]
        for m in METRICS:
            cells.append(f"{row[f'{m}_mean']:.4f} ± {row[f'{m}_std']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    (outdir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(" ".join(f"{h:17s}" for h in ["varian", "akurasi", "presisi", "recall", "f1"]))
    for row in table:
        cells = [row["variant"]]
        for m in METRICS:
            cells.append(f"{row[f'{m}_mean']:.4f} ± {row[f'{m}_std']:.4f}")
        print(" ".join(f"{c:17s}" for c in cells))

    # Confusion matrix tiap run dari checkpoint fase 2, lalu rata-rata per varian.
    rows = test_rows(manifest)
    per_run_cm = []
    variant_cm = {}
    for variant in variants:
        mats = []
        for seed, m, rundir in runs.get(variant, []):
            ckpt = rundir / "best_phase2.pt"
            if not ckpt.is_file():
                print(f"  checkpoint hilang: {variant}/seed{seed}, matrix dilewati")
                continue
            labels, preds = predict_test(
                variant, ckpt, rows, image_size, batch_size, device
            )
            cm = confusion_matrix(labels, preds, labels=LABELS)
            acc_cm = cm.diagonal().sum() / cm.sum()
            if abs(acc_cm - m["accuracy"]) > 1e-6:
                print(
                    f"  peringatan: akurasi matrix {variant}/seed{seed} "
                    f"({acc_cm:.4f}) beda dari test_metrics.json ({m['accuracy']:.4f})"
                )
            tn, fp, fn, tp = cm.ravel()
            per_run_cm.append([variant, seed, tn, fp, fn, tp])
            mats.append(cm)
        if mats:
            variant_cm[variant] = np.mean(mats, axis=0)

    with open(outdir / "confusion_matrix_per_run.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed", "tn", "fp", "fn", "tp"])
        w.writerows(per_run_cm)

    for variant, cm in variant_cm.items():
        with open(outdir / f"confusion_matrix_{variant}_mean.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["aktual \\ prediksi"] + LABEL_NAMES)
            for name, line in zip(LABEL_NAMES, cm):
                w.writerow([name] + [f"{v:.1f}" for v in line])

    print()
    for variant, cm in variant_cm.items():
        print(f"confusion matrix rata-rata {variant} (baris=aktual, kolom=prediksi):")
        print(f"              {LABEL_NAMES[0]:>14s} {LABEL_NAMES[1]:>14s}")
        for name, line in zip(LABEL_NAMES, cm):
            print(f"  {name:12s} {line[0]:14.1f} {line[1]:14.1f}")

    # Model terbaik: f1 test tertinggi, untuk konversi TFLite nanti.
    best = None
    per_variant_best = {}
    for variant in variants:
        done = runs.get(variant, [])
        if not done:
            continue
        seed, m, rundir = max(done, key=lambda r: r[1]["f1"])
        info = {
            "variant": variant,
            "seed": seed,
            "checkpoint": str(rundir.relative_to(ROOT) / "best_phase2.pt"),
            "test_f1": m["f1"],
            "test_accuracy": m["accuracy"],
        }
        per_variant_best[variant] = info
        if best is None or m["f1"] > best["test_f1"]:
            best = info

    if best:
        (outdir / "best_model.json").write_text(
            json.dumps({"overall": best, "per_variant": per_variant_best},
                       indent=2),
            encoding="utf-8",
        )
        print()
        print(f"model terbaik: {best['variant']} seed {best['seed']} "
              f"(f1 {best['test_f1']:.4f})")
    print(f"ringkasan tersimpan di {outdir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
