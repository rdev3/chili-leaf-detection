#!/usr/bin/env python3
"""Training 4 varian model x 3 seed dengan transfer learning dua fase.

Fase 1: backbone dibekukan, hanya classifier head yang dilatih.
Fase 2: seluruh jaringan di-fine-tuning dengan learning rate kecil.
Tiap fase memakai early stopping berdasarkan F1 data validasi.

Contoh pakai dari root proyek:
    python scripts/train.py                                # semua, 12x training
    python scripts/train.py --variants baseline --seeds 42 # smoke test cepat
    python scripts/train.py --overwrite                     # ulangi run yang selesai
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import make_loaders
from src.models import build_model, set_backbone_frozen


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in tqdm(loader, leave=False, desc="train"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        n += len(x)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    losses, preds, labels = [], [], []
    for x, y in tqdm(loader, leave=False, desc="eval"):
        x, y = x.to(device), y.to(device)
        out = model(x)
        losses.append(loss_fn(out, y).item())
        preds += out.argmax(1).cpu().tolist()
        labels += y.cpu().tolist()
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def train_phase(model, backbone, loaders, phase_cfg, es_cfg, device, ckpt_path, phase_name):
    set_backbone_frozen(backbone, phase_cfg["freeze_backbone"])
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=phase_cfg["lr"]
    )
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    patience_left = es_cfg["patience"]
    history = []

    print(f"  {phase_name}: lr={phase_cfg['lr']}, backbone beku={phase_cfg['freeze_backbone']}")
    for epoch in range(1, phase_cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, loss_fn, device)
        val = evaluate(model, loaders["val"], loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val})
        print(
            f"  epoch {epoch:3d} train_loss={train_loss:.4f} "
            f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f} val_f1={val['f1']:.4f}"
        )
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_left = es_cfg["patience"]
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  early stopping di epoch {epoch}, val_f1 terbaik {best_f1:.4f}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, ckpt_path)
    return history


def run_experiment(variant, seed, cfg, manifest, device, rundir, overwrite):
    test_metrics_path = rundir / "test_metrics.json"
    if test_metrics_path.exists() and not overwrite:
        print(f"[{variant} | seed {seed}] sudah ada hasilnya, dilewati")
        return

    rundir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model, backbone = build_model(variant, pretrained=True)
    model.to(device)
    loaders = make_loaders(
        manifest,
        cfg["data"]["image_size"],
        cfg["training"]["batch_size"],
        cfg["data"].get("num_workers", 0),
    )

    tcfg = cfg["training"]
    history = {}

    print(f"[{variant} | seed {seed}] fase 1")
    history["phase1"] = train_phase(
        model, backbone, loaders, tcfg["phase1"], tcfg["early_stopping"],
        device, rundir / "best_phase1.pt", "fase 1",
    )
    print(f"[{variant} | seed {seed}] fase 2")
    history["phase2"] = train_phase(
        model, backbone, loaders, tcfg["phase2"], tcfg["early_stopping"],
        device, rundir / "best_phase2.pt", "fase 2",
    )

    test = evaluate(model, loaders["test"], nn.CrossEntropyLoss(), device)
    with open(rundir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump({"variant": variant, "seed": seed, **test}, f, indent=2)
    print(
        f"[{variant} | seed {seed}] test "
        f"acc={test['accuracy']:.4f} prec={test['precision']:.4f} "
        f"rec={test['recall']:.4f} f1={test['f1']:.4f}"
    )


def main():
    ap = argparse.ArgumentParser(description="Training 4 varian x 3 seed.")
    ap.add_argument("--variants", nargs="*", default=None, help="subset varian, mis. baseline ca")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="subset seed, mis. 42")
    ap.add_argument("--overwrite", action="store_true", help="ulangi run yang sudah selesai")
    args = ap.parse_args()

    with open(ROOT / "configs" / "experiment.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    variants = args.variants or cfg["variants"]
    unknown = [v for v in variants if v not in cfg["variants"]]
    if unknown:
        print(f"varian tidak dikenal: {unknown}, pilih dari {cfg['variants']}")
        return 1
    seeds = args.seeds or cfg["training"]["seeds"]

    manifest = ROOT / "outputs" / "splits" / "manifest.csv"
    if not manifest.is_file():
        print("manifest.csv tidak ketemu, jalankan prepare_splits.py dulu")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    else:
        print("peringatan: berjalan di CPU, training akan jauh lebih lambat")

    t0 = time.time()
    for variant in variants:
        for seed in seeds:
            run_experiment(
                variant, seed, cfg, manifest, device,
                ROOT / "outputs" / "runs" / variant / f"seed{seed}",
                args.overwrite,
            )
    print(f"selesai dalam {(time.time() - t0) / 60:.0f} menit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
