import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model


def benchmark_latency(model, device, n_warmup=10, n_iter=100):
    model.eval()
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def main():
    ap = argparse.ArgumentParser(description="Konversi TFLite + benchmark.")
    ap.add_argument("--variant", default="ca", help="varian yang dikonversi")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    summary_path = ROOT / "outputs" / "summary" / "best_model.json"
    if not summary_path.is_file():
        print("best_model.json tidak ketemu, jalankan evaluate.py dulu")
        return 1
    with open(summary_path, encoding="utf-8") as f:
        best = json.load(f)
    info = best["per_variant"].get(args.variant)
    if info is None:
        print(f"varian '{args.variant}' tidak ada di best_model.json")
        return 1
    ckpt = ROOT / info["checkpoint"]
    if not ckpt.is_file():
        print(f"checkpoint tidak ketemu: {ckpt}")
        return 1

    outdir = Path(args.outdir) if args.outdir else ROOT / "outputs" / "export"
    outdir.mkdir(parents=True, exist_ok=True)

    model, _ = build_model(args.variant, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    print(f"checkpoint: {info['checkpoint']} (seed {info['seed']}, f1 {info['test_f1']:.4f})")

    cpu = torch.device("cpu")
    mean_ms, std_ms = benchmark_latency(model.to(cpu), cpu)
    pt_path = outdir / f"model_{args.variant}.pt"
    torch.save(model.state_dict(), pt_path)

    result = {
        "variant": args.variant,
        "seed": info["seed"],
        "checkpoint": info["checkpoint"],
        "params": int(sum(p.numel() for p in model.parameters())),
        "pt_size_mb": round(pt_path.stat().st_size / 1e6, 2),
        "torch_cpu_latency_ms": {"mean": round(mean_ms, 2), "std": round(std_ms, 2)},
        "tflite_size_mb": None,
    }

    try:
        import ai_edge_torch

        edge = ai_edge_torch.convert(model, (torch.randn(1, 3, 224, 224),))
        tflite_path = outdir / f"model_{args.variant}.tflite"
        edge.export(str(tflite_path))
        result["tflite_size_mb"] = round(tflite_path.stat().st_size / 1e6, 2)
        print(f"TFLite tersimpan: {tflite_path} ({result['tflite_size_mb']} MB)")
    except ImportError:
        print("ai-edge-torch belum terinstal, konversi TFLite dilewati.")
        print("Di Windows paket ini memang tidak tersedia; pakai")
        print("colab_convert_tflite.py di Google Colab sebagai gantinya.")
    except Exception as e:
        print(f"konversi TFLite gagal: {e}")

    with open(outdir / "benchmark.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"latensi CPU: {mean_ms:.1f} ± {std_ms:.1f} ms per citra")
    print(f"ukuran .pt: {result['pt_size_mb']} MB, params: {result['params']:,}")
    print(f"benchmark.json tersimpan di {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
