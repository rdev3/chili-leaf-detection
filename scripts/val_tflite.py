"""Validasi akurasi model TFLite pada data uji.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)[None, ...]  # ke NCHW


def main():
    ap = argparse.ArgumentParser(description="Validasi akurasi model TFLite pada data uji.")
    ap.add_argument("--tflite", required=True, help="Path file model_ca.tflite.")
    ap.add_argument("--manifest", required=True, help="Path outputs/splits/manifest.csv.")
    args = ap.parse_args()

    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=args.tflite)
    it.allocate_tensors()
    detail_in = it.get_input_details()[0]
    detail_out = it.get_output_details()[0]

    benar, total = 0, 0
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            it.set_tensor(detail_in["index"], preprocess(row["path"]).astype(np.float32))
            it.invoke()
            prediksi = int(np.argmax(it.get_tensor(detail_out["index"])))
            benar += prediksi == int(row["label"])
            total += 1

    print(f"Citra uji: {total}")
    print(f"Akurasi TFLite: {benar / total * 100:.2f}% ({benar}/{total} benar)")


if __name__ == "__main__":
    main()
