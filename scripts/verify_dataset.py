#!/usr/bin/env python3
"""
verify_dataset.py -- Verifikasi dataset PlantVillage lokal sebelum training.

Mengecek:
  1. Folder dua kelas pepper ketemu (bacterial spot & healthy),
     tahan terhadap variasi penamaan folder dan struktur nested
     (mis. PlantVillage/PlantVillage/...).
  2. Jumlah citra per kelas.
  3. File corrupt / tidak bisa dibuka sebagai gambar.
  4. Dimensi citra (min/maks).

Cara pakai (dari root proyek, di VSCode terminal):
    python scripts/verify_dataset.py --data "D:\\Machine Learning\\PlantVillage"

Kode keluar 1 kalau ada masalah fatal (kelas tidak ketemu / nol citra valid).
File corrupt hanya jadi peringatan: daftarnya disimpan di laporan JSON dan
nanti dilewati otomatis oleh prepare_splits.py.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# (nama_kelas_internal, token_yang_wajib_ada_di_nama_folder)
CLASS_RULES = [
    ("bacterial_spot", ("bacterial",)),
    ("healthy", ("healthy",)),
]


def normalize(name: str) -> str:
    """'Pepper,_bell___Bacterial_spot' -> 'pepperbellbacterialspot'."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_class_dir(root: Path, token: str):
    """Cari folder kelas pepper, kembalikan path dan catatan hasil pencarian."""
    hits = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        n = normalize(d.name)
        if "pepper" in n and token in n:
            hits.append(d)
    if not hits:
        return None, "TIDAK KETEMU"
    if len(hits) == 1:
        return hits[0], "ok"

    def n_images(d):
        return sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)

    # Struktur nested ganda bisa menghasilkan beberapa kandidat;
    # pilih yang berisi citra terbanyak.
    hits.sort(key=n_images, reverse=True)
    return hits[0], f"ok (diabaikan {len(hits) - 1} kandidat lain)"


def inspect_images(folder: Path) -> dict:
    files = sorted(
        f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    non_image = sum(
        1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() not in IMAGE_EXTS
    )
    valid, corrupt, sizes = [], [], []
    for f in files:
        try:
            with Image.open(f) as im:
                im.verify()  # cek struktur file tanpa decode penuh
            with Image.open(f) as im:
                im.load()
                if im.mode != "RGB":
                    im = im.convert("RGB")
                sizes.append(im.size)
            valid.append(f.name)
        except Exception as e:  # semua kegagalan buka dianggap corrupt
            corrupt.append({"file": f.name, "error": str(e)[:120]})
    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]
    return {
        "folder": str(folder),
        "total_file_gambar": len(files),
        "file_bukan_gambar": non_image,
        "valid": len(valid),
        "corrupt_n": len(corrupt),
        "corrupt": corrupt,
        "lebar_min": min(widths) if widths else None,
        "lebar_maks": max(widths) if widths else None,
        "tinggi_min": min(heights) if heights else None,
        "tinggi_maks": max(heights) if heights else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Verifikasi dataset PlantVillage lokal.")
    ap.add_argument("--data", required=True, help="Path ke folder dataset (boleh nested).")
    ap.add_argument("--output", default=None, help="Path laporan JSON (opsional).")
    args = ap.parse_args()

    root = Path(args.data).expanduser()
    if not root.is_dir():
        print(f"FATAL: path tidak ada / bukan folder: {root}")
        return 1

    project_root = Path(__file__).resolve().parents[1]
    out_path = Path(args.output) if args.output else project_root / "outputs" / "dataset_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Verifikasi dataset ===")
    print(f"Root : {root}\n")

    report = {
        "waktu": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "kelas": {},
    }
    fatal = []
    for cls_name, (token,) in CLASS_RULES:
        folder, note = find_class_dir(root, token)
        print(f"[{cls_name}] {note}")
        if folder is None:
            fatal.append(f"folder kelas '{cls_name}' tidak ketemu di bawah {root}")
            report["kelas"][cls_name] = {"status": "TIDAK KETEMU"}
            continue
        info = inspect_images(folder)
        info["status"] = "ok" if info["valid"] > 0 else "KOSONG"
        if info["valid"] == 0:
            fatal.append(f"kelas '{cls_name}': nol citra valid di {folder}")
        print(f"  folder : {folder}")
        print(f"  file   : {info['total_file_gambar']} "
              f"(valid {info['valid']}, corrupt {info['corrupt_n']}, "
              f"bukan-gambar {info['file_bukan_gambar']})")
        if info["lebar_min"] is not None:
            print(f"  dimensi: {info['lebar_min']}x{info['tinggi_min']} "
                  f"s/d {info['lebar_maks']}x{info['tinggi_maks']}")
        for c in info["corrupt"][:5]:
            print(f"  ! corrupt: {c['file']} ({c['error']})")
        if info["corrupt_n"] > 5:
            print(f"  ! ... dan {info['corrupt_n'] - 5} file corrupt lainnya (lihat JSON)")
        report["kelas"][cls_name] = info

    report["fatal"] = fatal
    report["kesimpulan"] = "OK" if not fatal else "BERMASALAH"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nLaporan JSON: {out_path}")
    if fatal:
        print("KESIMPULAN: BERMASALAH")
        for m in fatal:
            print(f"  - {m}")
        return 1
    print("KESIMPULAN: OK, dataset siap dipakai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
