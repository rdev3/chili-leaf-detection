# Deteksi Penyakit Bercak Daun Bakteri pada Tanaman Cabai
# MobileNetV3-Small + Coordinate Attention (Kerja Praktek)

Proyek ini berisi kode eksperimen untuk laporan KP:
melatih dan membandingkan 4 varian model
(baseline, +SE, +CBAM, +CA) pada dataset PlantVillage pepper,
lalu mengekspor model terbaik dan menampilkannya di aplikasi demo.

Semua keputusan desain eksperimen tercatat di `configs/experiment.yaml`
dan konsisten dengan BAB 3 laporan.

## Prasyarat

- Python 3.10+ (3.11/3.12 oke)
- VSCode + extension Python (Jupyter tidak wajib)
- GPU NVIDIA + driver terbaru (RTX 4060 sudah lebih dari cukup)
- Dataset PlantVillage sudah ada di laptop (tidak perlu download ulang)

## Langkah setup (sekali saja)

Buka folder proyek ini di VSCode, lalu di terminal VSCode (`Ctrl+`` `):

**1. Buat virtual environment**
```
python -m venv .venv
```
Aktifkan:
- Windows: `.venv\Scripts\activate`
- Linux/Mac: `source .venv/bin/activate`

**2. Instal PyTorch versi CUDA** (wajib sebelum requirements lain,
supaya tidak tertimpa versi CPU)
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Cek GPU kedetek:
```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Harus keluar `True` + nama GPU-mu. Kalau `False`, update driver NVIDIA dulu,
lalu ulangi langkah 2.

**3. Instal dependensi lain**
```
pip install -r requirements.txt
```

## Alur kerja

**Langkah 1 — Verifikasi dataset** (wajib pertama kali)
```
python scripts/verify_dataset.py --data "D:\path\ke\PlantVillage"
```
Ganti path dengan lokasi dataset di laptopmu. Skrip ini hanya *membaca*:
tidak mengubah, tidak menghapus file apa pun. Kalau keluar
`KESIMPULAN: OK`, lanjut. Kalau `BERMASALAH`, baca pesannya —
biasanya path salah atau nama folder tidak seperti dugaan.

**Langkah 2 — Split dataset**
```
python scripts/prepare_splits.py --report outputs/dataset_report.json
```
Membuat split 70:15:15 stratified → `outputs/splits/manifest.csv`.
Split memakai seed tetap yang sama untuk semua training (lihat `configs/experiment.yaml`).

**Langkah 3 — Training**
```
python scripts/train.py
```
Melatih 4 varian x 3 seed = 12x training berurutan, dua fase tiap run.
Hasil tiap run tersimpan di `outputs/runs/<varian>/seed<seed>/`:
`best_phase1.pt`, `best_phase2.pt`, `history.json`, `test_metrics.json`.
Run yang sudah selesai otomatis dilewati kalau skrip dijalankan ulang.

Smoke test cepat sebelum full run:
```
python scripts/train.py --variants baseline --seeds 42
```
Pertama kali jalan, torchvision mengunduh bobot pretrained ImageNet (~11 MB).

**Langkah 4 — Rangkuman hasil**
```
python scripts/evaluate.py
```
Membaca seluruh `test_metrics.json`, mencetak tabel mean ± std per varian,
menyimpan `comparison.csv` + `comparison.md` di `outputs/summary/`,
serta confusion matrix tiap varian dari checkpoint terbaiknya.
Juga menandai model terbaik di `best_model.json` untuk konversi TFLite.

**Langkah 5 — Konversi TFLite + benchmark**
```
python scripts/export_tflite.py
```
Mengukur latensi CPU serta ukuran file model. Hasil di
`outputs/export/benchmark.json` — angka ini yang dipakai sebagai bukti
kelayakan mobile di BAB 4.

Konversi TFLite-nya sendiri jalan di Google Colab (ai-edge-torch tidak
punya paket Windows):
1. Buka colab.research.google.com, buat notebook baru.
2. Satu sel: `!pip install -q ai-edge-torch`
3. Upload via panel Files: `colab_convert_tflite.py` dan `best_phase2.pt`
   dari `outputs/runs/ca/seed123/`.
4. Satu sel: `!python colab_convert_tflite.py`
5. File `model_ca.tflite` terdownload otomatis — catat ukurannya.

**Langkah 6 — Aplikasi demo**
```
streamlit run app/streamlit_app.py
```
Buka alamat yang muncul di terminal (biasanya http://localhost:8501).
Unggah foto daun, tekan Deteksi. Memakai checkpoint CA terbaik
(`outputs/runs/ca/seed123/best_phase2.pt`).

## Catatan versi

- v1–v3: verifikasi dataset + split
- v4: training 4 varian x 3 seed
- v5: evaluasi + ringkasan hasil
- v6: konversi TFLite via Colab
- v7: perbaikan nama paket litert-torch
- v8: aplikasi Streamlit + perapihan komentar dan pesan konsol

## Troubleshooting

| Gejala | Solusi |
|---|---|
| `torch.cuda.is_available()` = False | Update driver NVIDIA ke versi terbaru, ulangi langkah 2 |
| `pip` lambat / timeout | Tambahkan `--default-timeout=120` |
| Verifikasi: "folder kelas tidak ketemu" | Cek path `--data`; pastikan di dalamnya ada folder berawalan `Pepper` |
| Out of memory saat training | Kecilkan `batch_size` di `configs/experiment.yaml` |
