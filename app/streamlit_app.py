"""Aplikasi demo deteksi penyakit bercak daun bakteri pada daun cabai.

Memakai model MobileNetV3-Small + Coordinate Attention terbaik
dari hasil training. Dilengkapi layar pembuka, panduan penggunaan,
visualisasi Grad-CAM, dan unduh hasil deteksi sebagai PDF.

Jalankan dari root proyek:
    streamlit run app/streamlit_app.py
"""

import io
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import eval_transform
from src.models import build_model

CHECKPOINT = ROOT / "outputs" / "runs" / "ca" / "seed123" / "best_phase2.pt"
IMAGE_SIZE = 224
CLASS_NAMES = ["healthy", "bacterial_spot"]

# Di bawah ambang ini model dianggap tidak yakin pada jawabannya.
AMBANG_KEYAKINAN = 0.70

# Tautan aplikasi, dicetak sebagai kode QR pada hasil PDF.
URL_APLIKASI = "https://chili-leaf-detection-stikombali.streamlit.app/"

INFO = {
    "healthy": {
        "judul": "Daun Sehat",
        "saran": "Tidak ditemukan gejala bacterial spot. Lanjutkan perawatan rutin.",
    },
    "bacterial_spot": {
        "judul": "Bacterial Spot",
        "saran": (
            "Ditemukan gejala bercak daun bakteri. Pisahkan tanaman yang "
            "terinfeksi agar tidak menular, lalu konsultasikan dengan "
            "penyuluh pertanian setempat."
        ),
    },
}

BULAN = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]

# Tema warna dasar diatur di .streamlit/config.toml. Seluruh gaya lain
# (tipografi, kartu, hasil deteksi) ada di CSS berikut.
GAYA = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;700&family=Playfair+Display:ital,wght@1,500&display=swap");

:root {
    --ink: #11150C;
    --muted: #5F7161;
    --paper: ##EBE8E4;
    --card: #FFFFFF;
    --line: #D9E3C6;
    --green: #2E7D32;
    --green-aksi: #43A047;
    --green-dark: #1E5B2A;
    --green-soft: #E7F3DE;
    --lime: #C1E35C;
    --rust: #A63A1F;
    --rust-soft: #F9E9DE;
}

#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.stApp {background: var(--paper); font-family: "Figtree", "Segoe UI", sans-serif; color: var(--ink);}
.block-container {max-width: 1080px; padding-top: 1.6rem; padding-bottom: 4rem;}
h1, h2, h3 {font-family: "Figtree", "Segoe UI", sans-serif; font-weight: 700; letter-spacing: -0.01em; color: var(--ink);}

/* ---------- Bilah atas ---------- */
.topbar {display: flex; justify-content: space-between; align-items: center; padding: 0.2rem 0 1.2rem 0; gap: 1rem;}
.merek {display: flex; align-items: center; gap: 0.6rem; font-weight: 700; font-size: 1.05rem;}
.merek svg {width: 32px; height: 32px; flex: none;}
.tag {font-size: 0.8rem; color: var(--muted); background: var(--card); border: 1px solid var(--line); border-radius: 999px; padding: 0.3rem 0.85rem; white-space: nowrap;}

/* ---------- Tombol ---------- */
.stApp [data-testid="stBaseButton-primary"], .stApp button[kind="primary"] {
    background: var(--green-aksi); color: #0B1206; border: none; border-radius: 999px;
    font-weight: 700; padding: 0.55rem 1.4rem; transition: filter .12s ease;
}
.stApp [data-testid="stBaseButton-primary"]:hover, .stApp button[kind="primary"]:hover {
    background: var(--green-aksi); color: #0B1206; filter: brightness(0.93);
}
.stApp [data-testid="stBaseButton-secondary"], .stApp [data-testid="stDownloadButton"] button {
    background: var(--card); color: var(--ink); border: 1px solid #C4D2AC; border-radius: 999px;
    font-weight: 600; padding: 0.5rem 1.2rem;
}
.stApp [data-testid="stBaseButton-secondary"]:hover, .stApp [data-testid="stDownloadButton"] button:hover {
    border-color: var(--green); color: var(--green);
}
.stApp [data-testid="stDownloadButton"] button {background: var(--green-dark); color: #FFFFFF; border-color: var(--green-dark);}
.stApp [data-testid="stDownloadButton"] button:hover {background: var(--green); color: #FFFFFF; border-color: var(--green);}
.stApp button:focus-visible {outline: none; box-shadow: 0 0 0 3px rgba(67, 160, 71, 0.5);}

/* ---------- Hero ---------- */
.st-key-hero {background: #0C0F0A; border-radius: 24px; padding: 2.4rem 2.2rem;}
.eyebrow {
    font-family: ui-monospace, "Courier New", monospace; font-size: 0.75rem;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--lime); margin-bottom: 1rem;
}
.st-key-hero h1 {color: #FFFFFF; font-size: clamp(2.1rem, 4.2vw, 3.1rem); line-height: 1.1; margin: 0 0 0.6rem 0; padding: 0;}
.naskah {
    font-family: "Playfair Display", Georgia, serif; font-style: italic; font-weight: 500;
    font-size: 1.9rem; color: #DCE9C8; margin: 0 0 1rem 0;
}
.hero-sub {color: #A9B8A4; font-size: 1rem; line-height: 1.6; max-width: 30rem; margin: 0;}
.hero-art {display: flex; justify-content: center;}
.hero-art svg {width: 100%; max-width: 260px; height: auto;}

/* ---------- Angka ringkas, fitur, langkah ---------- */
.stat-grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1.2rem;}
.stat {background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 1.1rem 1.3rem;}
.stat b {display: block; font-size: 1.9rem; font-weight: 700; color: var(--green-dark); line-height: 1.2;}
.stat span {color: var(--muted); font-size: 0.88rem;}
.bab-label {font-size: 0.78rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--green); margin: 3.2rem 0 0.4rem 0;}
.judul-bab {font-size: 2rem; margin: 0 0 0.2rem 0; padding: 0;}
.naskah-inline {font-family: "Playfair Display", Georgia, serif; font-style: italic; font-weight: 500;}
.grid-fitur {display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1.2rem;}
.fitur {background: var(--card); border: 1.5px dotted #93AC7E; border-radius: 16px; padding: 1.5rem;}
.no-fitur {font-family: ui-monospace, "Courier New", monospace; font-size: 0.8rem; letter-spacing: 0.1em; color: var(--green); margin-bottom: 0.8rem;}
.judul-fitur {font-weight: 700; font-size: 1.05rem; margin-bottom: 0.35rem;}
.fitur small {display: block; font-size: 0.9rem; line-height: 1.55; color: var(--muted);}
.langkah-grid {column-count: 2; column-gap: 2.2rem; margin-top: 1.2rem;}
.langkah {display: flex; gap: 0.9rem; align-items: flex-start; break-inside: avoid; margin-bottom: 0.9rem;}
.langkah .no {
    flex: none; width: 2rem; height: 2rem; border-radius: 50%; background: var(--green-soft);
    color: var(--green-dark); font-weight: 700; font-size: 0.9rem; display: grid; place-items: center;
}
.langkah p {margin: 0.2rem 0 0 0; font-size: 0.95rem; line-height: 1.55;}
.st-key-cta {background: var(--card); border: 1px solid var(--line); border-radius: 24px; padding: 2.2rem 1.5rem 1.8rem 1.5rem; margin-top: 3rem; text-align: center;}
.st-key-cta h2 {font-size: 1.7rem; margin: 0 0 0.3rem 0; padding: 0;}
.st-key-cta [data-testid="stMarkdownContainer"] {text-align: center;}
.st-key-cta .cta-teks {color: var(--muted); margin: 0 0 1.2rem 0;}

/* ---------- Halaman deteksi ---------- */
.judul-halaman {font-size: 2.3rem; font-weight: 700; line-height: 1.15; margin: 0.2rem 0 0.3rem 0; letter-spacing: -0.01em;}
.sub-halaman {color: var(--muted); font-size: 1rem; margin: 0 0 1rem 0;}
.stepper {display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.4rem 0 1.4rem 0;}
.step {display: flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.9rem 0.4rem 0.45rem; border-radius: 999px; background: var(--card); border: 1px solid var(--line); color: var(--muted); font-size: 0.85rem; font-weight: 600;}
.step i {font-style: normal; width: 1.45rem; height: 1.45rem; border-radius: 50%; display: grid; place-items: center; background: #E4EBD2; font-size: 0.75rem;}
.step.aktif {border-color: var(--green); background: var(--green-soft); color: var(--green-dark);}
.step.aktif i, .step.selesai i {background: var(--green); color: #FFFFFF;}
.step.selesai {color: var(--green-dark);}

.st-key-panel-input, .st-key-panel-hasil, .st-key-panel-cam {
    background: var(--card); border: 1px solid var(--line); border-radius: 22px; padding: 1.4rem 1.5rem;
}
.st-key-panel-cam {margin-top: 1.2rem;}
.judul-panel {font-weight: 700; font-size: 1.05rem; margin: 0 0 0.9rem 0; display: flex; align-items: center; gap: 0.55rem;}
.judul-panel .bulat {width: 1.5rem; height: 1.5rem; border-radius: 50%; background: var(--green); color: #FFF; display: grid; place-items: center; font-size: 0.78rem;}
.st-key-panel-input [data-testid="stFileUploaderDropzone"] {
    background: #F4F8EC; border: 1.5px dashed #A9BC8E; border-radius: 16px; padding: 1.6rem 1rem;
}
.st-key-panel-input [data-testid="stFileUploaderDropzone"]:hover {border-color: var(--green);}
.st-key-panel-input [data-testid="stImage"] img, .st-key-panel-cam [data-testid="stImage"] img {border-radius: 14px;}
.st-key-panel-input [data-testid="stImage"] img {max-height: 300px; object-fit: contain; background: #F4F8EC;}
.info-berkas {display: flex; justify-content: space-between; gap: 0.6rem; font-size: 0.82rem; color: var(--muted); margin: 0.2rem 0 0.9rem 0;}

.kosong {
    border: 1.5px dashed #C9D4B2; border-radius: 18px; padding: 1.5rem 1.2rem;
    text-align: center; color: var(--muted); font-size: 0.95rem; line-height: 1.55;
}
.kosong b {display: block; color: var(--ink); font-size: 1.02rem; margin-bottom: 0.2rem;}

.hasil {border-radius: 18px; padding: 1.3rem 1.4rem;}
.hasil-sehat {background: var(--green-soft); border: 1px solid #BFE0AE;}
.hasil-sakit {background: var(--rust-soft); border: 1px solid #EFC4A8;}
.lencana {display: inline-block; font-weight: 700; font-size: 0.78rem; letter-spacing: 0.05em; text-transform: uppercase; padding: 0.28rem 0.75rem; border-radius: 999px; color: #FFF;}
.hasil-sehat .lencana {background: var(--green);}
.hasil-sakit .lencana {background: var(--rust);}
.nama-hasil {font-size: 1.8rem; font-weight: 700; margin: 0.7rem 0 0.1rem 0; line-height: 1.15;}
.keyakinan {font-size: 0.92rem; color: var(--muted);}
.keyakinan b {color: var(--ink); font-size: 1.05rem;}
.saran {margin: 0.9rem 0 0 0; font-size: 0.93rem; line-height: 1.55;}
.garis-hasil {border-top: 1px solid rgba(20, 32, 26, 0.12); margin-top: 1rem;}
.peringatan {background: #FFF6DD; border: 1px solid #F1D58A; color: #6B4A00; border-radius: 14px; padding: 0.8rem 1rem; font-size: 0.9rem; line-height: 1.5; margin-bottom: 0.9rem;}
.peringatan b {display: block; margin-bottom: 0.1rem;}
.bar {margin-top: 0.75rem;}
.bar-atas {display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600;}
.track {height: 8px; background: rgba(255, 255, 255, 0.65); border-radius: 999px; overflow: hidden; margin-top: 0.35rem;}
.isi-bar {height: 100%; border-radius: 999px;}
.isi-sakit {background: var(--rust);}
.isi-sehat {background: var(--green);}

.st-key-panel-cam .teks-cam {font-size: 0.93rem; line-height: 1.6; color: var(--muted);}
.legenda {height: 12px; border-radius: 999px; background: linear-gradient(90deg, #1D2BB8, #1FB6D6, #5BD16A, #F3E23B, #F58A1F, #D9261C); margin: 1rem 0 0.35rem 0;}
.legenda-label {display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--muted); margin-bottom: 1.2rem;}
.catatan-kaki {text-align: center; color: var(--muted); font-size: 0.82rem; margin-top: 2.2rem; line-height: 1.6;}

[data-testid="stExpander"] {background: var(--card); border: 1px solid var(--line); border-radius: 16px;}
[data-testid="stExpander"] details {border: none;}

@media (max-width: 760px) {
    .st-key-hero {padding: 2rem 1.4rem;}
    .stat-grid, .grid-fitur {grid-template-columns: 1fr;}
    .langkah-grid {column-count: 1;}
    .judul-halaman {font-size: 1.8rem;}
    .tag {display: none;}
}
</style>
"""

# Ikon daun dengan bingkai pemindai, dipakai di hero dan merek.
LOGO_SVG = """
<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <rect width="32" height="32" rx="9" fill="#2E7D32"/>
  <path d="M16 6c6 2.6 8.6 9 0 20C7.400 15 10 8.600 16 6Z" fill="#DCE9A8"/>
  <path d="M16 9v16" stroke="#2E7D32" stroke-width="1.4" stroke-linecap="round"/>
</svg>
"""

# Ilustrasi daun dengan bingkai pemindai untuk hero, mengikuti
# palet daun. Digambar pada kanvas persegi dan ditampilkan dengan
# lebar maksimal supaya proporsinya tidak tertarik.
DAUN_SVG = """
<svg viewBox="0 0 320 320" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Ilustrasi daun cabai dengan bingkai pemindai">
  <circle cx="160" cy="160" r="140" fill="#FFFFFF" fill-opacity="0.05"/>
  <path d="M160 36C246 76 280 172 160 288C40 172 74 76 160 36Z" fill="#43A047"/>
  <path d="M160 36C246 76 280 172 160 288Z" fill="#2E7D32"/>
  <g stroke="#1E5B2A" stroke-opacity="0.55" stroke-width="3" stroke-linecap="round" fill="none">
    <path d="M160 50V286"/>
    <path d="M160 100L118 78M160 100L202 78"/>
    <path d="M160 146L100 118M160 146L220 118"/>
    <path d="M160 192L108 160M160 192L212 160"/>
    <path d="M160 236L128 206M160 236L192 206"/>
  </g>
  <g fill="#7A3B14" fill-opacity="0.9">
    <circle cx="124" cy="134" r="9"/><circle cx="206" cy="172" r="11"/>
    <circle cx="140" cy="206" r="7"/><circle cx="188" cy="112" r="6"/>
  </g>
  <g stroke="#C1E35C" stroke-width="5" stroke-linecap="round" fill="none">
    <path d="M52 92V58H86"/><path d="M268 92V58H234"/>
    <path d="M52 228V262H86"/><path d="M268 228V262H234"/>
  </g>
</svg>
"""
LANGKAH_TUTORIAL = [
    "Siapkan foto satu helai daun cabai dengan background polos dan pencahayaan yang cukup.",
    "Pilih sumber gambar, unggah file foto atau ambil langsung dari kamera.",
    "Klik tombol Deteksi dan tunggu sampai hasil tampil.",
    "Baca hasilnya: label sehat atau bacterial spot, tingkat keyakinan model, "
    "dan peta Grad-CAM yang menunjukkan bagian daun yang paling diperhatikan model.",
    "Jika keyakinan model di bawah 70%, ambil ulang foto dengan background yang lebih polos.",
    "Hasil deteksi dapat disimpan lewat tombol Unduh Hasil PDF.",
]


def tanggal_hari_ini():
    now = datetime.now()
    return f"{now.day} {BULAN[now.month - 1]} {now.year}"


def load_model(checkpoint=CHECKPOINT, variant="ca", device=None):
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(
            f"Checkpoint tidak ketemu: {checkpoint}. Jalankan train.py sampai selesai dulu."
        )
    model, _ = build_model(variant, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    if device is not None:
        model.to(device)
    model.eval()
    return model


def predict_image(model, image, device):
    transform = eval_transform(IMAGE_SIZE)
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()
    hasil = {name: probs[i] for i, name in enumerate(CLASS_NAMES)}
    pred = max(hasil, key=hasil.get)
    return pred, hasil


class GradCAM:
    """Grad-CAM pada satu layer target.

    Aktivasi layer saat forward digabung dengan gradiennya saat
    backward menjadi peta panas satu channel.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.aktivasi = None
        self.gradien = None
        self._hook1 = target_layer.register_forward_hook(self._simpan_aktivasi)
        self._hook2 = target_layer.register_full_backward_hook(self._simpan_gradien)

    def _simpan_aktivasi(self, module, masukan, keluaran):
        self.aktivasi = keluaran.detach()

    def _simpan_gradien(self, module, grad_masuk, grad_keluar):
        self.gradien = grad_keluar[0].detach()

    def hitung(self, x, class_idx):
        self.model.zero_grad()
        skor = self.model(x)[0, class_idx]
        skor.backward()
        # Bobot tiap channel adalah rata-rata gradien di seluruh posisi spasial.
        bobot = self.gradien.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((bobot * self.aktivasi).sum(dim=1))[0]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()

    def lepas(self):
        self._hook1.remove()
        self._hook2.remove()


def gradcam_overlay(model, image, device, class_idx):
    """Buat gambar overlay Grad-CAM seukuran foto asli."""
    transform = eval_transform(IMAGE_SIZE)
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    # Layer spasial terakhir sebelum classifier head adalah keluaran
    # modul Coordinate Attention, jadi di situlah peta panas dihitung.
    alat = GradCAM(model, model[1][0])
    try:
        cam = alat.hitung(x, class_idx)
    finally:
        alat.lepas()

    asli = image.convert("RGB")
    cam_t = torch.from_numpy(cam).reshape(1, 1, *cam.shape)
    cam_t = F.interpolate(
        cam_t, size=(asli.height, asli.width), mode="bilinear", align_corners=False
    )
    cam_besar = cam_t[0, 0].numpy()

    import matplotlib

    warna = matplotlib.colormaps["jet"](cam_besar)[..., :3]
    panas = Image.fromarray((warna * 255).astype(np.uint8))
    return Image.blend(asli, panas, alpha=0.45)


def buat_qr(url):
    """Buat gambar kode QR berisi url, dikembalikan sebagai bytes PNG."""
    import qrcode

    qr = qrcode.QRCode(box_size=10, border=1)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def buat_pdf(gambar, overlay, pred, probs, nama_file):
    """Susun laporan hasil deteksi. Kembalikan isi PDF sebagai bytes."""
    from fpdf import FPDF

    info = INFO[pred]
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("helvetica", "B", 15)
    pdf.cell(0, 9, "Hasil Deteksi Penyakit Daun Cabai",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 6, "Model MobileNetV3-Small + Coordinate Attention",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 6, f"Tanggal pengujian: {tanggal_hari_ini()}",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 6, f"Nama file: {nama_file}",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    # Foto asli dan overlay Grad-CAM berdampingan. Ukurannya dikecilkan
    # dulu supaya file PDF tidak membengkak oleh foto resolusi penuh.
    with tempfile.TemporaryDirectory() as tmp:
        path_asli = Path(tmp) / "asli.jpg"
        path_cam = Path(tmp) / "cam.jpg"
        foto_kecil = gambar.convert("RGB")
        foto_kecil.thumbnail((900, 900))
        foto_kecil.save(path_asli, quality=85)
        cam_kecil = overlay.copy()
        cam_kecil.thumbnail((900, 900))
        cam_kecil.save(path_cam, quality=85)
        lebar = 86
        tinggi = max(
            lebar * foto_kecil.height / foto_kecil.width,
            lebar * cam_kecil.height / cam_kecil.width,
        )
        y0 = pdf.get_y()
        pdf.image(str(path_asli), x=10, y=y0, w=lebar)
        pdf.image(str(path_cam), x=110, y=y0, w=lebar)
        pdf.set_y(y0 + tinggi + 2)

    pdf.set_font("helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, "Kiri: foto yang diunggah. Kanan: peta Grad-CAM, warna merah "
                         "menandakan bagian yang paling berpengaruh pada prediksi. "
                         "Peta ini menunjukkan bagian citra yang menjadi fokus model, "
                         "bukan batas pasti bercak penyakit.",
                   align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 8, f"Hasil Prediksi: {info['judul']}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 6, f"Keyakinan model: {probs[pred] * 100:.1f}%",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Probabilitas Bacterial Spot: {probs['bacterial_spot'] * 100:.1f}%",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Probabilitas Daun Sehat: {probs['healthy'] * 100:.1f}%",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.multi_cell(0, 6, f"Saran: {info['saran']}")
    if probs[pred] < AMBANG_KEYAKINAN:
        pdf.ln(2)
        pdf.multi_cell(
            0, 6,
            "Catatan: keyakinan model berada di bawah 70% sehingga hasil ini "
            "belum bisa dipegang. Ambil ulang foto satu helai daun dengan "
            "background polos dan pencahayaan yang cukup.",
        )

    pdf.ln(4)
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 8, "Tentang Model", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(
        0, 6,
        "Model yang dipakai adalah MobileNetV3-Small dengan Coordinate "
        "Attention, dilatih pada dataset PlantVillage sebanyak 2.475 "
        "citra dari dua kelas, yaitu bacterial spot dan sehat. Pada data "
        "uji, model mencapai akurasi 99,73%.",
    )
    pdf.ln(2)
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 8, "Catatan", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(
        0, 6,
        "Hasil pada dokumen ini adalah prediksi model dan bukan "
        "diagnosis pasti. Model hanya membedakan dua kelas, yaitu sehat "
        "dan bacterial spot, sehingga penyakit atau kerusakan lain di "
        "luar dua kelas tersebut dapat terprediksi sebagai salah "
        "satunya. Gunakan hasil ini sebagai bahan pertimbangan awal "
        "dan konsultasikan dengan penyuluh pertanian untuk kepastian.",
    )

    # Kode QR berisi tautan aplikasi, ditaruh di kanan bawah isi dokumen.
    y_qr = pdf.get_y() + 6
    if y_qr + 34 > 282:
        pdf.add_page()
        y_qr = pdf.get_y()
    with tempfile.TemporaryDirectory() as tmp:
        path_qr = Path(tmp) / "qr.png"
        path_qr.write_bytes(buat_qr(URL_APLIKASI))
        pdf.image(str(path_qr), x=168, y=y_qr, w=28)
    pdf.set_xy(154, y_qr + 29)
    pdf.set_font("helvetica", "", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(56, 5, "Pindai untuk membuka aplikasi", align="C")
    pdf.set_text_color(0, 0, 0)

    pdf.set_y(-22)
    pdf.set_font("helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "Dokumen ini dibuat otomatis oleh aplikasi Deteksi Bercak Daun "
                   "Bakteri pada Daun Cabai, Kerja Praktek ITB STIKOM Bali.",
             align="C")
    return bytes(pdf.output())


# ----------------------------------------------------------------------
# Tampilan (UI). Bagian di bawah hanya mengatur layout dan gaya,
# logika model, Grad-CAM, dan PDF ada di fungsi-fungsi di atas.
# ----------------------------------------------------------------------

def html_langkah():
    return '<div class="langkah-grid">' + "".join(
        f'<div class="langkah"><div class="no">{i}</div><p>{teks}</p></div>'
        for i, teks in enumerate(LANGKAH_TUTORIAL, start=1)
    ) + "</div>"


def tampilkan_tutorial(st):
    st.markdown(html_langkah(), unsafe_allow_html=True)


def html_bilah_atas(dengan_tag=True):
    tag = '<div class="tag">Made Ranggadeva Ratyrananda Sandhi · ITB STIKOM Bali</div>' if dengan_tag else ""
    return (
        f'<div class="topbar"><div class="merek">{LOGO_SVG}'
        f"<span>Deteksi Daun Cabai</span></div>{tag}</div>"
    )


def html_stepper(tahap):
    """tahap: 1 = pilih foto, 2 = siap deteksi, 3 = hasil tampil."""
    nama = ["Pilih foto", "Klik Deteksi", "Baca hasil"]
    potongan = []
    for i, teks in enumerate(nama, start=1):
        if i < tahap or tahap == 3:
            kelas, tanda = "selesai", "✓"
        elif i == tahap:
            kelas, tanda = "aktif", str(i)
        else:
            kelas, tanda = "", str(i)
        potongan.append(f'<div class="step {kelas}"><i>{tanda}</i>{teks}</div>')
    return '<div class="stepper">' + "".join(potongan) + "</div>"


def html_hasil(pred, probs):
    info = INFO[pred]
    kelas = "hasil-sehat" if pred == "healthy" else "hasil-sakit"
    lencana = "Tidak ada gejala" if pred == "healthy" else "Terdeteksi gejala"
    peringatan = ""
    if probs[pred] < AMBANG_KEYAKINAN:
        peringatan = (
            '<div class="peringatan"><b>Keyakinan model rendah</b>'
            "Hasil di bawah belum bisa dipegang. Coba foto satu helai daun "
            "dengan background polos dan pencahayaan yang cukup.</div>"
        )
    bar = ""
    for nama, kelas_bar in [("bacterial_spot", "isi-sakit"), ("healthy", "isi-sehat")]:
        persen = probs[nama] * 100
        # Persen sangat kecil tetap digambar sebagai ruas pendek yang
        # terbaca sebagai data, bukan garis sisa di ujung track.
        lebar = persen if persen == 0 else max(persen, 1.5)
        bar += (
            f'<div class="bar"><div class="bar-atas"><span>{INFO[nama]["judul"]}</span>'
            f"<span>{persen:.1f}%</span></div>"
            f'<div class="track"><div class="isi-bar {kelas_bar}" '
            f'style="width: {lebar:.1f}%"></div></div></div>'
        )
    return (
        f"{peringatan}"
        f'<div class="hasil {kelas}"><span class="lencana">{lencana}</span>'
        f'<div class="nama-hasil">{info["judul"]}</div>'
        f'<div class="keyakinan">Keyakinan model <b>{probs[pred] * 100:.1f}%</b></div>'
        f'<p class="saran">{info["saran"]}</p>'
        f'<div class="garis-hasil"></div>{bar}</div>'
    )


def tampilkan_welcome(st):
    st.markdown(html_bilah_atas(), unsafe_allow_html=True)

    with st.container(key="hero"):
        kiri, kanan = st.columns([3, 2], vertical_alignment="center")
        with kiri:
            st.markdown(
                """
                <div class="eyebrow">Aplikasi Kerja Praktek \u00b7 ITB STIKOM Bali</div>
                <h1>Deteksi Bercak Daun Bakteri pada Daun Cabai</h1>
                <p class="naskah">Cukup dari satu foto daun.</p>
                <p class="hero-sub">Aplikasi ini memprediksi apakah sehelai daun cabai
                sehat atau terinfeksi bacterial spot dari sebuah foto, memakai model
                MobileNetV3-Small dengan Coordinate Attention, lengkap dengan
                peta Grad-CAM yang menunjukkan dasar keputusannya.</p>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Mulai Deteksi", type="primary", key="mulai_atas"):
                st.session_state.mulai = True
                st.rerun()
        with kanan:
            st.markdown(f'<div class="hero-art">{DAUN_SVG}</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="stat-grid">'
        '<div class="stat"><b>99,73%</b><span>Akurasi pada data uji</span></div>'
        '<div class="stat"><b>2.475</b><span>Citra daun dari PlantVillage</span></div>'
        '<div class="stat"><b>2 kelas</b><span>Sehat dan bacterial spot</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    fitur = [
        ("01", "Deteksi dari foto",
         "Unggah foto daun atau ambil dari kamera. Hasilnya label sehat "
         "atau bacterial spot beserta tingkat keyakinan model."),
        ("02", "Peta Grad-CAM",
         "Lihat bagian daun yang menjadi dasar prediksi model, ditandai "
         "dengan warna pada foto."),
        ("03", "Unduh hasil PDF",
         "Simpan hasil deteksi sebagai dokumen laporan lengkap dengan "
         "foto, peta Grad-CAM, dan saran."),
    ]
    kartu = "".join(
        f'<div class="fitur"><div class="no-fitur">{no}</div>'
        f'<div class="judul-fitur">{judul}</div><small>{isi}</small></div>'
        for no, judul, isi in fitur
    )
    st.markdown(
        '<div class="bab-label">Fitur</div>'
        '<h2 class="judul-bab">Tiga hal yang bisa dilakukan</h2>'
        f'<div class="grid-fitur">{kartu}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="bab-label">Panduan</div>'
        '<h2 class="judul-bab">Cara <span class="naskah-inline">penggunaan</span></h2>' + html_langkah(),
        unsafe_allow_html=True,
    )

    with st.container(key="cta"):
        st.markdown(
            "<h2>Siap mencoba?</h2><p class=\"cta-teks\">Siapkan satu foto daun cabai, "
            "hasilnya keluar dalam hitungan detik.</p>",
            unsafe_allow_html=True,
        )
        _, tengah, _ = st.columns([2, 1.2, 2])
        with tengah:
            if st.button("Mulai Deteksi", type="primary", key="mulai_bawah",
                         width="stretch"):
                st.session_state.mulai = True
                st.rerun()

    st.markdown(
        '<div class="catatan-kaki">Kerja Praktek ITB STIKOM Bali · '
        "Hasil adalah prediksi model, bukan diagnosis pasti.</div>",
        unsafe_allow_html=True,
    )


def main():
    import streamlit as st

    st.set_page_config(
        page_title="Deteksi Bacterial Spot Daun Cabai",
        page_icon="🌶️",
        layout="centered",
    )
    st.markdown(GAYA, unsafe_allow_html=True)

    if "mulai" not in st.session_state:
        st.session_state.mulai = False

    if not st.session_state.mulai:
        tampilkan_welcome(st)
        return

    @st.cache_resource
    def _model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return load_model(device=device), device

    atas = st.columns([6, 1.4], vertical_alignment="center")
    with atas[0]:
        st.markdown(html_bilah_atas(dengan_tag=False), unsafe_allow_html=True)
    with atas[1]:
        if st.button("← Beranda", width="stretch"):
            st.session_state.mulai = False
            st.rerun()

    st.markdown(
        '<h1 class="judul-halaman">Periksa daun cabai</h1>'
        '<p class="sub-halaman">Pilih foto, klik Deteksi, lalu baca hasilnya.</p>',
        unsafe_allow_html=True,
    )
    tempat_stepper = st.empty()

    with st.expander("Cara penggunaan"):
        tampilkan_tutorial(st)

    try:
        model, device = _model()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    st.write("")
    kol_kiri, kol_kanan = st.columns(2, gap="medium")

    gambar = None
    gambar_bytes = None
    berkas = None
    with kol_kiri:
        with st.container(key="panel-input"):
            st.markdown(
                '<div class="judul-panel"><span class="bulat">1</span>Foto daun</div>',
                unsafe_allow_html=True,
            )
            pilihan = ["Upload file", "Kamera"]
            if hasattr(st, "segmented_control"):
                sumber = st.segmented_control(
                    "Sumber gambar", pilihan, default=pilihan[0],
                    label_visibility="collapsed",
                ) or pilihan[0]
            else:
                sumber = st.radio("Sumber gambar", pilihan, horizontal=True)
            if sumber == "Upload file":
                berkas = st.file_uploader(
                    "Pilih foto daun", type=["jpg", "jpeg", "png"],
                    label_visibility="collapsed",
                )
                if berkas is not None:
                    gambar_bytes = berkas.getvalue()
                    gambar = Image.open(io.BytesIO(gambar_bytes))
            else:
                foto = st.camera_input("Ambil foto daun", label_visibility="collapsed")
                if foto is not None:
                    gambar_bytes = foto.getvalue()
                    gambar = Image.open(io.BytesIO(gambar_bytes))

            if gambar is not None:
                if sumber == "Upload file":
                    st.image(gambar, width="stretch")
                nama_tampil = berkas.name if sumber == "Upload file" else "Foto dari kamera"
                st.markdown(
                    f'<div class="info-berkas"><span>{nama_tampil}</span>'
                    f"<span>{gambar.width} × {gambar.height} px</span></div>",
                    unsafe_allow_html=True,
                )
                if st.button("Deteksi", type="primary", width="stretch"):
                    with st.spinner("Menganalisis foto..."):
                        pred, probs = predict_image(model, gambar, device)
                        overlay = gradcam_overlay(
                            model, gambar, device, CLASS_NAMES.index(pred)
                        )
                        nama_file = (
                            berkas.name if sumber == "Upload file"
                            else "Foto dari kamera"
                        )
                        pdf_bytes = buat_pdf(gambar, overlay, pred, probs, nama_file)
                    buf = io.BytesIO()
                    overlay.save(buf, format="PNG")
                    st.session_state.hasil = {
                        "gambar_id": hash(gambar_bytes),
                        "pred": pred,
                        "probs": probs,
                        "overlay_png": buf.getvalue(),
                        "pdf": pdf_bytes,
                    }

    hasil = st.session_state.get("hasil")
    hasil_valid = (
        gambar is not None and hasil is not None
        and hasil["gambar_id"] == hash(gambar_bytes)
    )

    with kol_kanan:
        with st.container(key="panel-hasil"):
            st.markdown(
                '<div class="judul-panel"><span class="bulat">2</span>Hasil deteksi</div>',
                unsafe_allow_html=True,
            )
            if not hasil_valid:
                st.markdown(
                    '<div class="kosong"><b>Belum ada hasil</b>'
                    "Hasil deteksi tampil di sini setelah foto dipilih "
                    "dan tombol Deteksi diklik.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    html_hasil(hasil["pred"], hasil["probs"]),
                    unsafe_allow_html=True,
                )

    # Foto asli dan peta Grad-CAM berdampingan dengan ukuran sama supaya
    # mudah dibandingkan, penjelasan dan unduh PDF di bawahnya.
    if hasil_valid:
        with st.container(key="panel-cam"):
            st.markdown(
                '<div class="judul-panel"><span class="bulat">3</span>Peta Grad-CAM</div>',
                unsafe_allow_html=True,
            )
            kol_foto, kol_cam = st.columns(2, gap="medium")
            with kol_foto:
                st.image(gambar, caption="Foto yang dipakai", width="stretch")
            with kol_cam:
                st.image(
                    Image.open(io.BytesIO(hasil["overlay_png"])),
                    caption="Peta Grad-CAM pada foto yang sama",
                    width="stretch",
                )
            st.markdown(
                "<p class=\"teks-cam\">Warna menunjukkan bagian daun yang paling berpengaruh pada "
                "prediksi model. Peta ini adalah fokus model, bukan batas pasti "
                "bercak penyakit.</p>"
                '<div class="legenda"></div>'
                '<div class="legenda-label"><span>Kurang berpengaruh</span>'
                "<span>Paling berpengaruh</span></div>",
                unsafe_allow_html=True,
            )
            nama_pdf = f"hasil-deteksi-{datetime.now():%Y%m%d-%H%M}.pdf"
            st.download_button(
                "Unduh Hasil (PDF)",
                data=hasil["pdf"],
                file_name=nama_pdf,
                mime="application/pdf",
                width="stretch",
            )

    tahap = 3 if hasil_valid else (2 if gambar is not None else 1)
    tempat_stepper.markdown(html_stepper(tahap), unsafe_allow_html=True)

    st.markdown(
        '<div class="catatan-kaki">Hasil adalah prediksi model, bukan diagnosis '
        "pasti. Model hanya membedakan daun sehat dan bacterial spot. "
        "Konsultasikan dengan penyuluh pertanian untuk kepastian.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
