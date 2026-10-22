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

# Tema warna diatur di .streamlit/config.toml mengikuti DESIGN.md.
# CSS di sini untuk tipografi, chrome bawaan Streamlit, dan bab tampilan.
GAYA = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;700&family=Playfair+Display:ital,wght@1,500&display=swap");
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stApp {font-family: "Figtree", "Segoe UI", sans-serif;}
h1, h2, h3 {font-weight: 700; letter-spacing: -0.01em;}
.st-key-hero {background: #000000; border-radius: 24px; padding: 2.4rem 2rem;}
.eyebrow {
    font-family: ui-monospace, "Courier New", monospace; font-size: 0.75rem;
    letter-spacing: 0.08em; text-transform: uppercase; color: #FBBF24; margin-bottom: 1rem;
}
.st-key-hero h1 {color: #FFFFFF; font-size: 3.1rem; line-height: 1.1; letter-spacing: -0.02em; margin: 0 0 0.6rem 0;}
.naskah {
    font-family: "Playfair Display", Georgia, serif; font-style: italic; font-weight: 500;
    font-size: 1.9rem; color: #FFFFFF; margin: 0 0 1rem 0;
}
.st-key-hero .isi {color: #9CA3AF; font-size: 0.95rem; line-height: 1.55; max-width: 480px; margin: 0;}
.kartu-gradien {
    background: linear-gradient(135deg, #F97316 0%, #EC4899 55%, #9333EA 100%);
    border-radius: 16px; padding: 1.1rem 0.8rem; box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
}
.judul-bab {font-size: 2.2rem; margin: 3rem 0 0.4rem 0;}
.naskah-inline {font-family: "Playfair Display", Georgia, serif; font-style: italic; font-weight: 500;}
.grid-fitur {display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-top: 1.4rem;}
@media (max-width: 640px) {.grid-fitur {grid-template-columns: 1fr;}}
.fitur {background: #FFFFFF; border: 1.5px dotted #9CA3AF; border-radius: 16px; padding: 1.5rem;}
.judul-fitur {font-weight: 700; font-size: 1.05rem; margin-bottom: 0.4rem;}
.fitur small {display: block; font-size: 0.875rem; line-height: 1.5;}
.bab-langkah {margin-top: 3rem;}
.langkah {display: flex; gap: 1rem; padding: 0.8rem 0; border-top: 1px solid rgba(156, 163, 175, 0.55);}
.langkah-awal {border-top: none;}
.langkah .no {font-size: 1.4rem; font-weight: 700; line-height: 1.2; min-width: 2.2rem;}
.langkah p {margin: 0; font-size: 0.95rem; line-height: 1.55;}
button[kind="primary"] {
    background-color: #F97316; color: #000000; border: none;
    border-radius: 999px; font-weight: 700;
}
button[kind="primary"]:hover {background-color: #F97316; color: #000000; filter: brightness(0.93);}
button[kind="primary"]:focus-visible {
    box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.5); outline: none;
}
</style>
"""

# Cluster smiley warna-warni, motif utama pada DESIGN.md, digambar
# sebagai lingkaran datar dengan wajah garis sederhana.
MARKA_SVG = """
<svg viewBox="0 0 260 190" width="100%" xmlns="http://www.w3.org/2000/svg">
  <circle cx="44" cy="40" r="26" fill="#FBBF24"/>
  <circle cx="178" cy="66" r="40" fill="#2563EB"/>
  <circle cx="200" cy="132" r="34" fill="#EC4899"/>
  <circle cx="130" cy="148" r="30" fill="#22C55E"/>
  <circle cx="88" cy="92" r="52" fill="#F97316"/>
  <g fill="#111111">
    <circle cx="36" cy="36" r="3.4"/><circle cx="52" cy="36" r="3.4"/>
    <circle cx="166" cy="60" r="4.4"/><circle cx="190" cy="60" r="4.4"/>
    <circle cx="189" cy="127" r="3.8"/><circle cx="211" cy="127" r="3.8"/>
    <circle cx="120" cy="144" r="3.4"/><circle cx="140" cy="144" r="3.4"/>
    <circle cx="70" cy="84" r="5.6"/><circle cx="106" cy="84" r="5.6"/>
  </g>
  <g fill="none" stroke="#111111" stroke-linecap="round">
    <path d="M36,46 Q44,53 52,46" stroke-width="3"/>
    <path d="M165,72 Q178,83 191,72" stroke-width="3.6"/>
    <path d="M189,137 Q200,145 211,137" stroke-width="3.2"/>
    <path d="M120,154 Q130,161 140,154" stroke-width="3"/>
    <path d="M69,98 Q88,114 107,98" stroke-width="4.6"/>
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


def tampilkan_tutorial(st):
    for i, langkah in enumerate(LANGKAH_TUTORIAL, start=1):
        st.write(f"{i}. {langkah}")


def tampilkan_welcome(st):
    with st.container(key="hero"):
        kiri, kanan = st.columns([2, 1])
        with kiri:
            st.markdown(
                """
                <div class="eyebrow">Aplikasi Kerja Praktek · ITB STIKOM Bali</div>
                <h1>Deteksi Bercak Daun Bakteri pada Daun Cabai</h1>
                <p class="naskah">Cukup dari satu foto daun.</p>
                <p class="isi">Aplikasi ini memprediksi apakah sehelai daun cabai
                sehat atau terinfeksi bacterial spot, memakai model
                MobileNetV3-Small dengan Coordinate Attention.</p>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Mulai Deteksi", type="primary"):
                st.session_state.mulai = True
                st.rerun()
        with kanan:
            st.markdown(
                f'<div class="kartu-gradien">{MARKA_SVG}</div>',
                unsafe_allow_html=True,
            )

    fitur = [
        ("Deteksi dari Foto",
         "Unggah foto daun atau ambil dari kamera. Hasilnya label sehat "
         "atau bacterial spot beserta tingkat keyakinan model."),
        ("Peta Grad-CAM", "Lihat bagian daun yang menjadi dasar prediksi"),
        ("Unduh Hasil PDF", "Simpan hasil deteksi sebagai dokumen laporan"),
    ]
    kartu = "".join(
        f'<div class="fitur"><div class="judul-fitur">{judul}</div>'
        f"<small>{isi}</small></div>"
        for judul, isi in fitur
    )
    st.markdown(
        f'<h2 class="judul-bab">Tiga Hal yang Bisa Dilakukan</h2>'
        f'<div class="grid-fitur">{kartu}</div>',
        unsafe_allow_html=True,
    )

    langkah_html = "".join(
        f'<div class="langkah{" langkah-awal" if i == 1 else ""}">'
        f'<div class="no">{i}</div><p>{teks}</p></div>'
        for i, teks in enumerate(LANGKAH_TUTORIAL, start=1)
    )
    st.markdown(
        f'<div class="bab-langkah"><h2 class="judul-bab" style="margin-top: 0">'
        f'Cara <span class="naskah-inline">Penggunaan</span></h2>'
        f"{langkah_html}</div>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.caption(
        "Kerja Praktek ITB STIKOM Bali · Akurasi model pada data uji 99,73%"
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

    atas = st.columns([5, 1])
    with atas[0]:
        st.title("Deteksi Penyakit Bercak Daun Bakteri")
    with atas[1]:
        st.write("")
        if st.button("Beranda"):
            st.session_state.mulai = False
            st.rerun()

    with st.expander("Cara Penggunaan"):
        tampilkan_tutorial(st)

    st.divider()

    try:
        model, device = _model()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    kol_kiri, kol_kanan = st.columns(2)

    gambar = None
    gambar_bytes = None
    with kol_kiri:
        st.subheader("Foto Daun")
        sumber = st.radio("Sumber gambar", ["Upload file", "Kamera"], horizontal=True)
        if sumber == "Upload file":
            berkas = st.file_uploader("Pilih foto daun", type=["jpg", "jpeg", "png"])
            if berkas is not None:
                gambar_bytes = berkas.getvalue()
                gambar = Image.open(io.BytesIO(gambar_bytes))
        else:
            foto = st.camera_input("Ambil foto daun")
            if foto is not None:
                gambar_bytes = foto.getvalue()
                gambar = Image.open(io.BytesIO(gambar_bytes))

        if gambar is not None:
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

    with kol_kanan:
        st.subheader("Hasil Deteksi")
        hasil = st.session_state.get("hasil")
        if gambar is None or hasil is None or hasil["gambar_id"] != hash(gambar_bytes):
            st.info(
                "Hasil deteksi tampil di sini setelah foto dipilih "
                "dan tombol Deteksi diklik."
            )
        else:
            pred = hasil["pred"]
            probs = hasil["probs"]
            info = INFO[pred]
            if probs[pred] < AMBANG_KEYAKINAN:
                st.warning(
                    "Keyakinan model rendah, hasil di bawah belum bisa dipegang. "
                    "Coba foto satu helai daun dengan background polos dan "
                    "pencahayaan yang cukup."
                )
            banner = f"Hasil: {info['judul']} · keyakinan {probs[pred] * 100:.1f}%"
            if pred == "healthy":
                st.success(banner)
            else:
                st.error(banner)
            st.write(info["saran"])
            for name in ["bacterial_spot", "healthy"]:
                st.write(f"{INFO[name]['judul']}: {probs[name] * 100:.1f}%")
                st.progress(probs[name])

    # Foto asli dan peta Grad-CAM ditampilkan berdampingan pada baris
    # tersendiri supaya ukurannya sama dan posisinya sejajar.
    if gambar is not None:
        kol_foto, kol_cam = st.columns(2)
        with kol_foto:
            st.image(gambar, caption="Foto yang dipakai", width="stretch")
        with kol_cam:
            if hasil is not None and hasil["gambar_id"] == hash(gambar_bytes):
                st.image(
                    Image.open(io.BytesIO(hasil["overlay_png"])),
                    caption=(
                        "Peta Grad-CAM. Warna merah menunjukkan bagian daun "
                        "yang paling berpengaruh pada prediksi model."
                    ),
                    width="stretch",
                )
                nama_pdf = f"hasil-deteksi-{datetime.now():%Y%m%d-%H%M}.pdf"
                st.download_button(
                    "Unduh Hasil (PDF)",
                    data=hasil["pdf"],
                    file_name=nama_pdf,
                    mime="application/pdf",
                    width="stretch",
                )


if __name__ == "__main__":
    main()
