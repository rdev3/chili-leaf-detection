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

# CSS seperlunya: sembunyikan menu dan footer bawaan Streamlit,
# sisanya gaya untuk layar pembuka.
GAYA = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.hero {text-align: center; padding: 1.5rem 1rem 0.5rem 1rem;}
.hero .ikon {font-size: 3.2rem;}
.hero h1 {font-size: 2rem; margin: 0.3rem 0;}
.hero p {color: #57606a;}
.fitur {
    background: #f0fdf4; border: 1px solid #86efac; border-radius: 14px;
    padding: 1.1rem 0.8rem; text-align: center; height: 100%;
    color: #14532d; box-shadow: 0 1px 3px rgba(20, 83, 45, 0.15);
}
.lencana {
    width: 52px; height: 52px; margin: 0 auto 0.6rem auto;
    background: #dcfce7; border: 1px solid #86efac; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
}
.judul-fitur {font-weight: 700; margin-bottom: 0.3rem;}
.fitur small {display: block; opacity: 0.85; line-height: 1.4;}
</style>
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

    Simpan aktivasi layer saat forward dan gradiennya saat backward,
    lalu gabungkan keduanya menjadi peta panas satu channel.
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


def buat_pdf(gambar, overlay, pred, probs):
    """Susun laporan hasil deteksi satu halaman. Kembalikan isi PDF sebagai bytes."""
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
    pdf.cell(0, 5, "Kiri: foto yang diunggah. Kanan: peta Grad-CAM, warna merah "
                   "menandakan bagian yang paling berpengaruh pada prediksi.",
             new_x="LMARGIN", new_y="NEXT", align="C")
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
    st.markdown(
        """
        <div class="hero">
            <div class="ikon">🌶️</div>
            <h1>Deteksi Bercak Daun Bakteri pada Daun Cabai</h1>
            <p>Aplikasi ini memprediksi apakah sehelai daun cabai sehat atau
            terinfeksi bacterial spot dari sebuah foto, memakai model
            MobileNetV3-Small dengan Coordinate Attention.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kol = st.columns(3)
    fitur = [
        ("📸", "Deteksi dari Foto", "Unggah foto daun atau ambil dari kamera"),
        ("🔥", "Peta Grad-CAM", "Lihat bagian daun yang menjadi dasar prediksi"),
        ("📄", "Unduh Hasil PDF", "Simpan hasil deteksi sebagai dokumen laporan"),
    ]
    for kolom, (ikon, judul, isi) in zip(kol, fitur):
        with kolom:
            st.markdown(
                f'<div class="fitur"><div class="lencana">{ikon}</div>'
                f'<div class="judul-fitur">{judul}</div>'
                f"<small>{isi}</small></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    tengah = st.columns([1, 2, 1])[1]
    with tengah:
        if st.button("Mulai Deteksi", type="primary", width="stretch"):
            st.session_state.mulai = True
            st.rerun()

    st.write("")
    with st.expander("Cara Penggunaan", expanded=True):
        tampilkan_tutorial(st)

    st.caption(
        "Made Ranggadeva Ratryananda Sandhi · ITB STIKOM Bali"
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
                    pdf_bytes = buat_pdf(gambar, overlay, pred, probs)
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
