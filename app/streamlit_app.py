"""
Aplikasi demo deteksi penyakit bercak daun bakteri pada daun cabai.

Memakai model MobileNetV3-Small + Coordinate Attention

"""

import sys
from pathlib import Path

import torch
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


def main():
    import streamlit as st

    st.set_page_config(page_title="Deteksi Bacterial Spot Daun Cabai", layout="centered")

    @st.cache_resource
    def _model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return load_model(device=device), device

    st.title("Deteksi Penyakit Bercak Daun Bakteri")
    st.write(
        "Ambil atau unggah foto daun cabai. Model MobileNetV3-Small + Coordinate "
        "Attention akan memprediksi apakah daun sehat atau terinfeksi bacterial spot."
    )

    try:
        model, device = _model()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    sumber = st.radio("Sumber gambar", ["Upload file", "Kamera"], horizontal=True)

    gambar = None
    if sumber == "Upload file":
        berkas = st.file_uploader("Pilih foto daun", type=["jpg", "jpeg", "png"])
        if berkas is not None:
            gambar = Image.open(berkas)
    else:
        foto = st.camera_input("Ambil foto daun")
        if foto is not None:
            gambar = Image.open(foto)

    if gambar is None:
        return
    st.image(gambar, caption="Foto yang dipakai", use_container_width=True)

    if st.button("Deteksi"):
        pred, probs = predict_image(model, gambar, device)
        if probs[pred] < AMBANG_KEYAKINAN:
            st.warning(
                "Keyakinan model rendah, hasil di bawah belum bisa dijadikan acuan. "
                "Coba foto dengan background polos dan "
                "pencahayaan yang cukup."
            )
        info = INFO[pred]
        st.subheader(f"Hasil: {info['judul']}")
        st.write(info["saran"])
        for name in ["bacterial_spot", "healthy"]:
            st.write(f"{INFO[name]['judul']}: {probs[name] * 100:.1f}%")
            st.progress(probs[name])


if __name__ == "__main__":
    main()
