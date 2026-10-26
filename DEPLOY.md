# Deploy ke Streamlit Community Cloud

Panduan sekali jalan: push proyek ke GitHub, lalu deploy gratis di
Streamlit Community Cloud sehingga aplikasi bisa dibuka dari HP atau
dibagikan ke pembimbing lewat URL publik.

## 1. Pastikan git terinstal

```
git --version
```

Kalau tidak dikenal, instal Git for Windows dari git-scm.com
(ikuti default saja), lalu tutup dan buka ulang terminal.

## 2. Siapkan repo lokal

Dari folder proyek (`chili-leaf-detection`):

```
git init
git add -A
git add -f outputs/runs/ca/seed123/best_phase2.pt
git commit -m "Aplikasi demo deteksi bacterial spot + model"
git branch -M main
```

Baris `git add -f` itu penting: file `.pt` sengaja di-ignore oleh
`.gitignore`, jadi checkpoint model harus ditambahkan paksa.
Hanya file itu yang ikut; dataset dan hasil training lain tidak.

## 3. Buat repo di GitHub

1. Buka github.com → New repository.
2. Nama mis. `chili-leaf-detection`, pilih **Public**
   (Community Cloud gratis hanya bisa membaca repo publik).
3. **Jangan** centang "Add a README" supaya tidak konflik.
4. Create repository, salin URL-nya.

## 4. Push

```
git remote add origin https://github.com/USERNAME/chili-leaf-detection.git
git push -u origin main
```

Ganti USERNAME dengan username GitHub-mu. Saat diminta login,
GitHub tidak menerima password biasa — pakai Personal Access Token
(github.com → Settings → Developer settings → Personal access tokens).

## 5. Deploy di Streamlit Cloud

1. Buka share.streamlit.io, login dengan GitHub.
2. New app → pilih repo `chili-leaf-detection`.
3. Main file path: `app/streamlit_app.py`
4. Klik "Advanced settings" → pilih Python 3.11 (wajib — file
   runtime.txt TIDAK dibaca oleh Community Cloud) → Deploy.
   Instalasi pertama makan waktu beberapa menit
   (torch CPU ~200 MB). Setelah jadi, dapat URL publik.

## Mengaktifkan fitur Tanya AI (opsional)

Fitur ini meminta penjelasan umum dari Gemini berdasarkan label prediksi
model saja; foto tidak dikirim. Tanpa kunci, aplikasi tetap jalan normal
dan bagian Tanya AI hanya menampilkan pesan belum dikonfigurasi.

1. Buat kunci di Google AI Studio → Get API key.
2. Di Streamlit Cloud: buka aplikasi → Manage app → Settings → Secrets,
   lalu tambahkan:
   ```
   GEMINI_API_KEY = "kunci-dari-AI-Studio"
   ```
   Baris `GEMINI_MODEL` opsional dan boleh tidak diisi: kode memakai
   default `gemini-3.8-flash` (Flash stabil untuk proyek baru, dicek
   26 Okt 2026). Kalau pemanggilan mengembalikan 404 di masa depan,
   tambahkan baris `GEMINI_MODEL = "id-model-flash-yang-aktif"` dengan
   ID yang aktif saat itu, karena ID model berubah dari waktu ke waktu.
3. Untuk uji lokal, buat file `.streamlit/secrets.toml` dengan isi yang
   sama. Berkas ini sudah masuk `.gitignore`, jadi tidak ikut ter-commit.
4. Jangan pernah menulis kunci di kode, di file `.env` yang ikut
   ter-commit, atau mengirimnya lewat chat.

## Catatan

- Repo publik berarti kode dan model bisa dilihat siapa pun.
  Untuk kerja praktek itu wajar dan justru transparan.
- Jangan pernah push dataset mentah atau file `.env` berisi kunci.
- Kalau nanti ada update kode: ubah file, lalu
  `git add -A`, `git commit -m "pesan"`, `git push`.
  Streamlit Cloud otomatis deploy ulang.
