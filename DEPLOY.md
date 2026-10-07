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
4. Deploy. Instalasi pertama makan waktu beberapa menit
   (torch CPU ~200 MB). Setelah jadi, dapat URL publik.

## Catatan

- Repo publik berarti kode dan model bisa dilihat siapa pun.
  Untuk kerja praktek itu wajar dan justru transparan.
- Jangan pernah push dataset mentah atau file `.env` berisi kunci.
- Kalau nanti ada update kode: ubah file, lalu
  `git add -A`, `git commit -m "pesan"`, `git push`.
  Streamlit Cloud otomatis deploy ulang.
