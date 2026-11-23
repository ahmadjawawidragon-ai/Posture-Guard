# PostureGuard

Aplikasi web Streamlit: **Deteksi Kamera AI** (layanan utama, mode Depan dan Samping) + halaman edukasi pendamping.

## Menjalankan
```
pip install -r requirements.txt
streamlit run app.py
```
Butuh Python 3.10+ dan Streamlit 1.40+. Webcam langsung memakai `streamlit-webrtc` dan `av`; bila keduanya gagal terpasang, mode Foto masih jalan. Model dilatih dengan scikit-learn 1.9.x; pakai versi yang sama bila muncul peringatan versi.

Mau dipakai lewat link publik di HP dan laptop? Lihat **HOSTING.md**.

## Struktur
| Lokasi | Isi |
|---|---|
| `app.py` | router: menu Layanan Utama (Beranda, Deteksi Kamera AI, Evaluasi Model) dan Belajar (Tentang Postur, Panduan, Latihan, Kuis, Tentang Kami) |
| `deteksi.py` | halaman deteksi: Depan, Samping, kalibrasi, webcam, foto, maskot |
| `edukasi.py`, `mascots.py`, `ilustrasi.py`, `assets/` | Beranda, Evaluasi Model, halaman edukasi; maskot Sapa/Apin/Awi, logo, dan ilustrasi 5 postur (SVG, dibuat lewat kode). Halaman Evaluasi membaca `hasil/depan` dan `hasil/side` |
| `pose_utils.py`, `features.py`, `features_side.py`, `knowledge.py` | pemrosesan landmark, fitur, basis pengetahuan |
| `posture_model.joblib`, `reference_stats.json` | model dan rentang dataset tampak depan (5 kelas) |
| `posture_model_robust.joblib` | model Depan tahan-pinggul, dipilih lewat "Pinggul di kamera" di sidebar (lihat `pelatihan/train_depan_robust.py`) |
| `posture_model_side.joblib`, `reference_stats_side.json` | model dan rentang dataset tampak samping (3 kelas, ruang fitur `world`) |
| `pose_landmarker_*.task` | model MediaPipe (lite dan full). Slider "Model pose" memilih salah satunya (heavy tidak disertakan agar repo ringan; bila ditaruh di folder ini, otomatis muncul) |
| `pelatihan/` | skrip untuk laporan: ekstraksi, pelatihan, pengujian (tidak perlu untuk menjalankan aplikasi) |
| `data/` | `landmarks_side.csv`; taruh juga `Data ML.csv` (Dataset 1) di sini |
| `hasil/side/`, `hasil/depan/` | tabel dan gambar hasil pelatihan |

## Komparasi 4 model Dataset 1 (untuk laporan)
```
pip install -r requirements-pelatihan.txt
python pelatihan/train_depan.py --csv "data/Data ML.csv"
```
Tambahkan `--fast` untuk uji cepat. Skrip mencari kolom subjek otomatis (`--group-col` untuk menentukan sendiri) dan TIDAK menimpa `posture_model.joblib`.
Hasil yang sudah dijalankan (data 4.794 baris, 13 subjek) ada di `hasil/depan/`: `tabel_perbandingan.md`, `03_perbandingan_model.csv`, `confusion_matrix_semua_model.png`, `perbandingan_model.png`, `feature_importance_rf.png`, `laporan_klasifikasi_terbaik.txt`.

## Catatan: pinggul tertutup meja
Model Depan biasa memakai pinggul sebagai titik acuan. Bila pinggul tidak andal (webcam laptop di meja), hampir semua postur bisa terbaca TUP. Untuk itu pilih **Pinggul di kamera: Tertutup meja** (bawaan). Pilihan ini terutama andal untuk miring kiri/kanan; condong depan/belakang lebih baik lewat mode Samping.

## Keterbatasan
Perkiraan model, bukan diagnosis. Pinggul tertutup meja menurunkan akurasi (terutama mode Samping); lutut dipakai model tetapi belum digambar di kerangka mode Samping; mode Samping tidak mengenali miring kiri/kanan; dataset samping tanpa ID subjek sehingga skornya bisa optimis. Kalibrasi tegak mengurangi pengaruh posisi kamera dan meja.
