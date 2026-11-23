# Hosting PostureGuard (link publik untuk HP dan laptop)

## Pilihan singkat
| Pilihan | Biaya | Link | Cocok untuk |
|---|---|---|---|
| **A. Streamlit Community Cloud** | gratis | tetap, `namamu.streamlit.app` | link publik yang selalu ada (disarankan) |
| **B. Jalankan di laptop + tunnel** | gratis | sementara, berubah tiap dijalankan | demo satu sesi, laptop harus menyala |

Hugging Face Spaces tidak kusarankan: menurut dokumentasi resminya, Space Streamlit kini berjalan lewat Docker dan membuat Space bercompute memerlukan paket berbayar.

## A. Streamlit Community Cloud
1. Buat akun GitHub, lalu buat repository baru (public paling mudah).
2. Upload **isi** folder `PostureGuard` ke repository, supaya `app.py` ada di akar repo. Pastikan folder tersembunyi `.streamlit/` ikut, juga `packages.txt` dan `requirements.txt`. Upload lewat browser dibatasi sekitar 25 MB per file (file terbesar kini ±9 MB, aman). Cara lain: GitHub Desktop atau `git`.
3. Buka share.streamlit.io, login dengan GitHub, pilih **Create app**, lalu isi repository, branch, dan **Main file path: `app.py`**.
4. Di **Advanced settings** pilih versi Python yang **sama dengan di laptopmu** (mis. 3.12 atau 3.13). MediaPipe harus punya paket untuk versi itu.
5. Tekan **Deploy**. Build butuh beberapa menit. Setelah jadi, salin link-nya dan buka di HP dan laptop.
6. Mengubah aplikasi: push ke GitHub, aplikasi ter-update otomatis. Mengubah `requirements.txt` memicu instal ulang.

Batas gratis (dokumentasi Streamlit, angka per Februari 2024 dan bisa berubah tanpa pemberitahuan): CPU maksimal 2 core, memori maksimal ±2,7 GB, dan aplikasi **tidur setelah 12 jam tanpa kunjungan**. Buka linknya beberapa menit sebelum presentasi untuk membangunkannya.

### Kamera di versi hosting
- HTTPS otomatis (kamera browser butuh HTTPS), jadi kamera bisa diminta izinnya di HP dan laptop.
- **Mode "Foto (kamera / unggah)"** tidak memakai WebRTC dan paling andal, terutama di HP. Pakai ini sebagai cadangan.
- **Mode "Webcam langsung"** memakai WebRTC. Dokumentasi streamlit-webrtc menyebut server TURN mungkin diperlukan, termasuk di Streamlit Community Cloud. Gejalanya: tombol START ditekan tapi video tidak tersambung. Bila terjadi, daftar ke penyedia TURN (mis. Twilio atau Metered), lalu tempel isi `.streamlit/secrets.toml.example` ke **Settings > Secrets** di Streamlit Cloud dengan data milikmu. Aplikasi membaca bagian `[turn]` otomatis. Jangan commit `secrets.toml` ke GitHub.
- Server gratis lebih lemah dari laptopmu. Pilih **Model pose: lite** di sidebar bila video patah-patah.

## B. Dari laptop lewat tunnel (tanpa hosting)
1. Jalankan `streamlit run app.py`.
2. Di terminal lain: `cloudflared tunnel --url http://localhost:8501` (atau `ngrok http 8501`). Pasang alatnya dari situs resminya. Kamu mendapat link https publik sementara.
3. Laptop harus tetap menyala dan online, dan link berubah tiap tunnel dijalankan. Webcam langsung tetap bisa butuh TURN, jadi siapkan mode Foto.

## Sebelum presentasi
- Buka link 10 menit sebelumnya dan cek di HP (mode Foto) dan laptop (Webcam langsung, lalu kalibrasi).
- Siapkan cadangan: rekaman layar demo dan aplikasi yang jalan lokal.
- Beri tahu penguji bahwa gambar kamera diproses di server aplikasi. Kode aplikasi tidak menyimpannya ke disk.
