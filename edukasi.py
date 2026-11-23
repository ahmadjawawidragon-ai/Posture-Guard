"""edukasi.py : Beranda, Evaluasi Model, dan halaman edukasi PostureGuard.
Layanan utama tetap Deteksi Kamera AI (deteksi.py). app.py mengisi PAGES agar tautan antarhalaman jalan."""
import json
import random
from pathlib import Path

import pandas as pd
import streamlit as st

import ilustrasi as il
import mascots as mc

PAGES = {}          # diisi app.py: kunci -> st.Page
HASIL = Path(__file__).parent / "hasil"

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
.stApp,.stApp button,.stApp input,.stApp textarea{font-family:'Plus Jakarta Sans',system-ui,-apple-system,'Segoe UI',sans-serif}
.stApp{background:radial-gradient(900px 520px at 6% -4%,rgba(244,139,181,.34),transparent 62%),radial-gradient(820px 520px at 100% 6%,rgba(114,169,232,.34),transparent 62%),radial-gradient(760px 620px at 55% 112%,rgba(146,152,165,.28),transparent 62%),#EEF0F7;background-attachment:fixed}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#FBDDEA 0%,#E4E3F6 52%,#D6E6FA 100%)}
#MainMenu,footer{visibility:hidden}
.block-container{max-width:1100px;padding-top:1.6rem;padding-bottom:3rem}
.stApp h1,.stApp h2,.stApp h3{font-weight:800;letter-spacing:-.015em;color:#2b3242}
[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primary"] p{color:#2a1f2c!important;font-weight:800}
[data-testid="stBaseButton-primary"]{border-radius:99px!important;padding:.55rem 1.5rem!important;box-shadow:0 8px 20px rgba(244,139,181,.45)}
[data-testid="stExpander"]{background:rgba(255,255,255,.72);border-radius:16px}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.pg-hero-mark){background:linear-gradient(120deg,#F8C3D8 0%,#E3D6F6 50%,#B7D3F6 100%);border:0;border-radius:34px;padding:1.8rem 2.2rem;box-shadow:0 22px 50px rgba(122,110,170,.28);position:relative;overflow:hidden}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.pg-hero-mark)::before{content:"";position:absolute;width:340px;height:340px;border-radius:50%;background:rgba(255,255,255,.35);right:-90px;top:-130px}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.pg-hero-mark)::after{content:"✦  ✦";position:absolute;left:46%;bottom:14px;font-size:1.5rem;letter-spacing:.8em;color:rgba(255,255,255,.9)}
.pg-eyebrow{display:inline-block;font-size:.76rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase;color:#9B2F63;background:rgba(255,255,255,.7);border-radius:99px;padding:4px 12px;margin-bottom:.7rem}
.pg-h1{font-size:clamp(2rem,4.6vw,3rem);line-height:1.1;font-weight:800;letter-spacing:-.025em;color:#2b3242;margin:0 0 .8rem}
.pg-lead{font-size:1.05rem;line-height:1.7;color:#3d4659;margin:0 0 1rem;max-width:34em}
.pg-panel{border-radius:30px;padding:28px 30px 26px;margin:28px 0;position:relative;overflow:hidden}
.pg-panel>*{position:relative;z-index:1}
.pg-panel::before{content:"";position:absolute;width:240px;height:240px;border-radius:50%;right:-70px;top:-100px;background:rgba(255,255,255,.4)}
.pg-panel::after{content:"✦";position:absolute;right:30px;bottom:12px;font-size:1.5rem;color:rgba(255,255,255,.95)}
.pk{background:#F8D3E3}.bl{background:#D2E4FA}.gr{background:#DCE0EA}
.pg-sec{font-size:1.55rem;font-weight:800;color:#2b3242;margin:0 0 .25rem;letter-spacing:-.015em}
.pg-sub{color:#4a5468;margin:0 0 1.1rem;font-size:.98rem}
.pg-grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(230px,1fr))}
.pg-grid5{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(185px,1fr))}
.pg-card{background:rgba(255,255,255,.9);border-radius:20px;padding:20px 22px;box-shadow:0 8px 20px rgba(80,90,130,.12);transition:transform .2s,box-shadow .2s}
.pg-card:hover{transform:translateY(-4px);box-shadow:0 14px 28px rgba(80,90,130,.18)}
.pg-card h4{margin:0 0 .35rem;font-size:1.02rem;font-weight:800;color:#2b3242}
.pg-card p{margin:0;color:#4a5468;font-size:.93rem;line-height:1.6}
.pg-num{display:inline-flex;width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#F48BB5,#72A9E8);color:#fff;font-weight:800;font-size:.95rem;align-items:center;justify-content:center;margin-bottom:.8rem}
.pg-chip{display:inline-block;background:rgba(255,255,255,.85);color:#3d4659;border-radius:99px;padding:4px 12px;font-size:.8rem;font-weight:700;margin:0 6px 6px 0}
.pg-pill{display:block;text-align:center;font-weight:800;font-size:.8rem;white-space:nowrap;color:#2a2233;border-radius:99px;padding:8px 4px;margin-bottom:8px}
.pg-pc{padding:14px 14px 16px;text-align:center}.pg-pc p{margin-top:6px;font-size:.86rem}
.pg-view{display:inline-block;margin-top:8px;font-size:.7rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:#6b7488;background:#EEF0F7;border-radius:99px;padding:2px 10px}
.pg-stat b{display:block;font-size:2.2rem;font-weight:800;line-height:1.1;background:linear-gradient(90deg,#E0679B,#4A86D4);-webkit-background-clip:text;background-clip:text;color:transparent}
.pg-stat span{font-size:.86rem;color:#4a5468}
.pg-note{border-left:5px solid #72A9E8;background:rgba(255,255,255,.78);border-radius:12px;padding:12px 16px;color:#2b3242;font-size:.94rem;margin:14px 0;line-height:1.6}
.pg-note.warn{border-color:#F48BB5}
.pg-say{display:flex;gap:14px;align-items:center;background:rgba(255,255,255,.85);border-radius:20px;padding:12px 20px;margin:14px 0;box-shadow:0 8px 20px rgba(80,90,130,.10)}
.pg-say p{margin:0;color:#2b3242;font-size:.96rem;line-height:1.55}
.pg-badge{display:inline-block;font-weight:800;font-size:1.25rem;border-radius:12px;padding:4px 14px;margin:2px 0 6px}
.pg-badge.ok{background:#CBF0DD;color:#0f4a29}.pg-badge.bad{background:#FBD3E2;color:#7a1741}
.pg-foot{margin-top:3rem;padding-top:1rem;border-top:1px solid rgba(43,50,66,.15);color:#5d667a;font-size:.82rem}
.pg-pagehead{display:flex;gap:18px;align-items:center;justify-content:space-between;flex-wrap:wrap;border-radius:30px;padding:24px 30px;margin-bottom:10px;position:relative;overflow:hidden}
.pg-pagehead::before{content:"";position:absolute;width:260px;height:260px;border-radius:50%;right:-60px;top:-110px;background:rgba(255,255,255,.4)}
.pg-pagehead>*{position:relative;z-index:1}.pg-ph-text{flex:1 1 320px}.pg-pagehead .pg-h1{font-size:clamp(1.8rem,3.6vw,2.5rem)}
.pg-pagehead.pk{background:linear-gradient(120deg,#F8C3D8,#E7D8F4)}.pg-pagehead.bl{background:linear-gradient(120deg,#C5DBF8,#E3DAF5)}.pg-pagehead.gr{background:linear-gradient(120deg,#D5D9E4,#E8DCEB)}
.pg-trio-row{display:flex;align-items:flex-end;justify-content:center;gap:0}.pg-trio-row>div{margin:0 -8px}
.pg-bub{position:relative;display:inline-block;background:#fff;border-radius:18px;padding:8px 16px;font-weight:800;font-size:.92rem;color:#2b3242;box-shadow:0 8px 18px rgba(80,90,130,.18);margin:0 0 14px 36px}
.pg-bub::after{content:"";position:absolute;bottom:-9px;left:28px;border:10px solid transparent;border-top-color:#fff;border-bottom:0}
.pg-float{animation:pgf 4.2s ease-in-out infinite}.pg-float.d2{animation-delay:-1.4s}.pg-float.d3{animation-delay:-2.8s}
@keyframes pgf{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}
.pg-mc{border-radius:24px;padding:20px 20px 22px;text-align:center;box-shadow:0 10px 24px rgba(80,90,130,.14);transition:transform .2s}.pg-mc:hover{transform:translateY(-5px)}
.pg-mc.sapa{background:linear-gradient(180deg,#FFE3EE,#fff 85%)}.pg-mc.apin{background:linear-gradient(180deg,#DDEBFF,#fff 85%)}.pg-mc.awi{background:linear-gradient(180deg,#E6E8EE,#fff 85%)}
.pg-mc .art{display:flex;align-items:flex-end;justify-content:center;height:250px}
.pg-mc .nm{font-size:1.5rem;font-weight:800;color:#2b3242;margin:.5rem 0 .1rem}
.pg-mc .quote{display:inline-block;background:#fff;border-radius:14px;padding:6px 12px;font-weight:700;font-size:.86rem;color:#2b3242;box-shadow:0 4px 12px rgba(80,90,130,.12);margin:.3rem 0 .6rem}
.pg-mc p{color:#4a5468;font-size:.9rem;line-height:1.55;margin:.5rem 0 0}
@media (prefers-reduced-motion:reduce){.pg-float{animation:none}.pg-card,.pg-mc{transition:none}}
</style>"""

TIPS = [("sapa", "Berdiri dan regangkan badan sebentar. Ganti posisi minimal tiap setengah jam."),
        ("apin", "Tepi atas layar sebaiknya sejajar mata atau sedikit di bawahnya."),
        ("awi", "Turunkan bahumu dari telinga, pelan-pelan saja."),
        ("apin", "Dari samping, telinga, bahu, dan pinggul idealnya hampir segaris."),
        ("awi", "Kalau mulai pegal, ganti posisi. Tidak perlu dipaksa tegak terus."),
        ("sapa", "Minum dulu, sekalian jalan sebentar.")]

REFS = [("OSHA: daftar periksa stasiun kerja komputer (arsip)", "https://webarchive.library.unt.edu/eot2008/20090830020136/http://osha.gov/SLTC/etools/computerworkstations/checklist.html"),
        ("Mayo Clinic: nyeri punggung, gejala dan penyebab", "https://www.mayoclinic.org/diseases-conditions/back-pain/symptoms-causes/syc-20369906"),
        ("Mayo Clinic: kapan harus ke dokter untuk nyeri punggung", "https://www.mayoclinic.org/symptoms/back-pain/basics/when-to-see-doctor/sym-20050878"),
        ("Medical News Today: aturan 20-20-20 untuk mata", "https://www.medicalnewstoday.com/articles/321536.php")]


# ------------------------------------------------------------------ komponen kecil
def _h(html):
    """Rapatkan HTML jadi satu baris (baris kosong/indentasi memecah blok HTML di Markdown)."""
    st.markdown(" ".join(line.strip() for line in html.strip().splitlines()), unsafe_allow_html=True)


def _sec(title, sub=None):
    _h(f'<div class="pg-sec" style="margin-top:2rem">{title}</div>' + (f'<p class="pg-sub">{sub}</p>' if sub else ""))


def _gridh(cards, cls="pg-grid"):
    return f'<div class="{cls}">' + "".join(cards) + "</div>"


def _grid(cards):
    _h(_gridh(cards))


def _panel(tone, title, sub, inner):
    return f'<div class="pg-panel {tone}"><div class="pg-sec">{title}</div>' + (f'<p class="pg-sub">{sub}</p>' if sub else "") + inner + "</div>"


def _mascot_cards():
    data = [("sapa", "Ayo gerak, ayo tegak!", ["Energik", "Ceria", "Tukang ngingetin"], "Merayakan saat posturmu tegak dan mengajakmu bergerak."),
            ("apin", "Hmm… kalau menurut datanya…", ["Pintar", "Pemalu", "Suka baca"], "Menjelaskan materi dan jawaban kuis."),
            ("awi", "Santai… tapi jangan nyender banget.", ["Tenang", "Kalem", "Cuek tapi peduli"], "Pengingat santai saat kamu keliru atau lagi lelah.")]
    cards = []
    for i, (k, quote, chips, role) in enumerate(data):
        chip_html = "".join(f'<span class="pg-chip">{c}</span>' for c in chips)
        cards.append(f'<div class="pg-mc {k}"><div class="art"><div class="pg-float d{i + 1}">{mc.img(k, 170)}</div></div><div class="nm">{mc.INFO[k][0]}</div>'
                     f'<div class="quote">“{quote}”</div><div>{chip_html}</div><p>{role}</p></div>')
    return _gridh(cards)


def _card(title, body, n=None):
    num = f'<div class="pg-num">{n}</div>' if n else ""
    return f'<div class="pg-card">{num}<h4>{title}</h4><p>{body}</p></div>'


def _note(text, warn=False):
    _h(f'<div class="pg-note{" warn" if warn else ""}">{text}</div>')


def _say(who, text):
    _h(f'<div class="pg-say">{mc.img(who, 72)}<p><b>{mc.INFO[who][0]}:</b> {text}</p></div>')


def _foot():
    _h('<div class="pg-foot">Posture Guard · Proyek pembelajaran mesin. Konten edukasi, bukan diagnosis atau pengobatan.</div>')


def _go(key, label, icon=":material/arrow_forward:"):
    st.page_link(PAGES[key], label=label, icon=icon)


def _n(x):
    return f"{int(x):,}".replace(",", ".")


def _fmt(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


@st.cache_data(show_spinner=False)
def _hasil(folder):
    """Baca tabel perbandingan dan ringkasan dari hasil/<folder>; None bila belum ada."""
    try:
        tab = pd.read_csv(HASIL / folder / "03_perbandingan_model.csv")
        res = json.load(open(HASIL / folder / "results.json", encoding="utf-8"))
        return tab, res
    except Exception:
        return None, None


def _best_f1(folder):
    tab, res = _hasil(folder)
    if tab is None:
        return None
    row = tab[tab["model"] == res["model_terbaik"]].iloc[0]
    return row["ho_f1_macro"], res


# ------------------------------------------------------------------ Beranda
def beranda():
    with st.container(border=True):
        _h('<span class="pg-hero-mark"></span>')
        c1, c2 = st.columns([1.2, 1], vertical_alignment="center")
        with c1:
            _h(mc.img("logo", 150, "Logo Posture Guard") +
               '<span class="pg-eyebrow" style="margin-top:.9rem">Proyek pembelajaran mesin</span>'
               '<h1 class="pg-h1">Cek posturmu langsung dari kamera.</h1>'
               '<p class="pg-lead">PostureGuard membaca titik tubuh dari kamera, lalu model Random Forest menentukan apakah '
               "dudukmu tegak, condong ke depan, atau condong ke belakang. Bila terlalu lama tidak tegak, kamu mendapat peringatan.</p>"
               '<span class="pg-chip">5 postur (tampak depan)</span><span class="pg-chip">3 postur (tampak samping)</span>'
               '<span class="pg-chip">Kalibrasi tegak</span><span class="pg-chip">Webcam atau foto</span>')
            if st.button("Mulai deteksi", type="primary"):
                st.switch_page(PAGES["deteksi"])
        with c2:
            _h('<div class="pg-bub">Halo! Yuk cek posturmu.</div><div class="pg-trio-row">'
               f'<div class="pg-float">{mc.img("sapa", 150)}</div><div class="pg-float d2">{mc.img("apin", 190)}</div><div class="pg-float d3">{mc.img("awi", 150)}</div></div>')

    _h(_panel("pk", "Cara memakai", "Tiga langkah, sekitar satu menit.", _gridh([
        _card("Pilih sudut kamera", "Mode Depan mengenali 5 postur. Mode Samping mengenali 3 postur dan butuh kamera di sisi tubuhmu.", 1),
        _card("Kalibrasi duduk tegak", "Duduk setegak yang nyaman, tekan Kalibrasi, lalu diam 5 detik. Ini menyesuaikan model dengan posisi kamera dan mejamu.", 2),
        _card("Duduk seperti biasa", "Label postur, tingkat keyakinan, dan peringatan muncul langsung di samping video.", 3)])))

    desc = [("TUP", "Telinga, bahu, dan pinggul sejajar. Tidak ada yang perlu diubah."),
            ("TLF", "Kepala atau badan maju ke arah layar. Naikkan layar dan tegakkan punggung."),
            ("TLB", "Badan bersandar terlalu rebah. Duduk lebih dalam dan topang pinggang."),
            ("TLL", "Badan miring ke kiri. Seimbangkan kedua bahu."),
            ("TLR", "Badan miring ke kanan. Seimbangkan kedua bahu.")]
    pcs = [f'<div class="pg-card pg-pc"><span class="pg-pill" style="background:{il.POSTUR[k][1]}">{k}: {il.POSTUR[k][0]}</span>{il.img(k)}'
           f'<p>{d_}</p><span class="pg-view">tampak {il.POSTUR[k][2]}</span></div>' for k, d_ in desc]
    _h(_panel("bl", "Postur yang dikenali", "Lima postur dari dua sudut kamera. Garis putus-putus adalah acuan tubuh yang tegak.", _gridh(pcs, "pg-grid5")))
    _say("awi", "TLL dan TLR cuma dikenali di mode Depan. Dari samping, miring kiri atau kanan nggak kelihatan.")

    _h(_panel("gr", "Bagaimana mesin menilai", None, _gridh([
        _card("Kamera", "Tiap frame dari webcam atau foto dibaca.", 1),
        _card("MediaPipe Pose", "Mencari titik tubuh: telinga, bahu, pinggul, dan lutut (tampak samping).", 2),
        _card("Fitur sudut", "Dari titik-titik itu dihitung sudut badan dan leher.", 3),
        _card("Random Forest", "Memberi label dan peluang tiap postur. Peringatan muncul bila tidak tegak terlalu lama.", 4)])))

    d, s = _best_f1("depan"), _best_f1("side")
    if d and s:
        _h(_panel("pk", "Hasil model", "Dari data uji yang dipisahkan dari data latih.", _gridh([
            f'<div class="pg-card pg-stat"><b>{_fmt(d[0])}</b><span>F1 makro, tampak depan ({_n(d[1]["n_holdout"])} sampel uji)</span></div>',
            f'<div class="pg-card pg-stat"><b>{_fmt(s[0])}</b><span>F1 makro, tampak samping ({_n(s[1]["n_holdout"])} sampel uji)</span></div>',
            f'<div class="pg-card pg-stat"><b>{_n(d[1]["n_total"])}</b><span>sampel Dataset 1 dari 13 subjek</span></div>'])))
        _note("Angka ini dari dataset berlabel. Di webcam, hasil bisa berbeda karena sudut kamera dan pencahayaan. Rincian dan keterbatasannya ada di halaman Evaluasi Model.")
        _go("evaluasi", "Lihat evaluasi model", ":material/analytics:")

    _h(_panel("gr", "Kenalan sama Sapa, Apin, dan Awi", "Tiga teman yang menemanimu belajar dan menjaga postur.", _mascot_cards()))

    _sec("Materi pendamping", "Supaya kamu paham kenapa postur penting.")
    cols = st.columns(4)
    for col, (k, l, ic, d_) in zip(cols, [("tentang", "Tentang Postur", ":material/accessibility_new:", "Postur dan tulang belakang."),
                                          ("panduan", "Panduan", ":material/menu_book:", "Duduk, berdiri, meja kerja."),
                                          ("latihan", "Latihan", ":material/self_improvement:", "Peregangan singkat."),
                                          ("kuis", "Kuis", ":material/quiz:", "Uji pemahamanmu.")]):
        with col:
            _go(k, l, ic)
            st.caption(d_)

    if "tip" not in st.session_state:
        st.session_state.tip = random.choice(TIPS)
    who, text = st.session_state.tip
    _sec("Tips hari ini")
    _say(who, text)
    if st.button("Tips lain"):
        st.session_state.tip = random.choice([t for t in TIPS if t != st.session_state.tip])
        st.rerun()
    _foot()


# ------------------------------------------------------------------ Evaluasi Model
def _tabel(folder, judul, catatan, gambar):
    tab, res = _hasil(folder)
    if tab is None:
        st.info("Hasil belum tersedia. Jalankan skrip pelatihan di folder pelatihan/ untuk membuatnya.")
        return
    t = pd.DataFrame({"Model": tab["model"], "Accuracy": tab["ho_accuracy"], "Precision": tab["ho_precision_macro"],
                      "Recall": tab["ho_recall_macro"], "F1 makro": tab["ho_f1_macro"], "F1 CV (latih)": tab["cv_f1_macro_mean"],
                      "F1 latih": tab["train_f1_macro"]})
    t["Dipilih"] = ["ya" if m == res["model_terbaik"] else "" for m in tab["model"]]
    _h(f'<p class="pg-sub">{judul}</p>')
    st.dataframe(t.style.format({c: "{:.3f}" for c in t.columns[1:7]}), hide_index=True)
    st.caption("Accuracy, Precision, Recall, dan F1 dihitung pada data uji. Precision, Recall, dan F1 adalah rata-rata makro antar kelas.")
    for g, cap in gambar:
        p = HASIL / folder / g
        if p.exists():
            st.image(str(p), caption=cap)
    _h(catatan)


def _ketahanan():
    p = HASIL / "depan" / "uji_ketahanan.csv"
    if not p.exists():
        return
    t = pd.read_csv(p).rename(columns={"model": "Model", "kondisi": "Kondisi pinggul", "f1_makro": "F1 makro", **{f"recall_{c}": c for c in ("TLB", "TLF", "TLL", "TLR", "TUP")}})
    _sec("Uji ketahanan: bila pinggul tidak andal",
         "Pinggul tertutup meja atau ditebak MediaPipe membuat titik atas tubuh bergeser terhadap pinggul. Disimulasikan pada subjek yang tidak dipakai melatih. Angka kelas adalah recall.")
    st.dataframe(t.style.format({c: "{:.2f}" for c in t.columns[2:]}), hide_index=True)
    _note("<b>Temuan.</b> Model biasa sangat bergantung pada pinggul: saat pinggul tidak andal sepenuhnya, recall TLB, TLF, TLL, dan TLR turun ke sekitar 0 sementara TUP tetap 1,00, "
          "artinya semua postur terbaca tegak. Model tahan-pinggul (dilatih dengan augmentasi pinggul) mempertahankan miring kiri/kanan (0,90 dan 0,92), "
          "tetapi condong depan/belakang tetap lemah (TLF 0,56, TLB 0,27) dan F1 pada pinggul normal turun dari 0,993 ke 0,877. Karena itu aplikasi menyediakan pilihan <i>Pinggul di kamera</i>.")
    _note("Ini hasil simulasi pada dataset, bukan data webcam nyata. Ketepatan di webcam perlu dicoba langsung.", warn=True)


def evaluasi():
    _h(mc.pagehead("Layanan utama", "Evaluasi Model", "Perbandingan beberapa model pada dua dataset. Model terbaik dipilih dari validasi silang bersarang pada data latih, bukan dari hasil uji.", "apin", "bl"))
    t1, t2 = st.tabs(["Tampak depan (Dataset 1)", "Tampak samping (Dataset 2)"])
    with t1:
        _tabel("depan", "Dataset 1: MultiPosture, 4.794 sampel dari 13 subjek, 5 kelas. Data dibagi per subjek: 3.774 latih dan 1.020 uji.",
               '<div class="pg-note"><b>Model terpilih: Random Forest</b> (F1 CV 0,989 ± 0,013). Hanya 2 dari 1.020 sampel uji yang salah.</div>'
               '<div class="pg-note warn"><b>Baca dengan hati-hati.</b> Data uji berasal dari satu lipatan berisi beberapa subjek saja, jadi skor hampir sempurna ini belum tentu berlaku untuk orang baru. '
               "Gradient Boosting (di sini <code>HistGradientBoosting</code>, bukan XGBoost) punya F1 latih 1,000 dan simpangan baku CV 0,086, tanda kurang stabil antar subjek dan kemungkinan overfit. "
               "Hasil di webcam bisa lebih rendah.</div>",
               [("confusion_matrix_semua_model.png", "Confusion matrix pada data uji"), ("perbandingan_model.png", "F1 makro: validasi silang vs data uji")])
        _ketahanan()
    with t2:
        _tabel("side", "Dataset 2: Sitting Posture Detection, 917 sampel yang landmark-nya terdeteksi dari 924 anotasi, 3 kelas. 692 latih dan 225 uji.",
               '<div class="pg-note"><b>Model terpilih: Random Forest</b> (F1 CV 0,806). Fitur terpenting adalah sudut badan (<code>trunk_tilt</code>). '
               "Kesalahan terbanyak adalah TUP dan TLB yang tertukar (19 sampel).</div>"
               '<div class="pg-note warn"><b>Keterbatasan.</b> Dataset ini tidak punya ID subjek, dan orang yang sama tampak muncul di beberapa pembagian data, jadi skor uji bisa terlalu optimis. '
               "Gradient Boosting dan MLP kemungkinan overfit (F1 latih 0,995 dan 0,974). Koordinat world jauh lebih baik daripada koordinat citra (F1 CV 0,79 vs 0,51), jadi itu yang dipakai.</div>",
               [("confusion_matrix_semua_model.png", "Confusion matrix pada data uji"), ("perbandingan_model.png", "F1 makro: validasi silang vs data uji"),
                ("feature_importance_rf.png", "Fitur terpenting menurut Random Forest")])
    _foot()


# ------------------------------------------------------------------ Tentang Postur
def tentang_postur():
    _h(mc.pagehead("Belajar", "Tentang Postur", "Postur adalah cara tubuhmu menahan diri melawan gravitasi saat duduk, berdiri, atau bergerak. "
                   "Posisi yang netral membuat beban terbagi lebih rata. Tapi tubuh dirancang untuk bergerak: Mayo Clinic menyarankan mengganti posisi minimal setiap setengah jam.", "apin", "bl"))
    _say("apin", "Kuncinya netral dan sering ganti posisi, bukan kaku.")
    _sec("Mengenal tulang belakang", "Tulang belakang punya lengkung alami berbentuk S yang membantu meredam beban.")
    _grid([_card("Leher (servikal) · 7 ruas", "Menopang kepala dan paling bebas bergerak. Kepala yang terus maju (TLF) menambah beban di sini."),
           _card("Punggung atas (toraks) · 12 ruas", "Terhubung dengan tulang rusuk sehingga geraknya lebih terbatas. Bahu membulat biasanya terasa di area ini."),
           _card("Punggung bawah (lumbal) · 5 ruas", "Memikul banyak beban badan. Sandaran yang menopang pinggang membantu saat duduk lama dan saat tidak terlalu rebah (TLB)."),
           _card("Sakrum dan koksigis", "Dasar tulang belakang yang menyambung ke panggul.")])
    _sec("Kaitannya dengan label PostureGuard")
    _grid([_card("TUP · Tegak", "Telinga, bahu, pinggul hampir segaris dari samping."),
           _card("TLF · Condong depan", "Kepala atau badan maju ke arah layar."),
           _card("TLB · Condong belakang", "Badan bersandar terlalu rebah.")])
    st.write("")
    _go("deteksi", "Cek posturmu sekarang", ":material/videocam:")
    _foot()


# ------------------------------------------------------------------ Panduan
def panduan():
    _h(mc.pagehead("Belajar", "Panduan Postur", "Berdasarkan daftar periksa ergonomi OSHA dan saran Mayo Clinic. Tiap orang berbeda, jadi sesuaikan dengan kenyamananmu.", "awi", "gr"))
    t1, t2, t3, t4 = st.tabs(["Duduk", "Berdiri", "Komputer", "Kesalahan umum"])
    with t1:
        st.markdown("- Telapak kaki rata di lantai atau di pijakan.\n- Paha kira-kira sejajar lantai, lutut setinggi pinggul.\n"
                    "- Sandaran menopang punggung bawah (area lumbal).\n- Bahu rileks, siku dekat badan.\n- Ganti posisi minimal tiap setengah jam.")
        _note("<b>Kalau PostureGuard menunjukkan TLB:</b> geser pinggul lebih dalam ke kursi dan topang pinggang.")
    with t2:
        st.markdown("- Berat badan terbagi rata di kedua kaki.\n- Lutut tidak dikunci.\n- Kepala, leher, dan badan segaris, tidak membungkuk.\n- Dagu sejajar lantai, bahu rileks.")
    with t3:
        st.markdown("- Tepi atas layar **sejajar atau di bawah** tinggi mata, supaya leher tidak menunduk atau mendongak.\n"
                    "- Layar tepat di depanmu (tidak perlu menoleh), berjarak kira-kira sepanjang lengan.\n- Kurangi silau dari jendela atau lampu.\n"
                    "- Keyboard dan mouse dekat badan, pergelangan tangan lurus.\n- Laptop: pakai dudukan layar dan keyboard terpisah bila dipakai lama.\n"
                    "- Untuk mata: aturan 20-20-20 (tiap 20 menit, lihat sejauh 6 meter selama 20 detik). Ini anjuran umum dokter mata, meski bukti ilmiahnya masih terbatas.")
        _note("<b>Kalau PostureGuard menunjukkan TLF:</b> naikkan layar, tarik dagu ke belakang, dan tegakkan punggung.")
    with t4:
        st.markdown("- Leher maju ke layar ponsel atau laptop.\n- Membungkuk saat menulis atau main gim.\n- Kaki menggantung atau menyilang lama.\n- Duduk berjam-jam tanpa berpindah posisi.")
    _say("awi", "Tidak perlu sempurna. Cukup sering ganti posisi, santai saja.")
    _go("deteksi", "Cek posturmu dengan kamera", ":material/videocam:")
    _foot()


# ------------------------------------------------------------------ Latihan
EX = [("1. Tarik dagu (chin tuck)", "Membantu TLF", "Duduk tegak, pandangan lurus.\nTarik dagu lurus ke belakang seperti membuat “dagu ganda”.\nTahan 5 detik, ulangi 8–10 kali."),
      ("2. Putar bahu", "Membantu TLF", "Angkat bahu ke arah telinga.\nPutar ke belakang dan turunkan perlahan.\nLakukan 10 putaran, lalu 10 ke arah sebaliknya."),
      ("3. Peregangan dada di pintu", "Membantu TLF", "Letakkan lengan di kusen pintu, siku sekitar 90°.\nLangkahkan satu kaki ke depan sampai dada terasa teregang ringan.\nTahan 20–30 detik, 2 kali tiap sisi."),
      ("4. Kucing-sapi (cat-cow)", "Untuk semua", "Merangkak dengan tangan di bawah bahu.\nTarik napas: dada naik, punggung melengkung lembut.\nEmbuskan: punggung membulat. Ulangi 8–10 napas."),
      ("5. Putaran badan di kursi", "Untuk semua", "Duduk tegak, tangan di sandaran atau lutut.\nPutar badan perlahan ke satu sisi.\nTahan 15–20 detik, ganti sisi.")]


def latihan():
    _h(mc.pagehead("Belajar", "Latihan dan Peregangan", "Gerakan ringan 1–2 menit untuk sela belajar atau bekerja. Label “Membantu TLF” hanya panduan umum untuk kebiasaan badan maju, bukan terapi.", "sapa", "pk"))
    for t, tag, steps in EX:
        with st.expander(f"{t}  ·  {tag}"):
            st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(steps.split("\n"), 1)))
    _say("sapa", "Lakukan pelan-pelan dan jangan dipaksa. Semangat!")
    _note("Hentikan bila terasa nyeri tajam, kesemutan, atau pusing. Bila punya cedera atau kondisi tertentu, tanyakan dulu ke tenaga kesehatan.", warn=True)
    _foot()


# ------------------------------------------------------------------ Kuis
Q = [("Di mana sebaiknya posisi layar komputer?", ["Jauh di atas kepala", "Tepi atas sejajar atau di bawah tinggi mata", "Di pangkuan"], 1,
      "Daftar periksa OSHA: tepi atas layar sejajar atau di bawah tinggi mata, agar leher tidak menunduk atau mendongak."),
     ("Seberapa sering sebaiknya mengganti posisi saat duduk lama?", ["Minimal tiap setengah jam", "Sekali sehari", "Cukup saat pulang"], 0,
      "Mayo Clinic menyarankan mengubah posisi minimal setiap setengah jam."),
     ("Berapa ruas tulang leher (servikal)?", ["5", "7", "12"], 1, "Leher 7 ruas, punggung atas 12, punggung bawah 5."),
     ("Dari samping, postur tegak yang baik menampilkan…", ["Kepala jauh di depan bahu", "Telinga, bahu, dan pinggul hampir segaris", "Punggung bungkuk"], 1,
      "Garis itulah yang dibaca PostureGuard pada mode Samping."),
     ("Kaki yang nyaman saat duduk…", ["Menggantung", "Telapak rata di lantai atau pijakan", "Dilipat di bawah kursi"], 1, "OSHA menganjurkan telapak kaki rata di lantai atau pijakan."),
     ("Nyeri punggung menjalar ke kaki disertai kebas atau kesemutan. Sebaiknya…", ["Abaikan saja", "Periksa ke tenaga kesehatan", "Tunggu sampai hilang sendiri"], 1,
      "Mayo Clinic: nyeri yang menjalar ke kaki atau disertai kelemahan, kebas, atau kesemutan perlu dikonsultasikan. Ini di luar jangkauan koreksi postur.")]


def _answer(i):
    ss = st.session_state
    ss.qa = i
    ss.qs += int(i == Q[ss.qi][2])


def _next():
    st.session_state.qi += 1
    st.session_state.qa = None


def _restart():
    st.session_state.qi, st.session_state.qs, st.session_state.qa = 0, 0, None
    st.session_state.bal = False


def kuis():
    ss = st.session_state
    for k, v in (("qi", 0), ("qs", 0), ("qa", None)):
        ss.setdefault(k, v)
    _h(mc.pagehead("Belajar", "Kuis Postur", "Enam soal singkat. Setiap jawaban ada penjelasannya.", "sapa", "pk"))
    if ss.qi >= len(Q):
        n = ss.qs
        who = "sapa" if n >= 5 else "apin" if n >= 3 else "awi"
        msg = "Kamu sudah paham dasar-dasarnya." if n >= 5 else "Lumayan. Baca panduan lagi supaya makin mantap." if n >= 3 else "Awal yang baik. Baca panduannya dulu, lalu coba lagi."
        st.progress(1.0)
        if n >= 5 and not ss.get("bal"):
            ss.bal = True
            st.balloons()
        _h(f'<div class="pg-sec">Skormu: {n} dari {len(Q)}</div>')
        _say(who, msg)
        st.button("Main lagi", type="primary", on_click=_restart)
        _go("panduan", "Buka panduan", ":material/menu_book:")
        _foot()
        return
    q, opts, ans, why = Q[ss.qi]
    st.progress(ss.qi / len(Q), text=f"Soal {ss.qi + 1} dari {len(Q)}")
    st.subheader(q)
    if ss.qa is None:
        for i, o in enumerate(opts):
            st.button(o, key=f"q{ss.qi}_{i}", on_click=_answer, args=(i,))
        return
    for i, o in enumerate(opts):
        st.markdown(("**Benar:** " if i == ans else "Jawabanmu: " if i == ss.qa else "") + o)
    ok = ss.qa == ans
    who = "sapa" if ok else "awi"
    _say(who, ("Benar!" if ok else "Belum tepat, tidak apa-apa.") + f" {why}")
    st.button("Lihat skor" if ss.qi == len(Q) - 1 else "Soal berikutnya", type="primary", on_click=_next)


# ------------------------------------------------------------------ Tentang Kami
def tentang_kami():
    _h(mc.pagehead("Tentang", "Tentang Posture Guard", "Posture Guard adalah proyek pembelajaran mesin untuk membantu orang sadar postur. Layanan utamanya adalah deteksi postur lewat kamera. Halaman edukasi ada sebagai pendamping.", "trio", "pk"))
    _go("deteksi", "Buka Deteksi Kamera AI", ":material/videocam:")
    _h(_panel("gr", "Tim maskot", "Sapa, Apin, dan Awi menemanimu di seluruh aplikasi.", _mascot_cards()))
    _sec("Teknologi")
    _h('<span class="pg-chip">MediaPipe Pose</span><span class="pg-chip">scikit-learn</span><span class="pg-chip">Streamlit</span><span class="pg-chip">Python</span>')
    _sec("Keterbatasan sistem")
    _note("Hasil adalah <b>perkiraan model</b>, bukan diagnosis.<br>"
          "Pinggul atau badan yang <b>tertutup meja</b> menurunkan akurasi sudut badan. Model Depan biasa sangat bergantung pada pinggul, sehingga tersedia model tahan-pinggul (miring kiri/kanan paling andal, condong depan/belakang tetap lemah).<br>"
          "Miring atau condong ringan (kurang dari sekitar 15°) umumnya tetap terbaca TUP, karena kelas miring pada dataset adalah miring yang jelas.<br>"
          "Di mode Samping, landmark lutut dipakai model tetapi <b>belum digambar</b> di kerangka video.<br>"
          "Mode Samping hanya 3 kelas dan tidak mengenali miring kiri atau kanan.<br>"
          "Dataset samping tidak punya ID subjek, jadi skor evaluasinya bisa lebih optimis dari performa nyata.<br>"
          "<b>Kalibrasi tegak</b> mengurangi pengaruh posisi kamera dan meja, tetapi tidak mengembalikan bagian tubuh yang tertutup.", warn=True)
    _sec("Bacaan lanjutan")
    for t, u in REFS:
        st.markdown(f"- [{t}]({u})")
    _note("Konten ini edukasi, bukan diagnosis atau pengobatan. Bila nyeri menetap, menjalar, atau disertai kesemutan dan lemah, periksakan ke tenaga kesehatan.")
    _foot()
