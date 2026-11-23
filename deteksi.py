"""
deteksi.py : HALAMAN DETEKSI KAMERA AI (dibuka dari app.py lewat st.navigation).
Isi sama dengan app.py lama (app_backup.py); perubahan hanya: tanpa set_page_config, maskot di hasil,
peringatan model samping, dan bagian Keterbatasan.

Aplikasi klasifikasi postur duduk (MultiPosture + MediaPipe + Random Forest)

Jalankan seluruh aplikasi:  streamlit run app.py

Aturan yang dijaga aplikasi ini (agar cocok dengan dataset):
 1. Memakai pose_world_landmarks (bukan pose_landmarks); titik tengah pinggul tetap dikurangkan.
 2. Frame yang masuk MediaPipe TIDAK di-mirror. Hanya tampilan untuk pengguna yang boleh dicermin.
 3. Fitur dihitung dengan features.py yang sama dengan notebook.
 4. Panel pembanding menampilkan nilai fitur kamera di samping rentang dataset.
"""
import json
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import features as ft
import features_side as fsd
import knowledge as kb
import mascots
import pose_utils as pu

APP_DIR = Path(__file__).parent
FRONT_MODEL, FRONT_REF = APP_DIR / "posture_model.joblib", APP_DIR / "reference_stats.json"
SIDE_MODEL, SIDE_REF = APP_DIR / "posture_model_side.joblib", APP_DIR / "reference_stats_side.json"
ROBUST_MODEL = APP_DIR / "posture_model_robust.joblib"      # model Depan tahan-pinggul (lihat pelatihan/train_depan_robust.py)

LABELS = {
    "TUP": ("Tegak", "✅"),
    "TLF": ("Condong ke depan", "⚠️"),
    "TLB": ("Condong ke belakang", "⚠️"),
    "TLL": ("Miring ke kiri", "⚠️"),
    "TLR": ("Miring ke kanan", "⚠️"),
}
STATUS_TEXT = {"ok": "dalam rentang", "tepi": "di tepi", "luar": "di luar dataset"}
STATUS_COLOR = {"ok": "", "tepi": "background-color: #fff3cd; color: #664d03",
                "luar": "background-color: #f8d7da; color: #842029"}

# st.set_page_config dipanggil sekali di app.py (router).

MASCOT_OF = {"TUP": "sapa", "TLF": "apin", "TLB": "awi", "TLL": "apin", "TLR": "apin"}
MASCOT_SAY = {"TUP": ("Sapa", "Mantap, postur tegak! Pertahankan ya."),
              "TLF": ("Apin", "Badan atau kepalamu maju. Tegakkan punggung dan tarik dagu pelan-pelan."),
              "TLB": ("Awi", "Terlalu rebah, nih. Duduk lebih dalam dan topang pinggangmu."),
              "TLL": ("Apin", "Badanmu miring ke kiri. Coba seimbangkan kedua bahu."),
              "TLR": ("Apin", "Badanmu miring ke kanan. Coba seimbangkan kedua bahu.")}


# ----------------------------------------------------------------- muat model & referensi
@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


@st.cache_data
def load_reference(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def static_estimator(complexity, det_conf):
    return pu.PoseEstimator(complexity, det_conf, 0.5, live=False, model_dir=str(APP_DIR))


with st.sidebar:
    st.header("Pengaturan")
    view = st.radio("Sudut kamera", ["Depan", "Samping"],
                    help="Depan: 5 kelas (TUP, TLF, TLB, TLL, TLR) dari MultiPosture. "
                         "Samping: 3 kelas (TUP, TLF, TLB) dari Sitting Posture Detection. "
                         "Ubah lalu Stop dan Start ulang kamera.")
SIDE = view == "Samping"
MODEL_PATH, REF_PATH = (SIDE_MODEL, SIDE_REF) if SIDE else (FRONT_MODEL, FRONT_REF)
K = "side" if SIDE else "front"            # kalibrasi & statistik sesi dipisah per sudut kamera
CK, CE, CS, SK = f"calib_{K}", f"calib_end_{K}", f"calib_samples_{K}", f"stats_{K}"

if not MODEL_PATH.exists():
    st.error(f"`{MODEL_PATH.name}` belum ada di folder aplikasi. " + (
        "Jalankan `train_side.py` lalu salin hasilnya ke folder ini." if SIDE else
        "Jalankan notebook sampai bagian 10, lalu salin file itu ke folder ini."))
    st.stop()
if not REF_PATH.exists():
    st.error(f"`{REF_PATH.name}` belum ada. Buat dengan:\n\n" + (
        "`python build_reference_side.py --csv landmarks_side.csv`" if SIDE else
        '`python build_reference.py --csv "path/ke/Data ML.csv"`'))
    st.stop()

bundle, ref = load_bundle(str(MODEL_PATH)), load_reference(str(REF_PATH))
if SIDE and bundle.get("feature_space", "world") != "world":
    st.warning("Model samping ini memakai ruang fitur 'image' (versi lama). Pakai posture_model_side.joblib dan "
               "reference_stats_side.json versi ruang 'world' agar deteksi webcam konsisten.")
BUNDLE_FS = bundle.get("feature_set") or (f"side_{bundle['feature_space']}" if "feature_space" in bundle else None)
if BUNDLE_FS != ref["meta"]["feature_set"]:
    st.error(f"Set fitur model ({BUNDLE_FS}) berbeda dengan referensi "
             f"({ref['meta']['feature_set']}). Jalankan ulang build_reference.py dengan --feature-set yang sama.")
    st.stop()
CLASSES = bundle["classes"]
ROBUST = None
if not SIDE and ROBUST_MODEL.exists():
    ROBUST = load_bundle(str(ROBUST_MODEL))
    if list(ROBUST["feature_names"]) != list(bundle["feature_names"]) or list(ROBUST["classes"]) != list(CLASSES):
        st.warning("posture_model_robust.joblib tidak cocok dengan model Depan (fitur/kelas berbeda); diabaikan.")
        ROBUST = None


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    mode = st.radio("Sumber input", ["Webcam langsung", "Foto (kamera / unggah)"])
    hip_mode = "Pinggul terlihat penuh"
    if not SIDE and ROBUST is not None:
        hip_mode = st.radio("Pinggul di kamera", ["Tertutup meja (webcam laptop)", "Otomatis", "Terlihat penuh"],
                            help="Model biasa memakai pinggul sebagai titik acuan. Bila pinggul tertutup meja atau ditebak MediaPipe, "
                                 "model biasa cenderung membaca semua postur sebagai TUP. 'Tertutup meja' memakai model yang dilatih "
                                 "tahan terhadap hal itu (miring kiri/kanan paling andal). 'Otomatis' memilih dari visibilitas pinggul.")
    pose_opts = [i for i in (0, 1, 2) if (APP_DIR / f"pose_landmarker_{pu.TASK_MODELS[i]}.task").exists() or i < 2] or [1]
    complexity = st.select_slider("Model pose", options=pose_opts, value=1 if 1 in pose_opts else pose_opts[0],
                                  format_func=lambda i: pu.TASK_MODELS[i],
                                  help="lite = paling ringan (disarankan bila di-hosting), full = lebih akurat. Ubah, lalu Stop dan Start ulang kamera.")
    det_conf = st.slider("Ambang deteksi", 0.3, 0.9, 0.5, 0.05)
    smooth_n = st.slider("Penghalusan (jumlah frame)", 1, 30, 10,
                         help="Median koordinat N frame terakhir; mengurangi getar.")
    alert_sec = st.slider("Peringatan bila tidak tegak selama (detik)", 2, 30, 8)
    if SK not in st.session_state:
        st.session_state[SK] = pu.SessionStats()
    if st.button("Reset statistik sesi"):
        st.session_state[SK].reset()
    st.divider()
    st.subheader("Kalibrasi tegak")
    for k, v in ((CK, None), (CE, None), (CS, [])):
        st.session_state.setdefault(k, v)
    st.caption(("Duduk tegak senyaman mungkin MENYAMPING ke kamera" if SIDE else "Duduk tegak senyaman mungkin menghadap kamera")
               + ", klik tombol, lalu diam 5 detik.")
    if st.button("Kalibrasi (5 detik)", disabled=(mode != "Webcam langsung")):
        st.session_state[CE], st.session_state[CS] = time.time() + 5.0, []
    st.checkbox("Pakai kalibrasi", value=True, key="use_calib")
    if st.button("Hapus kalibrasi"):
        st.session_state[CK] = None
    st.caption("Status: " + ("✅ terkalibrasi" if st.session_state[CK] is not None else "belum dikalibrasi"))
    st.divider()
    mirror_display = st.checkbox("Cerminkan tampilan video (hanya tampilan)", value=True)
    flip_input = st.checkbox("⚠️ Balik frame SEBELUM MediaPipe (koreksi darurat)", value=False,
                             help="Biarkan mati. Nyalakan hanya jika 'Tes arah sumbu' gagal.")
    st.caption(f"Peringatan: {'model biner' if bundle.get('binary_model') is not None else '1 − P(TUP) dari model 5 kelas'}")
    st.caption(f"Model: **{bundle['model_name']}** | set fitur: **{BUNDLE_FS}** "
               f"({len(bundle['feature_names'])} fitur)")

st.markdown(mascots.pagehead("Layanan utama", "Deteksi Postur Duduk",
            f'Mode {"Samping (3 kelas: TUP, TLF, TLB)" if SIDE else "Depan (5 kelas: TUP, TLF, TLB, TLL, TLR)"}. '
            'Kamera → MediaPipe → fitur sudut → model → label postur dan peringatan.', "sapa", "pk"), unsafe_allow_html=True)
st.markdown('<div class="pg-note"><b>Cara pakai:</b> 1) pilih sudut kamera dan sumber input di sidebar, '
            '2) kalibrasi saat duduk tegak, 3) tekan START lalu duduk seperti biasa.'
            + (" Duduklah <b>menyamping</b>: telinga, bahu, pinggul, dan lutut sisi yang menghadap kamera harus terlihat. Arah hadap dikenali otomatis." if SIDE else
               " Pastikan bahu dan pinggul terlihat di kamera.") + '</div>', unsafe_allow_html=True)

st.caption("Gambar kamera diproses di server aplikasi ini untuk menebak postur. Kode aplikasi tidak menyimpannya ke disk.")

with st.expander("Keterbatasan sistem (baca sebelum memakai hasilnya)"):
    st.markdown(
        "- Ini **perkiraan model**, bukan diagnosis medis.\n"
        "- Pinggul atau badan yang **tertutup meja/kursi** menurunkan akurasi sudut badan. Model Depan biasa sangat bergantung pada "
        "pinggul sebagai titik acuan; bila pinggul tidak andal, hampir semua postur terbaca TUP. Karena itu tersedia model tahan-pinggul "
        "(pilihan 'Pinggul di kamera' di sidebar), yang paling andal untuk miring kiri/kanan.\n"
        "- Postur ringan (miring/condong kurang dari sekitar 15°) umumnya tetap terbaca TUP: kelas miring pada dataset adalah miring yang jelas (rata-rata 30–37°).\n"
        "- Di mode Samping, **landmark lutut dipakai model tetapi belum digambar** pada kerangka di video.\n"
        "- Mode Samping hanya 3 kelas (TUP, TLF, TLB); miring kiri/kanan hanya bisa dikenali di mode Depan.\n"
        "- Hasil dipengaruhi sudut kamera, pencahayaan, dan pakaian. TUP dan TLB paling sering tertukar.\n"
        "- **Kalibrasi tegak** (sidebar) dirancang untuk mengurangi pengaruh posisi kamera dan meja yang berbeda-beda: "
        "ia menggeser fitur agar postur tegakmu setara median TUP dataset. Ia hanya menghapus offset tetap, "
        "tidak mengembalikan bagian tubuh yang tertutup.")

with st.expander("Tes arah sumbu (lakukan sekali, 30 detik)", expanded=False) if not SIDE else st.expander("Tentang mode samping"):
  if SIDE:
    st.markdown("Model samping dilatih dari citra tampak samping (Dataset 2). Kemiringan kiri/kanan tidak dapat dikenali dari sudut ini; "
                "gunakan mode Depan untuk itu. Fitur horizontal dihitung relatif terhadap arah hadap sehingga tidak bergantung ke sisi mana Anda menghadap.")
  else:
    st.markdown(
        "Di dataset, miring ke **kiri badan sendiri** (TLL) menghasilkan `trunk_tilt_lateral` ≈ **+37°**, "
        "dan miring ke kanan (TLR) ≈ **−31°**.\n\n"
        "1. Duduk tegak, lalu condongkan badan ke **kiri badanmu sendiri**.\n"
        "2. `trunk_tilt_lateral` harus **positif** dan prediksi menjadi **TLL**.\n"
        "3. Kalau terbalik, frame yang masuk MediaPipe ter-mirror. Perbaiki sumbernya "
        "(jangan flip di aplikasi kecuali darurat).\n\n"
        "Catatan: saat duduk **tegak**, tanda `left_shoulder_x` tetap positif walau frame ter-mirror, karena "
        "MediaPipe memberi label kiri/kanan berdasarkan tampilan. Jadi tanda itu bukan detektor mirror yang andal; "
        "gejala yang pasti adalah arah miring lateral tertukar (TLL dan TLR).")


# ----------------------------------------------------------------- analisis + tampilan
def tup_median():
    return pd.Series({n: ref["by_class"]["TUP"][n]["p50"] for n in bundle["feature_names"]})


def calibration_step(raw):
    """Kumpulkan sampel selama kalibrasi; kembalikan pesan status (atau None)."""
    ss = st.session_state
    end = ss.get(CE)
    if end is None or mode != "Webcam langsung":
        return None
    if time.time() < end:
        ss[CS].append(raw.to_numpy(float))
        return f"Kalibrasi berjalan: duduk tegak dan diam, {max(end - time.time(), 0):.0f} detik lagi."
    samples, ss[CE] = ss[CS], None
    if len(samples) >= 15:
        ss[CK] = pd.Series(np.median(np.vstack(samples), axis=0), index=raw.index)
        return "Kalibrasi selesai. Postur tegakmu kini dianggap setara TUP dataset."
    return "Kalibrasi gagal: pose terlalu jarang terdeteksi. Ulangi."


def analyze(row, smoother, hip_vis=None):
    srow = smoother.update(row)
    if SIDE:
        raw = pu.side_feature_frame(srow, bundle["feature_names"], bundle["feature_space"]).iloc[0]
    else:
        raw = pu.feature_frame(srow, bundle["feature_set"], bundle["feature_names"]).iloc[0]
    msg = calibration_step(raw)
    calib = st.session_state[CK] if st.session_state.get("use_calib", True) else None
    if calib is not None and set(calib.index) != set(raw.index):      # kalibrasi dari model lain: abaikan
        calib = None
    used = pu.apply_calibration(raw, calib, tup_median())
    X = pd.DataFrame([used.to_numpy(float)], columns=bundle["feature_names"])
    robust_on = False
    if ROBUST is not None and not SIDE:
        if hip_mode.startswith("Tertutup"):
            robust_on = True
        elif hip_mode == "Otomatis" and hip_vis is not None:      # EMA visibilitas pinggul + histeresis agar tidak berkedip
            ema = 0.9 * st.session_state.get("hip_vis_ema", hip_vis) + 0.1 * hip_vis
            st.session_state["hip_vis_ema"] = ema
            robust_on = ema < (0.65 if st.session_state.get("hip_robust") else 0.55)
            st.session_state["hip_robust"] = robust_on
    model = ROBUST["model"] if robust_on else bundle["model"]
    probs = dict(zip(CLASSES, model.predict_proba(X)[0]))
    bm = None if robust_on else bundle.get("binary_model")        # model biner TUP vs Non-TUP (kelas 1 = Non-TUP), bila disimpan
    p_non = float(bm.predict_proba(X)[0][1]) if bm is not None else 1.0 - float(probs.get("TUP", 0.0))
    return srow, raw, used, probs, p_non, msg, robust_on


def build_panel(raw, used, pred_cls):
    rows = []
    for name in bundle["feature_names"]:
        r, v, u = ref["features"][name], float(raw[name]), float(used[name])
        c = ref["by_class"].get(pred_cls, {}).get(name)
        rows.append({
            "Fitur": name, "Kamera": v, "Dipakai model": u, "Min": r["min"], "P01": r["p01"], "Median": r["p50"],
            "P99": r["p99"], "Max": r["max"],
            "Rentang kelas prediksi (P05–P95)": f"{c['p05']:.2f} s/d {c['p95']:.2f}" if c else "-",
            "_s": pu.status_of(u, r)})
    return pd.DataFrame(rows)


def style_table(p):
    """Tabel berwarna menurut status (kolom _s dibuang dari tampilan)."""
    flags = p["_s"].tolist()
    shown = p.assign(Status=p["_s"].map(STATUS_TEXT)).drop(columns="_s")
    return (shown.style
            .format({c: "{:.3f}" for c in ["Kamera", "Dipakai model", "Min", "P01", "Median", "P99", "Max"]})
            .apply(lambda col: [STATUS_COLOR[f] for f in flags], subset=["Dipakai model", "Status"]))


def render_side_panel(panel_box, panel, raw, used, vis, pred_cls):
    """Panel pembanding mode samping: sudut utama + rentang dataset (tanpa fitur z/putaran pinggul)."""
    n_out, n_edge = int((panel["_s"] == "luar").sum()), int((panel["_s"] == "tepi").sum())
    cls = ref["by_class"]
    med = lambda c, f: cls.get(c, {}).get(f, {}).get("p50")
    with panel_box.container():
        st.subheader("Panel pembanding: kamera vs rentang dataset (tampak samping)")
        d1, d2, d3, d4 = st.columns(4)
        parts = [f"{c} {med(c, 'trunk_tilt'):+.0f}°" for c in ("TUP", "TLF", "TLB") if med(c, "trunk_tilt") is not None]
        d1.metric("Sudut BADAN (+ = ke depan)", f"{raw['trunk_tilt']:+.0f}°", "dataset: " + ", ".join(parts), delta_color="off")
        d2.metric("Sudut LEHER (+ = ke depan)", f"{raw['neck_tilt']:+.0f}°")
        d3.metric("Sudut telinga-bahu-pinggul", f"{raw['esh_angle']:.0f}°")
        d4.metric("Visibilitas rata-rata", f"{raw['vis_mean']:.2f}")
        if raw["vis_mean"] < 0.6:
            st.warning("Landmark sisi tubuh kurang terlihat (tertutup meja/kursi atau keluar frame). Sudut badan bisa tidak akurat.")
        m1, m2 = st.columns(2)
        m1.metric("Fitur di luar rentang dataset", f"{n_out} / {len(panel)}")
        m2.metric("Fitur di tepi (di luar P01–P99)", f"{n_edge} / {len(panel)}")
        st.dataframe(style_table(panel), hide_index=True, use_container_width=True)


def render_result(box, panel_box, row, vis, smoother, alert_hist, alert_len, stats=None, dt=0.0):
    srow, raw, used, probs, p_non, cal_msg, robust_on = analyze(row, smoother, None if SIDE else min(vis["left_hip"], vis["right_hip"]))
    pred = max(probs, key=probs.get)
    name, icon = LABELS.get(pred, (pred, ""))
    non_tup = p_non >= 0.5                  # logika peringatan: skenario biner TUP vs Non-TUP
    if stats is not None:
        stats.update(pred, non_tup, dt)
    alert_hist.append(non_tup)
    alert = len(alert_hist) == alert_hist.maxlen and np.mean(alert_hist) >= 0.8

    with box.container():
        if cal_msg:
            st.info(cal_msg)
        mc, mt = st.columns([1, 4])
        if pred in MASCOT_OF:
            mc.markdown(mascots.small_html(MASCOT_OF[pred], 96), unsafe_allow_html=True)
        mt.markdown(f'<div class="pg-badge {"ok" if pred == "TUP" else "bad"}">{pred} · {name}</div>', unsafe_allow_html=True)
        mt.caption(f"Keyakinan model: {probs[pred]:.0%}" + ("  ·  model tahan-pinggul aktif" if robust_on else ""))
        if robust_on:
            mt.caption("Miring kiri/kanan paling andal di mode ini. Condong depan/belakang kurang akurat: untuk itu pakai mode Samping.")
        if pred in MASCOT_SAY:
            mt.markdown(f"**{MASCOT_SAY[pred][0]}:** {MASCOT_SAY[pred][1]}")
        for c in sorted(probs, key=probs.get, reverse=True):
            st.progress(float(probs[c]), text=f"{c} ({LABELS.get(c, (c,))[0]}): {probs[c]:.1%}")
        if alert:
            st.warning(f"Postur tidak tegak selama ±{alert_len} detik terakhir. Coba luruskan badan.")
        if stats is not None and stats.total > 0:
            s1, s2, s3 = st.columns(3)
            s1.metric("Waktu tidak tegak", f"{stats.pct_nontup:.0f}%")
            s2.metric("Durasi sesi", f"{stats.total / 60:.1f} mnt")
            s3.metric("Terlama tidak tegak", f"{stats.longest:.0f} dtk")
            if stats.total >= 5:
                st.caption("Waktu per postur: " + ", ".join(
                    f"{k} {100 * v / stats.total:.0f}%" for k, v in sorted(stats.secs.items(), key=lambda kv: -kv[1])))
        info = kb.KNOWLEDGE.get(pred)
        if info:
            with st.expander("Informasi edukatif (berbasis aturan)", expanded=(pred != "TUP")):
                if info["penyebab"]:
                    st.markdown(f"**Kemungkinan penyebab:** {info['penyebab']}")
                    st.markdown(f"**Potensi keluhan jika berlangsung lama:** {info['keluhan']}")
                st.markdown(f"**Saran pencegahan:** {info['saran']}")
                st.caption(kb.DISCLAIMER)
        if SIDE:      # periksa sisi tubuh yang dipakai model (visibility rata-rata tertinggi)
            side = max(("left", "right"), key=lambda s_: sum(vis[f"{s_}_{p}"] for p in pu.SIDE_PARTS))
            low = [f"{side}_{p}" for p in pu.SIDE_PARTS if vis[f"{side}_{p}"] < 0.5]
            hint = ". Pastikan telinga, bahu, pinggul, dan lutut sisi yang menghadap kamera masuk frame dan tidak tertutup meja."
        else:
            low = [n for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip") if vis[n] < 0.5]
            hint = ". Pastikan bahu dan pinggul masuk frame dan pencahayaan cukup."
        if low:
            st.info("Landmark kurang terlihat: " + ", ".join(low) + hint)

    panel = build_panel(raw, used, pred)
    if SIDE:
        render_side_panel(panel_box, panel, raw, used, vis, pred)
        return pred
    n_out, n_edge = int((panel["_s"] == "luar").sum()), int((panel["_s"] == "tepi").sum())
    hr = ref["extras"]["hip_rotation_deg"]
    cam_hip = float(pu.hip_rotation_deg(srow["left_hip_z"], srow["right_hip_z"],
                                        srow["left_hip_x"], srow["right_hip_x"]))
    dz = srow["left_hip_z"] - srow["right_hip_z"]
    hz = ref["extras"]["hip_z_diff"]

    with panel_box.container():
        st.subheader("Panel pembanding: kamera vs rentang dataset")
        eng = ft.engineer_features(pd.DataFrame([srow])).iloc[0]
        tll = ref["by_class"].get("TLL", {}).get("trunk_tilt_lateral", {}).get("p50")
        tlr = ref["by_class"].get("TLR", {}).get("trunk_tilt_lateral", {}).get("p50")
        hip_vis = min(vis["left_hip"], vis["right_hip"])
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Sudut BADAN kiri/kanan", f"{eng['trunk_tilt_lateral']:+.0f}°",
                  f"dataset: TLL {tll:+.0f}°, TLR {tlr:+.0f}°" if tll is not None and tlr is not None else None,
                  delta_color="off")
        d2.metric("Sudut LEHER kiri/kanan", f"{eng['neck_tilt_lateral']:+.0f}°")
        d3.metric("Sudut badan depan/belakang", f"{eng['trunk_tilt_sagittal']:+.0f}°")
        d4.metric("Visibilitas pinggul (terendah)", f"{hip_vis:.2f}")
        if hip_vis < 0.6:
            st.warning("Pinggul kurang terlihat (tertutup meja/kursi atau keluar frame). Sudut BADAN bergantung pada "
                       "posisi pinggul, jadi miring badan bisa tidak terbaca walau kepala/leher terbaca.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fitur di luar rentang dataset", f"{n_out} / {len(panel)}")
        m2.metric("Fitur di tepi (di luar P01–P99)", f"{n_edge} / {len(panel)}")
        m3.metric("Selisih z pinggul (kiri − kanan)", f"{dz:+.3f}", f"dataset median {hz['p50']:+.3f}",
                  delta_color="off")
        m4.metric("Putaran pinggul terhadap kamera", f"{cam_hip:+.1f}°", f"dataset median {hr['p50']:+.1f}°",
                  delta_color="off")
        if abs(cam_hip - hr["p50"]) > 8:
            st.warning("Putaran pinggul dari kamera jauh berbeda dari dataset. Fitur `shoulder_rotation` dan "
                       "fitur `_z` bisa menyimpang walau postur sebenarnya benar. Lihat baris bertanda ⚠️/❌ di bawah.")

        st.markdown("**Fitur rawan** (bergantung pada `z` dan putaran badan)")
        st.dataframe(style_table(panel[panel["Fitur"].isin(pu.RISK_FEATURES)]),
                     hide_index=True, use_container_width=True)
        with st.expander(f"Semua {len(panel)} fitur", expanded=False):
            st.dataframe(style_table(panel), hide_index=True, use_container_width=True, height=420)
    return pred


# ----------------------------------------------------------------- pemeriksaan backend
kind, task_path, ready = pu.backend_status(complexity, str(APP_DIR))
if not ready:
    st.error(f"Versi MediaPipe terpasang memakai Tasks API dan butuh file model "
             f"`{Path(task_path).name}` di folder aplikasi.")
    if st.button("Unduh model sekarang"):
        with st.spinner("Mengunduh..."):
            pu.download_task_model(complexity, str(APP_DIR))
        st.rerun()
    st.stop()


# ----------------------------------------------------------------- mode webcam langsung
def ice_servers():
    """STUN Google + (opsional) server TURN dari st.secrets["turn"]. TURN sering diperlukan saat di-hosting (lihat HOSTING.md)."""
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    try:
        t = st.secrets["turn"]
        urls = t["urls"]
        servers.append({"urls": [urls] if isinstance(urls, str) else list(urls), "username": t["username"], "credential": t["credential"]})
    except Exception:       # tidak ada secrets / bagian [turn]: pakai STUN saja
        pass
    return servers


if mode == "Webcam langsung":
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
    import av

    class PoseProcessor(VideoProcessorBase):
        """recv() hanya menayangkan video (cepat). MediaPipe jalan di thread terpisah (AsyncPose)."""
        def __init__(self, complexity, det_conf, model_dir, side=False):
            self.flip_input = False
            self.mirror_display = True
            self.worker = pu.AsyncPose(lambda: pu.PoseEstimator(
                complexity, det_conf, 0.5, live=True, model_dir=model_dir),
                row_fn=pu.world_to_side_row if side else None)

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")                 # frame ASLI dari kamera
            if self.flip_input:                                    # hanya koreksi darurat
                img = cv2.flip(img, 1)
            self.worker.push(img)                                  # inferensi pada frame terbaru saja
            disp = img.copy()
            res = self.worker.result()
            if res is not None and res[2] is not None and time.time() - res[3] < 0.5:
                pu.draw_pose(disp, res[2])
            if self.mirror_display:                                # cermin HANYA untuk tampilan
                disp = cv2.flip(disp, 1)
            return av.VideoFrame.from_ndarray(disp, format="bgr24")

        def latest(self):
            res = self.worker.result()
            if res is None or res[0] is None or time.time() - res[3] > 1.0:
                return None
            return res[0], res[1], res[3]

        def on_ended(self):
            self.worker.close()

    col_v, col_r = st.columns([3, 2])
    with col_v:
        ctx = webrtc_streamer(
            key=f"posture_{view}", mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": ice_servers()},
            media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": {"ideal": 20}}, "audio": False},
            video_processor_factory=lambda: PoseProcessor(complexity, det_conf, str(APP_DIR), SIDE),
            async_processing=True)
    result_box = col_r.empty()
    with st.expander("Detail teknis: perbandingan fitur dengan dataset (untuk analisis)", expanded=False):
        panel_box = st.empty()

    smoother = pu.Smoother(smooth_n)
    step = 0.15
    alert_len = alert_sec
    alert_hist = deque(maxlen=max(2, int(alert_sec / step)))
    last_t = time.monotonic()
    while ctx.state.playing:
        now = time.monotonic()
        dt, last_t = now - last_t, now
        proc = ctx.video_processor
        if proc is not None:
            proc.flip_input, proc.mirror_display = flip_input, mirror_display
            snap = proc.latest()
            if proc.worker.error:
                result_box.error(f"MediaPipe gagal: {proc.worker.error}")
            elif snap is None:
                result_box.info("Pose belum terdeteksi. " + ("Duduk menyamping ke kamera dengan seluruh sisi badan terlihat." if SIDE else "Duduk menghadap kamera dengan bahu dan pinggul terlihat."))
            else:
                render_result(result_box, panel_box, snap[0], snap[1], smoother, alert_hist, alert_len,
                              st.session_state[SK], dt)
        time.sleep(step)
    if not ctx.state.playing:
        result_box.info("Klik START untuk menyalakan kamera. Izinkan akses kamera di browser.")

# ----------------------------------------------------------------- mode foto
else:
    tab_cam, tab_up = st.tabs(["Ambil foto", "Unggah gambar"])
    with tab_cam:
        shot = st.camera_input("Duduk menyamping, telinga-bahu-pinggul-lutut terlihat" if SIDE else "Duduk menghadap kamera, bahu dan pinggul terlihat")
    with tab_up:
        up = st.file_uploader("Gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    data = shot or up
    if data is not None:
        img = cv2.imdecode(np.frombuffer(data.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        src = cv2.flip(img, 1) if flip_input else img
        world, image = static_estimator(complexity, det_conf).process(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        col_v, col_r = st.columns([3, 2])
        if world is None:
            col_r.error("Pose tidak terdeteksi pada gambar ini.")
        else:
            row, vis = (pu.world_to_side_row(world, image, (src.shape[1], src.shape[0]))
                        if SIDE else pu.world_to_row(world))
            disp = pu.draw_pose(src.copy(), image)
            col_v.image(cv2.cvtColor(cv2.flip(disp, 1) if mirror_display else disp, cv2.COLOR_BGR2RGB),
                        channels="RGB")
            with st.expander("Detail teknis: perbandingan fitur dengan dataset (untuk analisis)", expanded=False):
                pbox = st.empty()
            render_result(col_r.empty(), pbox, row, vis, pu.Smoother(1),
                          deque(maxlen=2), alert_sec)
    else:
        st.info("Ambil foto atau unggah gambar untuk dianalisis.")
