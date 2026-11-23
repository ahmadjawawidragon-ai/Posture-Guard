"""
features.py
Fungsi fitur untuk project klasifikasi postur duduk (MultiPosture + MediaPipe).

File ini dipakai DUA kali supaya konsisten:
  1) di notebook training (EDA + modeling), dan
  2) di aplikasi (Streamlit/Gradio) saat memproses landmark dari kamera.

Asumsi koordinat (sesuai data.csv MultiPosture):
  - titik nol di pusat pinggul, x = kiri-kanan, y = atas-bawah (y negatif = di atas pinggul),
    z = kedalaman (negatif = lebih dekat ke kamera).
  - kamera di depan pengguna.
"""
import numpy as np
import pandas as pd

# Landmark yang relevan untuk postur batang tubuh dan kepala
LANDMARKS = ["nose", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder",
             "left_hip", "right_hip"]

RAW_COLS = [f"{lm}_{ax}" for lm in LANDMARKS for ax in ("x", "y", "z")]

ENG_FEATURES = [
    "trunk_tilt_lateral",    # kemiringan batang tubuh kiri/kanan (derajat, bidang x-y)
    "trunk_tilt_sagittal",   # kemiringan batang tubuh depan/belakang (derajat, bidang z-y)
    "neck_tilt_lateral",     # kemiringan leher-kepala kiri/kanan (bahu -> hidung)
    "neck_tilt_sagittal",    # kemiringan leher-kepala depan/belakang
    "shoulder_slope",        # kemiringan garis bahu (derajat)
    "shoulder_rotation",     # putaran bahu terhadap kamera (derajat, dari selisih z)
    "head_offset_x",         # posisi hidung terhadap titik tengah bahu, dinormalisasi lebar bahu
    "head_offset_y",
    "head_offset_z",
    "ear_offset_x",          # posisi titik tengah telinga terhadap titik tengah bahu
    "ear_offset_z",
    "shoulder_width",        # lebar bahu (faktor skala)
    "trunk_length",          # panjang batang tubuh (faktor skala)
]


def _pt(df, name):
    return df[[f"{name}_x", f"{name}_y", f"{name}_z"]].to_numpy(dtype=float)


def _tilt_deg(horizontal, vertical):
    """Sudut vektor terhadap sumbu vertikal ke atas (y negatif = atas)."""
    return np.degrees(np.arctan2(horizontal, -vertical))


def engineer_features(df):
    """Hitung fitur turunan (sudut dan offset) dari koordinat landmark."""
    nose = _pt(df, "nose")
    l_ear, r_ear = _pt(df, "left_ear"), _pt(df, "right_ear")
    l_sh, r_sh = _pt(df, "left_shoulder"), _pt(df, "right_shoulder")
    l_hip, r_hip = _pt(df, "left_hip"), _pt(df, "right_hip")

    mid_sh = (l_sh + r_sh) / 2
    mid_hip = (l_hip + r_hip) / 2
    mid_ear = (l_ear + r_ear) / 2

    trunk = mid_sh - mid_hip          # pinggul -> bahu
    head = nose - mid_sh              # bahu -> hidung
    ear = mid_ear - mid_sh            # bahu -> telinga
    sh_vec = l_sh - r_sh              # garis bahu

    eps = 1e-9
    sh_width = np.linalg.norm(sh_vec, axis=1) + eps
    trunk_len = np.linalg.norm(trunk, axis=1) + eps

    out = pd.DataFrame({
        "trunk_tilt_lateral": _tilt_deg(trunk[:, 0], trunk[:, 1]),
        "trunk_tilt_sagittal": _tilt_deg(trunk[:, 2], trunk[:, 1]),
        "neck_tilt_lateral": _tilt_deg(head[:, 0], head[:, 1]),
        "neck_tilt_sagittal": _tilt_deg(head[:, 2], head[:, 1]),
        "shoulder_slope": np.degrees(np.arctan2(sh_vec[:, 1], np.abs(sh_vec[:, 0]) + eps)),
        "shoulder_rotation": np.degrees(np.arctan2(sh_vec[:, 2], np.abs(sh_vec[:, 0]) + eps)),
        "head_offset_x": head[:, 0] / sh_width,
        "head_offset_y": head[:, 1] / sh_width,
        "head_offset_z": head[:, 2] / sh_width,
        "ear_offset_x": ear[:, 0] / sh_width,
        "ear_offset_z": ear[:, 2] / sh_width,
        "shoulder_width": sh_width,
        "trunk_length": trunk_len,
    }, index=df.index)
    return out[ENG_FEATURES]


def build_feature_sets(df):
    """Kembalikan beberapa kombinasi fitur untuk dibandingkan."""
    all_coords = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
    eng = engineer_features(df)
    return {
        "A_semua_koordinat": df[all_coords].copy(),
        "B_landmark_terpilih": df[RAW_COLS].copy(),
        "C_terpilih+sudut": pd.concat([df[RAW_COLS], eng], axis=1),
        "D_sudut_saja": eng.copy(),
    }
