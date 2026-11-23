"""
features_side.py  -  fitur untuk model TAMPAK SAMPING (Dataset 2, kelas TUP/TLF/TLB).

Dipakai DUA kali agar konsisten: (1) saat training (train_side.py) dan (2) di aplikasi mode samping.
Prinsip: bidang gambar (x horizontal, y vertikal ke bawah) adalah bidang sagital orang yang duduk menyamping.
 - Sisi tubuh dipilih dari visibility rata-rata (telinga, bahu, pinggul, lutut) tertinggi.
 - Arah hadap = tanda (hidung - telinga) pada sumbu x; semua fitur horizontal dikalikan arah hadap
   sehingga nilai POSITIF selalu berarti "ke depan" (tidak terpengaruh flip / orang menghadap kiri atau kanan).
 - Jarak dinormalisasi dengan panjang torso (bahu-pinggul) -> tidak bergantung ukuran orang / jarak kamera.
"""
import numpy as np
import pandas as pd

SIDE_FEATURES = [
    "trunk_tilt",        # sudut torso (pinggul->bahu) terhadap vertikal, + = condong ke depan (derajat)
    "neck_tilt",         # sudut leher (bahu->telinga) terhadap vertikal, + = ke depan (derajat)
    "esh_angle",         # sudut telinga-bahu-pinggul di bahu (derajat); 180 = lurus
    "hip_angle",         # sudut bahu-pinggul-lutut di pinggul (derajat)
    "thigh_tilt",        # sudut paha (pinggul->lutut) terhadap horizontal ke depan (derajat)
    "ear_off_x",         # offset horizontal telinga terhadap bahu / panjang torso, + = ke depan
    "nose_off_x",        # offset horizontal hidung terhadap bahu / panjang torso, + = ke depan
    "ear_off_y",         # offset vertikal telinga terhadap bahu / panjang torso (negatif = di atas bahu)
    "vis_mean",          # rata-rata visibility landmark yang dipakai (indikator kualitas deteksi)
]
PARTS = ["ear", "shoulder", "hip", "knee"]


def _xy(df, name, space, bw, bh):
    if space == "world":
        return np.c_[df[f"{name}_x"].to_numpy(float), df[f"{name}_y"].to_numpy(float)]
    # ruang citra: skala piksel agar sudut tidak terdistorsi aspek crop/frame
    return np.c_[df[f"{name}_ix"].to_numpy(float) * bw, df[f"{name}_iy"].to_numpy(float) * bh]


def _angle(a, b, c):
    """Sudut di titik b (derajat) antara vektor b->a dan b->c."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def engineer_side_features(df, space="image", box_w=None, box_h=None):
    """df: kolom landmark hasil extract_side.py (atau dari aplikasi). space: 'image' atau 'world'."""
    if space == "world":                       # ruang dunia tidak butuh ukuran crop/frame
        bw = bh = np.ones(len(df))
    else:
        if box_w is None and "box_w" not in df.columns:
            raise ValueError("space='image' butuh kolom box_w/box_h (ukuran crop atau frame) dan koordinat *_ix/*_iy.")
        bw = df["box_w"].to_numpy(float) if box_w is None else np.asarray(box_w, float)
        bh = df["box_h"].to_numpy(float) if box_h is None else np.asarray(box_h, float)
    bw, bh = (bw[:, None] if np.ndim(bw) else bw), (bh[:, None] if np.ndim(bh) else bh)
    vis_l = np.mean([df[f"left_{p}_vis"].to_numpy(float) for p in PARTS], axis=0)
    vis_r = np.mean([df[f"right_{p}_vis"].to_numpy(float) for p in PARTS], axis=0)
    use_left = vis_l >= vis_r

    def pick(part):
        L = _xy(df, f"left_{part}", space, bw, bh)
        R = _xy(df, f"right_{part}", space, bw, bh)
        return np.where(use_left[:, None], L, R)

    ear, sh, hip, knee = pick("ear"), pick("shoulder"), pick("hip"), pick("knee")
    nose = _xy(df, "nose", space, bw, bh)
    d = np.sign(nose[:, 0] - ear[:, 0]); d[d == 0] = 1.0           # arah hadap (+1 = kanan layar)
    torso = sh - hip
    tl = np.linalg.norm(torso, axis=1) + 1e-9
    neck, thigh = ear - sh, knee - hip
    out = pd.DataFrame({
        "trunk_tilt": np.degrees(np.arctan2(d * torso[:, 0], -torso[:, 1])),
        "neck_tilt": np.degrees(np.arctan2(d * neck[:, 0], -neck[:, 1])),
        "esh_angle": _angle(ear, sh, hip),
        "hip_angle": _angle(sh, hip, knee),
        "thigh_tilt": np.degrees(np.arctan2(thigh[:, 1], d * thigh[:, 0])),
        "ear_off_x": d * (ear[:, 0] - sh[:, 0]) / tl,
        "nose_off_x": d * (nose[:, 0] - sh[:, 0]) / tl,
        "ear_off_y": (ear[:, 1] - sh[:, 1]) / tl,
        "vis_mean": np.where(use_left, vis_l, vis_r),
    }, index=df.index)
    return out[SIDE_FEATURES]
