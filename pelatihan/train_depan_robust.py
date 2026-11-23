"""
train_depan_robust.py : model Depan yang TAHAN terhadap posisi pinggul yang tidak andal (pinggul tertutup meja, ditebak MediaPipe).

Masalah yang ditemukan: model biasa (posture_model.joblib) sangat bergantung pada pinggul sebagai titik acuan. Bila pinggul dari
webcam laptop tidak andal, titik atas tubuh bergeser terhadap pinggul dan hampir semua postur salah terbaca TUP.
Solusi: augmentasi saat pelatihan. Tiap sampel latih diperbanyak dengan pinggul yang "ikut" bahu (geser lateral bx dan kedalaman bz,
acak 0..1) ditambah derau kecil, sehingga model belajar memakai petunjuk kepala-bahu.

Jalankan dari folder proyek:  python pelatihan/train_depan_robust.py --csv "data/Data ML.csv"
Keluaran: posture_model_robust.joblib (dipakai aplikasi), hasil/depan/uji_ketahanan.csv
"""
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import argparse, json, warnings
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, recall_score
import features as ft
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=str(ROOT / "data" / "Data ML.csv"))
ap.add_argument("--base-model", default=str(ROOT / "posture_model.joblib"))
ap.add_argument("--model-out", default=str(ROOT / "posture_model_robust.joblib"))
ap.add_argument("--out", default=str(ROOT / "hasil" / "depan"))
ap.add_argument("--copies", type=int, default=3)
ap.add_argument("--trees", type=int, default=300)
ap.add_argument("--skip-cv", action="store_true")
a = ap.parse_args()
SEED = 42; rng = np.random.default_rng(SEED)

df = pd.read_csv(a.csv).reset_index(drop=True)
base = joblib.load(a.base_model); names = base["feature_names"]; classes = list(base["classes"])
y = base["label_encoder"].transform(df["upperbody_label"]) if "label_encoder" in base else np.array([classes.index(c) for c in df["upperbody_label"]])
g = df["subject"].values


def corrupt(d, bx, bz, noise=0.0):
    """Pinggul 'ikut' bahu: pusat pinggul bergeser bx (lateral) dan bz (kedalaman) kali titik tengah bahu; titik atas dipusatkan ulang."""
    d = d.copy(); n = len(d)
    mx = ((d.left_shoulder_x + d.right_shoulder_x) / 2).values; mz = ((d.left_shoulder_z + d.right_shoulder_z) / 2).values
    bx = np.broadcast_to(bx, n); bz = np.broadcast_to(bz, n)
    for lm in ft.LANDMARKS:
        if "hip" in lm:
            continue
        d[lm + "_x"] = d[lm + "_x"].values - bx * mx + rng.normal(0, noise, n)
        d[lm + "_z"] = d[lm + "_z"].values - bz * mz + rng.normal(0, noise, n)
        d[lm + "_y"] = d[lm + "_y"].values + rng.normal(0, noise / 2, n)
    return d


def X(d):
    e = ft.engineer_features(d)
    return pd.concat([d[ft.RAW_COLS].reset_index(drop=True), e.reset_index(drop=True)], axis=1)[names]


def augmented(d, yy):
    parts, ys = [d], [yy]
    for _ in range(a.copies):
        parts.append(corrupt(d, rng.uniform(0, 1, len(d)), rng.uniform(0, 1, len(d)), noise=0.015)); ys.append(yy)
    return pd.concat([X(p) for p in parts]), np.concatenate(ys)


def rf():
    return RandomForestClassifier(a.trees, class_weight="balanced", random_state=SEED, n_jobs=1)


rows = []
if not a.skip_cv:
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    conds = {"pinggul normal": (0, 0), "pinggul ikut bahu 50%": (0.5, 0.5), "pinggul ikut bahu 100%": (1.0, 1.0)}
    pred = {m: {c: np.zeros(len(y), int) for c in conds} for m in ("biasa", "robust")}
    for tr, te in cv.split(df, y, g):
        m_b = rf().fit(X(df.iloc[tr]), y[tr]); Xa, ya = augmented(df.iloc[tr], y[tr]); m_r = rf().fit(Xa, ya)
        for c, (bx, bz) in conds.items():
            Xt = X(corrupt(df.iloc[te], bx, bz) if bx else df.iloc[te])
            pred["biasa"][c][te] = m_b.predict(Xt); pred["robust"][c][te] = m_r.predict(Xt)
    for m in pred:
        for c in conds:
            r = recall_score(y, pred[m][c], labels=range(len(classes)), average=None)
            rows.append({"model": "Random Forest biasa (sekarang)" if m == "biasa" else "Random Forest tahan-pinggul (augmentasi)", "kondisi": c,
                         "f1_makro": f1_score(y, pred[m][c], average="macro"), **{f"recall_{k}": v for k, v in zip(classes, r)}})
    t = pd.DataFrame(rows); t.round(3).to_csv(f"{a.out}/uji_ketahanan.csv", index=False); print(t.round(3).to_string(index=False))

Xa, ya = augmented(df, y)
final = rf().fit(Xa, ya)
bundle = {k: v for k, v in base.items() if k != "model"}
bundle.update({"model": final, "model_name": "Random Forest tahan-pinggul (augmentasi)", "robust": True})
bundle.pop("binary_model", None)
joblib.dump(bundle, a.model_out); print("Tersimpan:", a.model_out, "| kunci:", sorted(bundle))
