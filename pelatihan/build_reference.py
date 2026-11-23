"""
build_reference.py
Hitung rentang nilai fitur dataset (min, persentil, median; total dan per kelas)
untuk panel pembanding di aplikasi.

Jalankan SEKALI (CSV tidak perlu dikirim bersama aplikasi):
    python build_reference.py --csv "C:\\Users\\Thinkpad\\Documents\\Data ML.csv"

Hasil: reference_stats.json (satu folder dengan app.py)
"""
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import argparse
import json

import pandas as pd

import features as ft
import pose_utils as pu


def build(df, feature_set="C_terpilih+sudut", label_col="upperbody_label"):
    X = ft.build_feature_sets(df)[feature_set]
    labels = df[label_col]
    out = {
        "meta": {"n_rows": int(len(df)), "feature_set": feature_set,
                 "n_subjects": int(df["subject"].nunique()) if "subject" in df else None},
        "features": {c: pu.describe_series(X[c]) for c in X.columns},
        "by_class": {},
        "extras": {},
    }
    for cls, idx in labels.groupby(labels).groups.items():
        sub = X.loc[idx]
        out["by_class"][str(cls)] = {c: {k: v for k, v in pu.describe_series(sub[c]).items()
                                         if k in ("p05", "p50", "p95")} for c in X.columns}
    hip_rot = pd.Series(pu.hip_rotation_deg(df["left_hip_z"], df["right_hip_z"],
                                            df["left_hip_x"], df["right_hip_x"]), index=df.index)
    out["extras"]["hip_rotation_deg"] = pu.describe_series(hip_rot)
    out["extras"]["hip_z_diff"] = pu.describe_series(df["left_hip_z"] - df["right_hip_z"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--feature-set", default="C_terpilih+sudut",
                    help="harus sama dengan FEATURE_SET di notebook")
    ap.add_argument("--out", default=str(ROOT / "reference_stats.json"))
    a = ap.parse_args()
    data = pd.read_csv(a.csv)
    ref = build(data, a.feature_set)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=1)
    hr = ref["extras"]["hip_rotation_deg"]
    print(f"Tersimpan {a.out}: {ref['meta']['n_rows']} baris, {len(ref['features'])} fitur")
    print(f"Putaran pinggul dataset: median {hr['p50']:.1f} derajat "
          f"(P01 {hr['p01']:.1f}, P99 {hr['p99']:.1f})")
