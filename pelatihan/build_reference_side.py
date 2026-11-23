"""
build_reference_side.py - rentang nilai fitur Dataset 2 (tampak samping) untuk panel pembanding di aplikasi.
Jalankan SEKALI:  python build_reference_side.py --csv landmarks_side.csv     ->  reference_stats_side.json
"""
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import argparse, json
import pandas as pd
import features_side as fs
import pose_utils as pu

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=str(ROOT / "data" / "landmarks_side.csv")); ap.add_argument("--space", default="world"); ap.add_argument("--out", default=str(ROOT / "reference_stats_side.json"))
a = ap.parse_args()
raw = pd.read_csv(a.csv); df = raw[raw["detected"] & ~raw["conflict"]].reset_index(drop=True)
X, y = fs.engineer_side_features(df, a.space), df["label"]
ref = {"meta": {"n_rows": int(len(df)), "feature_set": f"side_{a.space}", "n_subjects": None},
       "features": {c: pu.describe_series(X[c]) for c in X.columns}, "by_class": {}, "extras": {}}
for cls, idx in y.groupby(y).groups.items():
    ref["by_class"][str(cls)] = {c: {k: v for k, v in pu.describe_series(X.loc[idx, c]).items() if k in ("p05", "p50", "p95")} for c in X.columns}
json.dump(ref, open(a.out, "w"), indent=1)
print(f"Tersimpan {a.out}: {len(df)} baris, {len(X.columns)} fitur")
