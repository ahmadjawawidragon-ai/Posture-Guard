"""Tes mode TAMPAK SAMPING tanpa kamera & tanpa Streamlit.
 1) landmark -> world_to_side_row -> fitur aplikasi  ==  fitur train_side.py (harus identik)
 2) gambar nyata -> PoseEstimator -> baris -> fitur -> model -> kelas (uji ujung ke ujung, frame utuh seperti kamera)
Pakai:  python test_side.py --root "Sitting Posture Detection.v2i.coco"   (bagian 2 dilewati bila --root tidak diberikan)
"""
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import argparse, json, os, types
import cv2, joblib, numpy as np, pandas as pd
import features_side as fsd, pose_utils as pu
from extract_side import NAMES

ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--n", type=int, default=80); a = ap.parse_args()
bundle = joblib.load(ROOT / "posture_model_side.joblib"); names = bundle["feature_names"]

# ---- 1) kesetaraan fitur aplikasi vs training
df = pd.read_csv(ROOT / "data" / "landmarks_side.csv"); df = df[df["detected"] & ~df["conflict"]].sample(60, random_state=0)
LM = types.SimpleNamespace
worst = 0.0
for _, r in df.iterrows():
    world = [LM(x=r[f"{n}_x"], y=r[f"{n}_y"], z=r[f"{n}_z"], visibility=r[f"{n}_vis"]) for n in NAMES]
    row, vis = pu.world_to_side_row(world)
    f_app = pu.side_feature_frame(row, names, bundle["feature_space"]).iloc[0].to_numpy(float)
    f_trn = fsd.engineer_side_features(r.to_frame().T.astype(float, errors="ignore"), "world")[names].iloc[0].to_numpy(float)
    worst = max(worst, float(np.abs(f_app - f_trn).max()))
assert worst < 1e-9, worst
print(f"OK 1: fitur aplikasi identik dengan train_side.py pada 60 sampel (selisih maks {worst:.1e})")

# ---- 2) ujung ke ujung dengan gambar nyata (frame utuh)
if a.root:
    est = pu.PoseEstimator(1, 0.5, 0.5, live=False, model_dir=str(ROOT))
    inv = {"good_posture": "TUP", "leaning_forward": "TLF", "leaning_backward": "TLB"}
    ok = tot = miss = 0; conf = {}
    for split in ["valid", "test"]:
        d = json.load(open(os.path.join(a.root, split, "_annotations.coco.json")))
        cat = {c["id"]: c["name"] for c in d["categories"]}; per = {}
        for an in d["annotations"]: per.setdefault(an["image_id"], []).append(an)
        for im in d["images"]:
            ans = per.get(im["id"], [])
            if len(ans) != 1 or cat[ans[0]["category_id"]] not in inv or tot + miss >= a.n: continue
            img = cv2.imread(os.path.join(a.root, split, im["file_name"]))
            world, _ = est.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if world is None: miss += 1; continue
            row, vis = pu.world_to_side_row(world)
            X = pu.side_feature_frame(row, names, bundle["feature_space"])
            probs = dict(zip(bundle["classes"], bundle["model"].predict_proba(X)[0]))
            pred, true = max(probs, key=probs.get), inv[cat[ans[0]["category_id"]]]
            ok += pred == true; tot += 1; conf[(true, pred)] = conf.get((true, pred), 0) + 1
    print(f"OK 2: {tot} gambar diproses ujung ke ujung ({miss} gagal deteksi); benar {ok}/{tot} = {ok/tot:.0%}")
    print("     (model tersimpan dilatih ulang dengan SEMUA data, jadi angka ini hanya uji kelancaran pipeline, bukan hasil evaluasi)")
    print("     ", dict(sorted(conf.items())))
