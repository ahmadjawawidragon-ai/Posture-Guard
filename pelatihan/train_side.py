"""
train_side.py  -  Dataset 2 (tampak samping): EDA ringkas, preprocessing, 5 model, nested CV, evaluasi, simpan model.
Jalankan setelah extract_side.py:   python train_side.py --csv landmarks_side.csv
Keluaran: folder hasil_side/ (tabel CSV, gambar PNG, results.json) dan posture_model_side.joblib
"""
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import argparse, json, os, time, warnings
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, recall_score as rs)
import features_side as fs
warnings.filterwarnings("ignore")
SEED = 42
ap = argparse.ArgumentParser(); ap.add_argument("--csv", default=str(ROOT / "data" / "landmarks_side.csv")); ap.add_argument("--out", default=str(ROOT / "hasil" / "side"))
ap.add_argument("--model-out", default=str(ROOT / "posture_model_side.joblib")); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# ---------------------------------------------------------------- 1. data & deteksi
raw = pd.read_csv(a.csv)
det = raw.groupby(["split", "label"]).agg(anotasi=("label", "size"), terdeteksi=("detected", "sum"), konflik=("conflict", "sum")).reset_index()
det["gagal_deteksi"] = det["anotasi"] - det["terdeteksi"]
det.to_csv(f"{a.out}/01_deteksi_per_kelas.csv", index=False)
one = raw[raw["full_detected"].notna()]
cmp = {"crop_box_(citra_1_box)": float(one["detected"].mean()), "citra_utuh_(citra_1_box)": float(one["full_detected"].astype(bool).mean()), "n": int(len(one))}
df = raw[raw["detected"] & ~raw["conflict"]].reset_index(drop=True)
print("Sampel dipakai:", len(df), "dari", len(raw)); print(det.to_string(index=False)); print("Deteksi crop vs utuh:", cmp)

# ---------------------------------------------------------------- 2. pilih ruang fitur (image vs world) dengan RF + CV grup
tr, ho = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] != "train"].reset_index(drop=True)
le = LabelEncoder().fit(["TLB", "TLF", "TUP"])
cv_out = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
def cv_f1(model, X, y, g):
    s = []
    for tri, tei in cv_out.split(X, y, g):
        m = model.fit(X.iloc[tri], y[tri]); s.append(f1_score(y[tei], m.predict(X.iloc[tei]), average="macro"))
    return float(np.mean(s))
ytr, yho = le.transform(tr["label"]), le.transform(ho["label"])
space_res = {sp: cv_f1(RandomForestClassifier(300, class_weight="balanced", random_state=SEED), fs.engineer_side_features(tr, sp), ytr, tr["group"]) for sp in ["image", "world"]}
SPACE = max(space_res, key=space_res.get); print("Ruang fitur:", space_res, "->", SPACE)
Xtr, Xho = fs.engineer_side_features(tr, SPACE), fs.engineer_side_features(ho, SPACE)

# ---------------------------------------------------------------- 3. EDA (gambar)
sns.set_theme(style="whitegrid")
dist = df.groupby(["split", "label"]).size().unstack(0).fillna(0)[["train", "valid", "test"]]
dist.plot(kind="bar", figsize=(6, 3.6)); plt.title("Distribusi kelas Dataset 2 per split (setelah seleksi deteksi)"); plt.ylabel("jumlah sampel"); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(f"{a.out}/eda_distribusi_kelas.png", dpi=130); plt.close()
fig, axs = plt.subplots(3, 3, figsize=(11, 8))
Xall = pd.concat([Xtr, Xho]); lab = pd.concat([tr["label"], ho["label"]]).values
for ax, c in zip(axs.ravel(), fs.SIDE_FEATURES):
    sns.boxplot(x=lab, y=Xall[c], order=["TUP", "TLF", "TLB"], ax=ax, hue=lab, legend=False); ax.set_title(c); ax.set_xlabel("")
plt.tight_layout(); plt.savefig(f"{a.out}/eda_fitur_per_kelas.png", dpi=120); plt.close()
plt.figure(figsize=(6.5, 5)); sns.heatmap(Xall.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7}); plt.title("Korelasi antar fitur"); plt.tight_layout(); plt.savefig(f"{a.out}/eda_korelasi.png", dpi=130); plt.close()
Xall.assign(label=lab).groupby("label").agg(["mean", "std"]).round(3).T.to_csv(f"{a.out}/02_statistik_fitur_per_kelas.csv")
Xall.describe().T.round(3).to_csv(f"{a.out}/02_statistik_deskriptif.csv")

# ---------------------------------------------------------------- 4. model & grid
try:
    from xgboost import XGBClassifier
    gb_name, gb = "XGBoost", (XGBClassifier(eval_metric="mlogloss", random_state=SEED, n_jobs=1), {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]})
except Exception:
    gb_name, gb = "Gradient Boosting (pengganti XGBoost)", (HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced"), {"max_iter": [200, 400], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]})
MODELS = {
    "Logistic Regression": (Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(class_weight="balanced", max_iter=3000))]), {"m__C": [0.1, 1, 10]}),
    "Random Forest": (RandomForestClassifier(class_weight="balanced", random_state=SEED), {"n_estimators": [200, 400], "max_depth": [None, 6, 10], "min_samples_leaf": [1, 3]}),
    "SVM (RBF)": (Pipeline([("sc", StandardScaler()), ("m", SVC(class_weight="balanced", probability=True, random_state=SEED))]), {"m__C": [1, 10], "m__gamma": ["scale", 0.1]}),
    gb_name: gb,
    "MLP": (Pipeline([("sc", StandardScaler()), ("m", MLPClassifier((32, 16), max_iter=1500, random_state=SEED))]), {"m__alpha": [1e-3, 1e-2]}),
}
def search(est, grid):
    return GridSearchCV(est, grid, scoring="f1_macro", cv=StratifiedGroupKFold(3, shuffle=True, random_state=SEED), n_jobs=1)

# ---------------------------------------------------------------- 5. nested CV (train) + evaluasi holdout (valid+test)
rows, preds_ho, best_params, fitted = [], {}, {}, {}
classes = list(le.classes_)
for name, (est, grid) in MODELS.items():
    t0 = time.time(); fold_f1, oof = [], np.zeros(len(ytr), int)
    for tri, tei in cv_out.split(Xtr, ytr, tr["group"]):
        gs = search(est, grid).fit(Xtr.iloc[tri], ytr[tri], groups=tr["group"].iloc[tri])
        oof[tei] = gs.predict(Xtr.iloc[tei]); fold_f1.append(f1_score(ytr[tei], oof[tei], average="macro"))
    gs = search(est, grid).fit(Xtr, ytr, groups=tr["group"])
    best_params[name], fitted[name] = gs.best_params_, gs.best_estimator_
    t1 = time.time(); p = gs.predict(Xho); ms = (time.time() - t1) / len(Xho) * 1000
    preds_ho[name] = p
    r = dict(model=name, cv_f1_macro_mean=np.mean(fold_f1), cv_f1_macro_std=np.std(fold_f1),
             train_f1_macro=f1_score(ytr, gs.predict(Xtr), average="macro"),
             ho_accuracy=accuracy_score(yho, p), ho_precision_macro=precision_score(yho, p, average="macro"),
             ho_recall_macro=recall_score(yho, p, average="macro"), ho_f1_macro=f1_score(yho, p, average="macro"),
             ms_per_sample=ms)
    for k, c in enumerate(le.transform(classes)):
        r[f"ho_recall_{le.classes_[k]}"] = recall_score(yho, p, labels=[c], average="macro")
    rows.append(r); print(f"{name:40s} CV F1={r['cv_f1_macro_mean']:.3f}±{r['cv_f1_macro_std']:.3f}  HO F1={r['ho_f1_macro']:.3f}  ({time.time()-t0:.0f}s)")
res = pd.DataFrame(rows); res.round(4).to_csv(f"{a.out}/03_perbandingan_model.csv", index=False)

best = res.sort_values("cv_f1_macro_mean", ascending=False).iloc[0]["model"]   # dipilih dari CV pada data latih, BUKAN dari holdout
print("Model terbaik (berdasar CV F1 makro):", best)

# ---------------------------------------------------------------- 6. gambar hasil
fig, axs = plt.subplots(1, len(MODELS), figsize=(3.2 * len(MODELS), 3.2))
for ax, (n, p) in zip(axs, preds_ho.items()):
    cm = confusion_matrix(yho, p, labels=range(3)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(n.split(" (")[0], fontsize=9); ax.set_xlabel("prediksi"); ax.set_ylabel("aktual")
plt.tight_layout(); plt.savefig(f"{a.out}/confusion_matrix_semua_model.png", dpi=130); plt.close()
x = np.arange(len(res)); plt.figure(figsize=(8, 3.8))
plt.bar(x - 0.2, res["cv_f1_macro_mean"], 0.4, yerr=res["cv_f1_macro_std"], label="CV (latih, bersarang)"); plt.bar(x + 0.2, res["ho_f1_macro"], 0.4, label="Holdout (valid+uji)")
plt.xticks(x, [m.split(" (")[0] for m in res["model"]], rotation=15); plt.ylabel("F1 makro"); plt.ylim(0, 1); plt.legend(); plt.title("Perbandingan model tampak samping"); plt.tight_layout(); plt.savefig(f"{a.out}/perbandingan_model.png", dpi=130); plt.close()
rf = fitted["Random Forest"]
pd.Series(rf.feature_importances_, index=fs.SIDE_FEATURES).sort_values().plot(kind="barh", figsize=(6, 3.5), title="Feature importance (Random Forest)"); plt.tight_layout(); plt.savefig(f"{a.out}/feature_importance_rf.png", dpi=130); plt.close()

# ---------------------------------------------------------------- 7. analisis kesalahan + simpan model
bp = preds_ho[best]; err = ho.assign(pred=le.inverse_transform(bp))[bp != yho]
err[["split", "file", "group", "label", "pred"]].to_csv(f"{a.out}/04_kesalahan_prediksi_terbaik.csv", index=False)
final = fitted[best].set_params(**{}) if False else search(*MODELS[best]).fit(pd.concat([Xtr, Xho]), np.r_[ytr, yho], groups=pd.concat([tr["group"], ho["group"]])).best_estimator_
joblib.dump({"model": final, "label_encoder": le, "classes": classes, "feature_names": fs.SIDE_FEATURES, "feature_space": SPACE,
             "model_name": best, "view": "side", "feature_set": f"side_{SPACE}"}, a.model_out)
json.dump({"n_dipakai": len(df), "n_total": len(raw), "n_train": len(tr), "n_holdout": len(ho), "deteksi_crop_vs_utuh": cmp,
           "ruang_fitur_cv_f1": space_res, "ruang_fitur_dipilih": SPACE, "model_terbaik": best, "best_params": {k: {a_: (b if not isinstance(b, np.generic) else b.item()) for a_, b in v.items()} for k, v in best_params.items()},
           "holdout_dist": ho["label"].value_counts().to_dict(), "train_dist": tr["label"].value_counts().to_dict()}, open(f"{a.out}/results.json", "w"), indent=1, default=str)
print(res.round(3).to_string(index=False)); print("holdout:", ho["label"].value_counts().to_dict(), "| kesalahan model terbaik:", len(err))
