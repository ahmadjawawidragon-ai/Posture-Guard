"""
train_depan.py  -  Dataset 1 (MultiPosture, tampak depan): perbandingan 4 model, nested CV, evaluasi holdout.
Meniru metodologi train_side.py supaya laporan konsisten (CV bersarang pada data latih, model terbaik dipilih
dari CV, bukan dari hasil uji). Memakai features.py yang SAMA dengan aplikasi.

Jalankan dari folder proyek:
    python pelatihan/train_depan.py --csv "data/Data ML.csv"
Opsi: --n-jobs -1 (paralel)  --fast (uji cepat)  --label-col upperbody_label   --group-col subject   --feature-set C_terpilih+sudut
Keluaran (hasil/depan/): 03_perbandingan_model.csv, tabel_perbandingan.md, confusion_matrix_semua_model.png,
  perbandingan_model.png, feature_importance_rf.png, 04_kesalahan_model_terbaik.csv, laporan_klasifikasi_terbaik.txt, results.json.
Model aplikasi (posture_model.joblib) TIDAK ditimpa kecuali kamu memberi --model-out.
"""
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import argparse, json, os, time, warnings
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import features as ft
warnings.filterwarnings("ignore")
SEED = 42

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=str(ROOT / "data" / "Data ML.csv"))
ap.add_argument("--label-col", default="upperbody_label")
ap.add_argument("--group-col", default=None, help="kolom ID subjek (otomatis dicari bila tidak diisi)")
ap.add_argument("--feature-set", default="C_terpilih+sudut")
ap.add_argument("--out", default=str(ROOT / "hasil" / "depan"))
ap.add_argument("--n-jobs", type=int, default=1, help="paralel untuk grid search (-1 = semua core)")
ap.add_argument("--fast", action="store_true", help="grid hiperparameter dikecilkan (uji cepat, bukan untuk laporan final)")
ap.add_argument("--model-out", default=None, help="isi bila ingin menyimpan model terbaik (hati-hati: menimpa file yang sama)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# ---------------------------------------------------------------- 1. data
df = pd.read_csv(a.csv)
assert a.label_col in df.columns, f"Kolom label '{a.label_col}' tidak ada. Kolom: {list(df.columns)[:15]}..."
df = df.dropna(subset=[a.label_col]).reset_index(drop=True)
gcol = a.group_col or next((c for c in ("subject", "subject_id", "participant", "person", "id") if c in df.columns), None)
X_all = ft.build_feature_sets(df)[a.feature_set]
le = LabelEncoder().fit(df[a.label_col]); y_all = le.transform(df[a.label_col]); classes = list(le.classes_)
if gcol:
    groups = df[gcol].astype(str).values
    print(f"Pembagian per subjek memakai kolom '{gcol}' ({len(set(groups))} subjek).")
else:
    groups = np.arange(len(df))
    print("PERINGATAN: kolom subjek tidak ditemukan. Pembagian acak per baris; skor bisa terlalu optimis (kebocoran antar-subjek).")
print("Sampel:", len(df), "| kelas:", {c: int(n) for c, n in zip(classes, np.bincount(y_all))}, "| fitur:", X_all.shape[1])

# ---------------------------------------------------------------- 2. holdout (1 dari 5 lipatan, terstratifikasi dan terkelompok)
cv_out = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
tr_idx, ho_idx = next(cv_out.split(X_all, y_all, groups))
Xtr, Xho, ytr, yho, gtr = X_all.iloc[tr_idx], X_all.iloc[ho_idx], y_all[tr_idx], y_all[ho_idx], groups[tr_idx]
print("Latih:", len(Xtr), "| uji (holdout):", len(Xho))

# ---------------------------------------------------------------- 3. EDA ringkas
sns.set_theme(style="whitegrid")
pd.Series(df[a.label_col]).value_counts().reindex(classes).plot(kind="bar", figsize=(5.5, 3.4), title="Distribusi kelas Dataset 1")
plt.ylabel("jumlah sampel"); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(f"{a.out}/eda_distribusi_kelas.png", dpi=130); plt.close()

# ---------------------------------------------------------------- 4. 4 model + grid
try:
    from xgboost import XGBClassifier
    gb_name, gb = "XGBoost", (XGBClassifier(eval_metric="mlogloss", random_state=SEED, n_jobs=1), {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]})
except Exception:
    gb_name, gb = "Gradient Boosting (pengganti XGBoost)", (HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced"), {"max_iter": [200, 400], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]})
MODELS = {
    "Logistic Regression": (Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(class_weight="balanced", max_iter=3000))]), {"m__C": [0.1, 1, 10]}),
    "Random Forest": (RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1), {"n_estimators": [200, 400], "max_depth": [None, 10], "min_samples_leaf": [1, 3]}),
    "SVM (RBF)": (Pipeline([("sc", StandardScaler()), ("m", SVC(class_weight="balanced", probability=True, random_state=SEED))]), {"m__C": [1, 10], "m__gamma": ["scale", 0.1]}),
    gb_name: gb,
}
if a.fast:
    MODELS = {k: (e, {p: v[:1] for p, v in g.items()}) for k, (e, g) in MODELS.items()}

def search(est, grid):
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=SEED) if gcol else StratifiedKFold(3, shuffle=True, random_state=SEED)
    return GridSearchCV(est, grid, scoring="f1_macro", cv=cv, n_jobs=a.n_jobs)
def fit_search(est, grid, X, y, g):
    gs = search(est, grid)
    return gs.fit(X, y, groups=g) if gcol else gs.fit(X, y)

# ---------------------------------------------------------------- 5. nested CV (latih) + evaluasi holdout
rows, preds, best_params, fitted = [], {}, {}, {}
for name, (est, grid) in MODELS.items():
    t0 = time.time(); fold_f1 = []
    for tri, tei in cv_out.split(Xtr, ytr, gtr):
        gs = fit_search(est, grid, Xtr.iloc[tri], ytr[tri], gtr[tri])
        fold_f1.append(f1_score(ytr[tei], gs.predict(Xtr.iloc[tei]), average="macro"))
    gs = fit_search(est, grid, Xtr, ytr, gtr)
    best_params[name], fitted[name] = gs.best_params_, gs.best_estimator_
    t1 = time.time(); p = gs.predict(Xho); ms = (time.time() - t1) / len(Xho) * 1000; preds[name] = p
    r = dict(model=name, cv_f1_macro_mean=np.mean(fold_f1), cv_f1_macro_std=np.std(fold_f1), train_f1_macro=f1_score(ytr, gs.predict(Xtr), average="macro"),
             ho_accuracy=accuracy_score(yho, p), ho_precision_macro=precision_score(yho, p, average="macro", zero_division=0),
             ho_recall_macro=recall_score(yho, p, average="macro", zero_division=0), ho_f1_macro=f1_score(yho, p, average="macro"), ms_per_sample=ms)
    for k, c in enumerate(classes):
        r[f"ho_recall_{c}"] = recall_score(yho, p, labels=[k], average="macro", zero_division=0)
    rows.append(r); print(f"{name:40s} CV F1={r['cv_f1_macro_mean']:.3f}±{r['cv_f1_macro_std']:.3f}  HO F1={r['ho_f1_macro']:.3f}  ({time.time()-t0:.0f}s)")
res = pd.DataFrame(rows); res.round(4).to_csv(f"{a.out}/03_perbandingan_model.csv", index=False)
best = res.sort_values("cv_f1_macro_mean", ascending=False).iloc[0]["model"]      # dipilih dari CV latih, BUKAN dari holdout
print("Model terbaik (berdasar CV F1 makro):", best)

# tabel siap tempel untuk laporan
tab = res[["model", "ho_accuracy", "ho_precision_macro", "ho_recall_macro", "ho_f1_macro", "cv_f1_macro_mean", "train_f1_macro"]].copy()
tab.columns = ["Model", "Accuracy", "Precision (makro)", "Recall (makro)", "F1 (makro)", "F1 CV (latih)", "F1 latih"]
t3 = tab.round(3)
md = "| " + " | ".join(t3.columns) + " |\n|" + "---|" * len(t3.columns) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in t3.values)
open(f"{a.out}/tabel_perbandingan.md", "w", encoding="utf-8").write(md + "\n")

# ---------------------------------------------------------------- 6. gambar
fig, axs = plt.subplots(1, len(MODELS), figsize=(3.4 * len(MODELS), 3.4))
for ax, (n, p) in zip(axs, preds.items()):
    cm = confusion_matrix(yho, p, labels=range(len(classes))); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(n.split(" (")[0], fontsize=9); ax.set_xlabel("prediksi"); ax.set_ylabel("aktual")
plt.tight_layout(); plt.savefig(f"{a.out}/confusion_matrix_semua_model.png", dpi=130); plt.close()
x = np.arange(len(res)); plt.figure(figsize=(7.5, 3.8))
plt.bar(x - 0.2, res["cv_f1_macro_mean"], 0.4, yerr=res["cv_f1_macro_std"], label="CV (latih, bersarang)"); plt.bar(x + 0.2, res["ho_f1_macro"], 0.4, label="Holdout (uji)")
plt.xticks(x, [m.split(" (")[0] for m in res["model"]], rotation=15); plt.ylabel("F1 makro"); plt.ylim(0, 1); plt.legend(); plt.title("Perbandingan model Dataset 1 (tampak depan)")
plt.tight_layout(); plt.savefig(f"{a.out}/perbandingan_model.png", dpi=130); plt.close()
rf = fitted["Random Forest"]
pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values().tail(15).plot(kind="barh", figsize=(6, 4.5), title="Feature importance (Random Forest, 15 teratas)")
plt.tight_layout(); plt.savefig(f"{a.out}/feature_importance_rf.png", dpi=130); plt.close()

# ---------------------------------------------------------------- 7. analisis kesalahan, ringkasan, (opsional) simpan model
bp = preds[best]
err = pd.DataFrame({"aktual": le.inverse_transform(yho[bp != yho]), "prediksi": le.inverse_transform(bp[bp != yho])})
err.value_counts().rename("jumlah").reset_index().to_csv(f"{a.out}/04_kesalahan_model_terbaik.csv", index=False)
open(f"{a.out}/laporan_klasifikasi_terbaik.txt", "w", encoding="utf-8").write(f"Model: {best}\n" + classification_report(yho, bp, target_names=classes, digits=3, zero_division=0))
if a.model_out:
    final = fit_search(*MODELS[best], X_all, y_all, groups).best_estimator_
    joblib.dump({"model": final, "label_encoder": le, "classes": classes, "feature_names": list(X_all.columns), "feature_set": a.feature_set, "model_name": best}, a.model_out)
json.dump({"n_total": len(df), "n_train": len(Xtr), "n_holdout": len(Xho), "kolom_subjek": gcol, "model_terbaik": best,
           "best_params": {k: {a_: (b if not isinstance(b, np.generic) else b.item()) for a_, b in v.items()} for k, v in best_params.items()},
           "holdout_dist": pd.Series(le.inverse_transform(yho)).value_counts().to_dict()}, open(f"{a.out}/results.json", "w"), indent=1, default=str)
print(res.round(3).to_string(index=False)); print("Kesalahan model terbaik pada holdout:", int((bp != yho).sum()), "dari", len(yho)); print("Hasil di:", a.out)
