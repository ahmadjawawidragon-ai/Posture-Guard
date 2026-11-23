"""Tes logika inti tanpa kamera: landmark -> fitur -> model -> panel."""
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import json, types
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import features as ft, pose_utils as pu, build_reference as br

rng = np.random.default_rng(0)
# ---- data sintetis mirip statistik dataset (mean/std dari describe() di notebook)
MEAN = {"nose":(0.005,-0.452,-0.343),"left_ear":(0.080,-0.482,-0.228),"right_ear":(-0.061,-0.437,-0.230),
        "left_shoulder":(0.180,-0.357,-0.095),"right_shoulder":(-0.138,-0.407,-0.045),
        "left_hip":(0.100,0.002,0.030),"right_hip":(-0.098,-0.005,-0.027)}
def make_df(n=600):
    rows=[]
    for _ in range(n):
        r={}
        lean = rng.choice(["TUP","TLL","TLR"])
        dx = {"TUP":0,"TLL":0.25,"TLR":-0.25}[lean]
        for lm,(x,y,z) in MEAN.items():
            top = lm not in ("left_hip","right_hip")
            r[f"{lm}_x"]=x+(dx*(1 if top else 0))+rng.normal(0,0.01)
            r[f"{lm}_y"]=y+rng.normal(0,0.01); r[f"{lm}_z"]=z+rng.normal(0,0.01)
        r["upperbody_label"]=lean; r["subject"]=int(rng.integers(1,5)); rows.append(r)
    return pd.DataFrame(rows)
df = make_df()

# ---- 1) landmark MediaPipe palsu (33 titik) + offset -> harus pulih persis setelah dikurangi titik tengah pinggul
class LM(types.SimpleNamespace): pass
def fake_world(row, offset=(0,0,0), mirror=False):
    lms=[LM(x=0,y=0,z=0,visibility=0.99) for _ in range(33)]
    for name,i in pu.MP_INDEX.items():
        src = name
        if mirror:   # frame ter-mirror: x dibalik DAN MediaPipe menukar label kiri/kanan
            src = name.replace("left","TMP").replace("right","left").replace("TMP","right")
        x,y,z = row[f"{src}_x"], row[f"{src}_y"], row[f"{src}_z"]
        if mirror: x=-x
        lms[i]=LM(x=x+offset[0],y=y+offset[1],z=z+offset[2],visibility=0.99)
    return lms
r0 = df.iloc[0][ft.RAW_COLS].to_dict()
mid = {ax: (r0[f"left_hip_{ax}"]+r0[f"right_hip_{ax}"])/2 for ax in "xyz"}
expected = {k: r0[k]-mid[k[-1]] for k in ft.RAW_COLS}          # r0 digeser agar titik tengah pinggul = 0
row, vis = pu.world_to_row(fake_world(r0, offset=(0.3,0.2,-0.1)))
assert all(abs(row[k]-expected[k])<1e-9 for k in ft.RAW_COLS), "recentering gagal"
cols = ft.build_feature_sets(df.iloc[:1])["C_terpilih+sudut"].columns
f_app = pu.feature_frame(row,"C_terpilih+sudut",cols)
f_nb  = ft.build_feature_sets(pd.DataFrame([expected]))["C_terpilih+sudut"]
assert np.allclose(f_app.to_numpy(), f_nb.to_numpy()), "fitur app != fitur notebook"
print("OK 1: offset arbitrer dihapus; fitur app identik dengan features.py di notebook (34 fitur);",
      f"koreksi titik tengah pinggul data ini hanya {max(abs(v) for v in mid.values()):.4f}")

# ---- 2) efek mirror (x dibalik + label kiri/kanan ditukar MediaPipe)
cols = ft.build_feature_sets(df.iloc[:1])["C_terpilih+sudut"].columns
def feats_of(r, mirror):
    row,_ = pu.world_to_row(fake_world(r, mirror=mirror)); return row, pu.feature_frame(row,"C_terpilih+sudut",cols)
tup = df[df.upperbody_label=="TUP"].iloc[0][ft.RAW_COLS].to_dict()
tll = df[df.upperbody_label=="TLL"].iloc[0][ft.RAW_COLS].to_dict()
(ua,_),(ub,_) = feats_of(tup,False), feats_of(tup,True)
print(f"OK 2a: TEGAK  asli left_shoulder_x={ua['left_shoulder_x']:+.3f} | mirror {ub['left_shoulder_x']:+.3f}  -> tanda sama, TIDAK bisa jadi detektor mirror")
assert ua["left_shoulder_x"]>0 and ub["left_shoulder_x"]>0
(_,fa),(_,fb) = feats_of(tll,False), feats_of(tll,True)
print(f"OK 2b: TLL    asli trunk_tilt_lateral={fa.trunk_tilt_lateral.iloc[0]:+.1f} | mirror {fb.trunk_tilt_lateral.iloc[0]:+.1f}  -> arah tertukar (TLL jadi TLR)")
assert fa.trunk_tilt_lateral.iloc[0]>0 > fb.trunk_tilt_lateral.iloc[0]
a,_ = pu.world_to_row(fake_world(tll))

# ---- 3) referensi + status + model + panel
ref = br.build(df); json.dumps(ref)
names = list(ref["features"])
X = ft.build_feature_sets(df)["C_terpilih+sudut"]
y = df.upperbody_label.values
rf = RandomForestClassifier(100, random_state=0).fit(X, y)
classes = list(rf.classes_)
probe = pu.feature_frame(a,"C_terpilih+sudut",names)
p = rf.predict_proba(probe)[0]; print("OK 3: prediksi TLL ->", classes[int(np.argmax(p))], np.round(p,2))
assert classes[int(np.argmax(p))]=="TLL"
fr = ref["features"]["nose_x"]
assert pu.status_of(fr["p50"],fr)=="ok" and pu.status_of(fr["max"]+1,fr)=="luar" and pu.status_of((fr["p99"]+fr["max"])/2,fr)=="tepi"
print("OK 4: status ok/tepi/luar benar")

# ---- 4) putaran pinggul pada data sintetis (target ~16 derajat seperti dataset asli)
hr = ref["extras"]["hip_rotation_deg"]["p50"]; print(f"OK 5: putaran pinggul median sintetis {hr:.1f} derajat")
assert 14 < hr < 18

# ---- 5) smoother, gambar, backend
sm = pu.Smoother(3)
for v in (0.0, 1.0, 100.0): out = sm.update({"k":v})
assert out["k"]==1.0; print("OK 6: smoother median (tahan lonjakan)")
import cv2
img = np.zeros((480,640,3),np.uint8)
imglm=[LM(x=0.5,y=0.5,visibility=1.0) for _ in range(33)]
pu.draw_pose(img, imglm); assert img.sum()>0; print("OK 7: draw_pose")
kind,path,ready = pu.backend_status(1, "."); print("Backend terdeteksi:", kind, "| model siap:", ready)
try:
    pu.PoseEstimator(1, model_dir=".")
except FileNotFoundError as e: print("OK 8: pesan error model hilang ->", str(e)[:70], "...")
try:
    pu.feature_frame(r0,"A_semua_koordinat",[])
except ValueError as e: print("OK 9: set A ditolak dengan pesan jelas")
# ---- 6) statistik sesi + basis pengetahuan + model biner
import knowledge as kb
st_ = pu.SessionStats()
for lbl,non,dt in [("TUP",False,1.0),("TLF",True,0.5),("TLF",True,0.5),("TUP",False,1.0),("TLL",True,0.2)]: st_.update(lbl,non,dt)
assert abs(st_.pct_nontup-100*1.2/3.2)<1e-9 and abs(st_.longest-1.0)<1e-9 and abs(st_.streak-0.2)<1e-9
st_.update("TUP",False,500); assert abs(st_.total-4.2)<1e-9     # dt dibatasi 1 detik (kamera terputus)
st_.reset(); assert st_.total==0 and st_.pct_nontup==0
print("OK 10: SessionStats (persen tidak tegak, streak terlama, batas dt)")
assert set(kb.KNOWLEDGE)=={"TUP","TLF","TLB","TLL","TLR"} and kb.KNOWLEDGE["TLL"]["saran"]==kb.KNOWLEDGE["TLR"]["saran"]
print("OK 11: basis pengetahuan mencakup 5 kelas")
from sklearn.base import clone
yb=(df.upperbody_label!="TUP").astype(int).values
bm=clone(rf).fit(X,yb); pn=bm.predict_proba(pu.feature_frame(a,"C_terpilih+sudut",names))[0][1]
pn5=1-dict(zip(classes,rf.predict_proba(pu.feature_frame(a,"C_terpilih+sudut",names))[0]))["TUP"]
assert pn>0.5 and pn5>0.5; print(f"OK 12: P(non-TUP) biner={pn:.2f} vs 1-P(TUP) 5-kelas={pn5:.2f} (sama-sama non-TUP)")
# ---- 7) AsyncPose: estimator lambat tidak boleh membuat antrean / delay menumpuk
import time, threading
class SlowEst:
    def process(self, rgb):
        time.sleep(0.4)
        return fake_world(r0), [LM(x=.5,y=.5,visibility=1.0) for _ in range(33)]
ap = pu.AsyncPose(lambda: SlowEst())
frame = np.zeros((48,64,3),np.uint8); pushed=0; t0=time.time(); push_cost=[]
while time.time()-t0 < 3.0:                      # kamera 30 fps selama 3 detik
    a0=time.time(); ap.push(frame); pushed+=1; push_cost.append(time.time()-a0); time.sleep(1/30)
time.sleep(0.5)
assert max(push_cost) < 0.01, "push() harus instan"
assert ap.processed <= 9 and ap.processed >= 4, ap.processed   # ~3.5 detik / 0.4 dtk, BUKAN 90 frame antre
age = time.time() - ap.result()[3]
assert ap.result()[0] is not None and age < 1.0
ap.close()
print(f"OK 13: {pushed} frame masuk, hanya {ap.processed} diproses (frame lama dibuang); push maks {max(push_cost)*1000:.2f} ms; hasil terbaru {age:.2f} dtk")
bad = pu.AsyncPose(lambda: (_ for _ in ()).throw(FileNotFoundError("model hilang"))); time.sleep(0.2)
assert "model hilang" in bad.error; print("OK 14: error pembuatan estimator tertangkap dan bisa ditampilkan di app")
# ---- 8) kalibrasi: offset tetap (mis. kamera rendah) dikoreksi, kelas lain tetap terbedakan
tupm = X[y=="TUP"].median()
bias = pd.Series(0.0, index=X.columns); bias["nose_z"] = -0.2; bias["head_offset_z"] = -0.25   # kamera dekat: kepala tampak maju
user_tup = X[y=="TUP"].iloc[0] + bias
calib = user_tup                                      # hasil kalibrasi duduk tegak
raw_now = X[y=="TUP"].iloc[1] + bias
fixed = pu.apply_calibration(raw_now, calib, tupm)
assert abs(fixed["nose_z"]-tupm["nose_z"]) < 0.05 and abs(raw_now["nose_z"]-tupm["nose_z"]) > 0.15
assert pu.apply_calibration(raw_now, None, tupm) is raw_now
print("OK 15: kalibrasi menghapus offset z tetap (nose_z menyimpang >0,15 jadi <0,05)")
print("\nSEMUA TES LULUS")
