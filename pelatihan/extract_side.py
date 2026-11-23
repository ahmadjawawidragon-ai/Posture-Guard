"""
extract_side.py  -  Dataset 2 (Sitting Posture Detection, tampak samping)
Citra + anotasi COCO  ->  landmark MediaPipe Pose (world) per bounding box  ->  landmarks_side.csv

Langkah (sesuai proposal Bagian 9.2):
  1. Buang kelas 'slouch'; pemetaan good_posture/leaning_forward/leaning_backward -> TUP/TLF/TLB.
  2. Satu bounding box = satu sampel: crop (padding 10%) lalu jalankan MediaPipe pada crop.
  3. Box yang saling tumpang tindih (IoU > 0.3) dengan label berbeda dibuang (label konflik).
  4. Hasil deteksi pada crop dibandingkan dengan citra utuh (hanya citra 1 box) -> tingkat deteksi.
  5. Kunci grup = nama foto asli (sebelum '.rf.') agar 3 versi augmentasi tidak terpisah.

Pakai:  python extract_side.py --root "Sitting Posture Detection.v2i.coco" --model pose_landmarker_full.task
"""
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import argparse, json, os, re, sys
import numpy as np, pandas as pd, cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

LABEL_MAP = {"good_posture": "TUP", "leaning_forward": "TLF", "leaning_backward": "TLB"}  # slouch dibuang
NAMES = ["nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
         "left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder","left_elbow","right_elbow",
         "left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb",
         "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel",
         "left_foot_index","right_foot_index"]

def iou(a, b):
    ax2, ay2, bx2, by2 = a[0]+a[2], a[1]+a[3], b[0]+b[2], b[1]+b[3]
    iw, ih = max(0, min(ax2, bx2)-max(a[0], b[0])), max(0, min(ay2, by2)-max(a[1], b[1]))
    inter = iw*ih
    return inter / (a[2]*a[3] + b[2]*b[3] - inter + 1e-9)

def make_landmarker(path):
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=path),
        running_mode=vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3)
    return vision.PoseLandmarker.create_from_options(opts)

def detect(lm, rgb):
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.pose_world_landmarks:
        return None
    return res.pose_world_landmarks[0], res.pose_landmarks[0]

def flat(world, image):
    row = {}
    for i, n in enumerate(NAMES):
        w, p = world[i], image[i]
        row[f"{n}_x"], row[f"{n}_y"], row[f"{n}_z"] = w.x, w.y, w.z
        row[f"{n}_vis"] = getattr(w, "visibility", 1.0)
        row[f"{n}_ix"], row[f"{n}_iy"] = p.x, p.y          # koordinat citra (0-1) terhadap crop
    return row

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--model", default=str(ROOT / "pose_landmarker_full.task"))
    ap.add_argument("--out", default=str(ROOT / "data" / "landmarks_side.csv")); ap.add_argument("--pad", type=float, default=0.10)
    a = ap.parse_args()
    lm = make_landmarker(a.model)
    rows, log = [], []
    for split in ["train", "valid", "test"]:
        d = json.load(open(os.path.join(a.root, split, "_annotations.coco.json")))
        cat = {c["id"]: c["name"] for c in d["categories"]}
        imgs = {i["id"]: i for i in d["images"]}
        by_img = {}
        for an in d["annotations"]:
            by_img.setdefault(an["image_id"], []).append(an)
        for iid, ans in by_img.items():
            info = imgs[iid]
            img = cv2.imread(os.path.join(a.root, split, info["file_name"]))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); H, W = rgb.shape[:2]
            group = info["file_name"].split(".rf.")[0]
            full_ok = None
            if len(ans) == 1:                                   # perbandingan citra utuh (hanya 1 box)
                full_ok = detect(lm, rgb) is not None
            for k, an in enumerate(ans):
                name = cat[an["category_id"]]
                if name not in LABEL_MAP:
                    continue                                    # slouch dibuang
                conflict = any(j != k and cat[o["category_id"]] != name and iou(an["bbox"], o["bbox"]) > 0.3
                               for j, o in enumerate(ans))
                x, y, w, h = an["bbox"]
                px, py = int(w*a.pad), int(h*a.pad)
                x0, y0, x1, y1 = max(0, int(x)-px), max(0, int(y)-py), min(W, int(x+w)+px), min(H, int(y+h)+py)
                rec = dict(split=split, file=info["file_name"], group=group, ann_id=an["id"], label=LABEL_MAP[name],
                           n_boxes=len(ans), conflict=conflict, full_detected=full_ok, box_w=x1-x0, box_h=y1-y0)
                r = detect(lm, rgb[y0:y1, x0:x1])
                rec["detected"] = r is not None
                if r is not None:
                    rec.update(flat(*r))
                rows.append(rec)
        print(split, "selesai", len(rows), file=sys.stderr)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    print(df.groupby(["split", "label"]).agg(n=("label", "size"), detected=("detected", "sum")))
