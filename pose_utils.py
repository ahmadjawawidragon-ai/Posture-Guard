"""
pose_utils.py
Logika inti aplikasi (tanpa Streamlit) supaya bisa diuji terpisah.

Alur:  frame ASLI (tidak di-mirror) -> MediaPipe -> pose_world_landmarks
       -> kurangi titik tengah pinggul -> 21 kolom RAW_COLS
       -> features.build_feature_sets (fungsi yang SAMA dengan notebook)
       -> model.
"""
import math
import os
import threading
import time
import urllib.request

import cv2
import numpy as np
import pandas as pd

import features as ft
import features_side as fsd

# Indeks landmark BlazePose (33 titik) untuk 7 landmark yang dipakai model
MP_INDEX = {
    "nose": 0, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
}

# Garis kerangka untuk tampilan (indeks BlazePose); tidak bergantung versi MediaPipe
UPPER_CONNECTIONS = [(0, 7), (0, 8), (7, 11), (8, 12), (11, 12), (11, 23),
                     (12, 24), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16)]

# Fitur yang paling rentan terhadap ketidakcocokan kamera vs dataset
RISK_FEATURES = ["shoulder_rotation", "left_hip_z", "right_hip_z",
                 "head_offset_z", "ear_offset_z", "neck_tilt_sagittal",
                 "trunk_tilt_sagittal"]

TASK_MODELS = {0: "lite", 1: "full", 2: "heavy"}
TASK_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_{name}/float16/latest/pose_landmarker_{name}.task")


# ----------------------------------------------------------------- landmark -> fitur
def world_to_row(world_lms):
    """world_lms: 33 landmark (punya .x .y .z .visibility) dari pose_world_landmarks.

    Titik tengah pinggul DIKURANGKAN walaupun world landmarks sudah berpusat di sana
    (aman bila versi/model berbeda). Mengembalikan (row, visibility).
    """
    pts = {n: np.array([world_lms[i].x, world_lms[i].y, world_lms[i].z], dtype=float)
           for n, i in MP_INDEX.items()}
    center = (pts["left_hip"] + pts["right_hip"]) / 2.0
    row = {}
    for name, p in pts.items():
        q = p - center
        row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"] = map(float, q)
    vis = {n: float(getattr(world_lms[i], "visibility", 1.0)) for n, i in MP_INDEX.items()}
    return row, vis


SIDE_INDEX = {"nose": 0, "left_ear": 7, "right_ear": 8, "left_shoulder": 11, "right_shoulder": 12,
              "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26}
SIDE_PARTS = ("ear", "shoulder", "hip", "knee")


def world_to_side_row(world_lms, image_lms=None, frame_wh=None):
    """Mode TAMPAK SAMPING: x, y, z, visibility landmark world (tanpa pemusatan pinggul, karena fitur
    samping hanya memakai selisih antar-titik). Mengembalikan (row, vis) seperti world_to_row.

    Bila image_lms dan frame_wh=(lebar, tinggi) diberikan, baris juga memuat koordinat citra
    ({n}_ix, {n}_iy) dan box_w/box_h = ukuran frame. Webcam memproses FRAME UTUH, jadi ukuran frame
    berperan sebagai 'box'. Dibutuhkan hanya oleh model berruang fitur 'image'; ruang 'world' tidak
    memakainya."""
    row, vis = {}, {}
    for n, i in SIDE_INDEX.items():
        lm = world_lms[i]
        v = float(getattr(lm, "visibility", 1.0))
        row[f"{n}_x"], row[f"{n}_y"], row[f"{n}_z"], row[f"{n}_vis"] = float(lm.x), float(lm.y), float(lm.z), v
        if image_lms is not None:
            row[f"{n}_ix"], row[f"{n}_iy"] = float(image_lms[i].x), float(image_lms[i].y)
        vis[n] = v
    if image_lms is not None and frame_wh is not None:
        row["box_w"], row["box_h"] = float(frame_wh[0]), float(frame_wh[1])
    return row, vis


def side_feature_frame(row, feature_names, space="world"):
    """Fitur samping dengan features_side.py yang SAMA dengan train_side.py."""
    if space != "world":
        need = {"box_w", "box_h", "nose_ix", "nose_iy"}
        if not need <= set(row):
            raise ValueError(
                f"Model samping memakai ruang fitur '{space}', tetapi baris kamera tidak punya kolom "
                f"{sorted(need - set(row))}. Pakai posture_model_side.joblib terbaru (ruang 'world'), "
                f"atau pastikan world_to_side_row dipanggil dengan image_lms dan frame_wh.")
    return fsd.engineer_side_features(pd.DataFrame([row]), space)[list(feature_names)]


def feature_frame(row, feature_set, feature_names):
    """Hitung fitur dengan features.py yang sama seperti saat training."""
    if feature_set.startswith("A_"):
        raise ValueError("Set fitur A (99 koordinat) tidak bisa dipakai di aplikasi: "
                         "hanya 7 landmark yang diambil. Gunakan set B, C, atau D.")
    df = pd.DataFrame([row])
    X = ft.build_feature_sets(df)[feature_set]
    return X[list(feature_names)]


def hip_rotation_deg(left_z, right_z, left_x, right_x):
    """Putaran garis pinggul terhadap kamera (derajat); versi vektor & skalar."""
    dz = np.asarray(left_z, float) - np.asarray(right_z, float)
    dx = np.abs(np.asarray(left_x, float) - np.asarray(right_x, float)) + 1e-9
    return np.degrees(np.arctan2(dz, dx))


def status_of(value, ref):
    """'ok' di dalam P01-P99, 'tepi' di dalam min-max tapi di luar P01-P99, 'luar' di luar min-max."""
    if value < ref["min"] or value > ref["max"]:
        return "luar"
    if value < ref["p01"] or value > ref["p99"]:
        return "tepi"
    return "ok"


def describe_series(s):
    q = s.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    return {"min": float(s.min()), "p01": float(q[0.01]), "p05": float(q[0.05]),
            "p50": float(q[0.5]), "p95": float(q[0.95]), "p99": float(q[0.99]),
            "max": float(s.max())}


class Smoother:
    """Median koordinat beberapa frame terakhir (mengurangi getar landmark)."""
    def __init__(self, n):
        from collections import deque
        self.buf = deque(maxlen=max(1, int(n)))

    def update(self, row):
        self.buf.append(row)
        return {k: float(np.median([r[k] for r in self.buf])) for k in row}


# ----------------------------------------------------------------- gambar
def draw_pose(img_bgr, image_lms, min_vis=0.3):
    """Gambar kerangka pada img_bgr (in-place). Hanya bentuk, tanpa teks,
    sehingga aman bila nanti seluruh gambar di-mirror untuk tampilan."""
    h, w = img_bgr.shape[:2]
    pt = lambda i: (int(image_lms[i].x * w), int(image_lms[i].y * h))
    ok = lambda i: getattr(image_lms[i], "visibility", 1.0) >= min_vis
    for a, b in UPPER_CONNECTIONS:
        if ok(a) and ok(b):
            cv2.line(img_bgr, pt(a), pt(b), (255, 200, 0), 2)
    used = set(MP_INDEX.values())
    for i in used:
        if ok(i):
            cv2.circle(img_bgr, pt(i), 5, (0, 255, 120), -1)
    return img_bgr


# ----------------------------------------------------------------- backend MediaPipe
def backend_status(complexity, model_dir="."):
    """('legacy'|'tasks', path_atau_None, siap:bool). Dipakai sebelum kamera dinyalakan."""
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        return "legacy", None, True
    path = os.path.join(model_dir, f"pose_landmarker_{TASK_MODELS[complexity]}.task")
    return "tasks", path, os.path.exists(path)


def download_task_model(complexity, model_dir="."):
    name = TASK_MODELS[complexity]
    path = os.path.join(model_dir, f"pose_landmarker_{name}.task")
    urllib.request.urlretrieve(TASK_URL.format(name=name), path)
    return path


class PoseEstimator:
    """Pembungkus tunggal untuk MediaPipe legacy (mp.solutions) dan Tasks (PoseLandmarker)."""

    def __init__(self, complexity=1, det_conf=0.5, trk_conf=0.5, live=True, model_dir="."):
        import mediapipe as mp
        self.mp, self.live, self._ts = mp, live, 0
        self.backend, path, ready = backend_status(complexity, model_dir)
        if self.backend == "legacy":
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=not live, model_complexity=complexity,
                smooth_landmarks=True, enable_segmentation=False,
                min_detection_confidence=det_conf, min_tracking_confidence=trk_conf)
        else:
            if not ready:
                raise FileNotFoundError(
                    f"File model '{path}' tidak ditemukan. Unduh dari "
                    f"{TASK_URL.format(name=TASK_MODELS[complexity])}")
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            opts = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=path),
                running_mode=(vision.RunningMode.VIDEO if live else vision.RunningMode.IMAGE),
                num_poses=1, min_pose_detection_confidence=det_conf,
                min_pose_presence_confidence=det_conf, min_tracking_confidence=trk_conf,
                output_segmentation_masks=False)
            self._pose = vision.PoseLandmarker.create_from_options(opts)

    def process(self, rgb):
        """rgb: uint8 HxWx3 dari frame ASLI (jangan di-mirror).
        Mengembalikan (world_landmarks, image_landmarks) atau (None, None)."""
        if self.backend == "legacy":
            rgb = np.ascontiguousarray(rgb)
            rgb.flags.writeable = False
            res = self._pose.process(rgb)
            if not res.pose_world_landmarks:
                return None, None
            return list(res.pose_world_landmarks.landmark), list(res.pose_landmarks.landmark)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.live:
            self._ts = max(self._ts + 1, int(time.monotonic() * 1000))
            res = self._pose.detect_for_video(img, self._ts)
        else:
            res = self._pose.detect(img)
        if not res.pose_world_landmarks:
            return None, None
        return res.pose_world_landmarks[0], res.pose_landmarks[0]


class SessionStats:
    """Akumulasi waktu per kelas dalam satu sesi pemantauan (untuk persentase waktu tidak tegak)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.secs, self.total, self.nontup, self.streak, self.longest = {}, 0.0, 0.0, 0.0, 0.0

    def update(self, label, non_tup, dt):
        dt = min(max(float(dt), 0.0), 1.0)      # dt besar (kamera terputus) tidak dihitung penuh
        self.secs[label] = self.secs.get(label, 0.0) + dt
        self.total += dt
        if non_tup:
            self.nontup += dt
            self.streak += dt
            self.longest = max(self.longest, self.streak)
        else:
            self.streak = 0.0

    @property
    def pct_nontup(self):
        return 100.0 * self.nontup / self.total if self.total > 0 else 0.0


class AsyncPose:
    """Jalankan estimator di thread sendiri, hanya pada frame TERBARU (frame lama dibuang).

    Penayangan video tidak menunggu MediaPipe, jadi tidak ada antrean frame yang
    menumpuk menjadi delay puluhan detik. push() dipanggil per frame dan sangat cepat.
    """
    def __init__(self, make_estimator, idle_timeout=30.0, row_fn=None):
        self._make, self._idle = make_estimator, idle_timeout
        self._row_fn = row_fn or world_to_row
        self._lock = threading.Lock()
        self._frame, self._fid, self._done = None, 0, 0
        self._result = None            # (row, vis, image_lms, waktu); row None bila pose tidak terdeteksi
        self._stop, self._last_push = False, time.time()
        self.error, self.fps, self.processed = None, 0.0, 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, bgr):
        with self._lock:
            self._frame, self._fid = bgr, self._fid + 1
        self._last_push = time.time()

    def result(self):
        with self._lock:
            return self._result

    def close(self):
        self._stop = True

    def _run(self):
        try:
            est = self._make()
        except Exception as e:                     # mis. file model hilang
            self.error = repr(e)
            return
        t0, n = time.time(), 0
        while not self._stop:
            with self._lock:
                frame, fid = self._frame, self._fid
            if frame is None or fid == self._done:
                if time.time() - self._last_push > self._idle:
                    break
                time.sleep(0.003)
                continue
            self._done = fid
            try:
                world, image = est.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if world is None:
                    res = (None, None, None, time.time())
                else:
                    if self._row_fn is world_to_side_row:      # mode samping: sertakan koordinat citra + ukuran frame
                        row, vis = world_to_side_row(world, image, (frame.shape[1], frame.shape[0]))
                    else:
                        row, vis = self._row_fn(world)
                    res = (row, vis, image, time.time())
            except Exception as e:
                self.error = repr(e)
                break
            with self._lock:
                self._result = res
            n += 1
            self.processed = n
            if n % 10 == 0:
                self.fps = 10.0 / max(time.time() - t0, 1e-6)
                t0 = time.time()


def apply_calibration(raw, calib, tup_median):
    """Geser fitur kamera supaya postur tegak pengguna (calib) setara median TUP dataset.
    raw, calib, tup_median: pd.Series berindeks nama fitur. calib None = tanpa koreksi.
    Koreksi ini hanya menghilangkan OFFSET tetap (mis. kamera rendah/dekat); tidak memperbaiki
    sensitivitas yang hilang (mis. pinggul tertutup meja sehingga sudut badan tidak terbaca)."""
    if calib is None:
        return raw
    return raw - calib + tup_median
