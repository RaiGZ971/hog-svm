# import numpy as np
# import cv2
# import os
# import joblib
# import mediapipe as mp
# import time
# from scipy.spatial.distance import cdist
# from dotenv import load_dotenv
#
# load_dotenv()
#
# KEYFRAME_K = int(os.getenv("HYPERPARAMETER_K", 30))
# WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 120))
# IDLE_MOTION_THRESHOLD = float(os.getenv("IDLE_MOTION_THRESHOLD", 0.01))
#
#
# class FslSvmInfer:
#     def __init__(self, model_path):
#         data = joblib.load(model_path)
#
#         self.svm = data["model"]
#         self.scaler = data["scaler"]
#
#         self.mp = self._create_mp_task_video()
#
#         # buffers
#         self.buffer = []
#         self.motion = []
#         self.prev = None
#
#         # real-time timing (FIXED)
#         self.prev_time = None
#
#         # smoothing
#         self.motion_ema = 0.0
#         self.ema_alpha = 0.3
#
#         # optional stability (vote smoothing)
#         self.pred_buffer = []
#
#     # -------------------------
#     # MediaPipe setup
#     # -------------------------
#     def _create_mp_task_video(self):
#         BaseOptions = mp.tasks.BaseOptions
#         HandLandmarker = mp.tasks.vision.HandLandmarker
#         HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
#         VisionRunningMode = mp.tasks.vision.RunningMode
#
#         options = HandLandmarkerOptions(
#             base_options=BaseOptions(model_asset_path="ml/hand_landmarker.task"),
#             running_mode=VisionRunningMode.VIDEO,
#             num_hands=2
#         )
#
#         return HandLandmarker.create_from_options(options)
#
#     # -------------------------
#     # Landmark extraction
#     # -------------------------
#     def _extract_landmarks(self, frame, timestamp_ms):
#         if frame is None or frame.size == 0:
#             return None
#
#         frame = cv2.resize(frame, (320, 240))
#         frame = cv2.flip(frame, 1)
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#
#         result = self.mp.detect_for_video(mp_image, int(timestamp_ms))
#
#         left = [[0, 0, 0]] * 21
#         right = [[0, 0, 0]] * 21
#
#         if result and result.hand_landmarks:
#             for i, hand in enumerate(result.hand_landmarks):
#                 hand_type = result.handedness[i][0].category_name
#                 pts = [[lm.x, lm.y, lm.z] for lm in hand]
#
#                 if hand_type == "Left":
#                     left = pts
#                 else:
#                     right = pts
#
#         return left + right
#
#     # -------------------------
#     # Chamfer distance (motion)
#     # -------------------------
#     def _chamfer(self, a, b):
#         a = np.array(a)
#         b = np.array(b)
#
#         return (
#             cdist(a, b).min(axis=1).mean()
#             + cdist(b, a).min(axis=1).mean()
#         )
#
#     # -------------------------
#     # Keyframes
#     # -------------------------
#     def _keyframes(self, motion):
#         k = KEYFRAME_K
#         n = len(motion)
#
#         if n < 2:
#             return list(range(n))
#
#         idx = set([0, n - 1])
#
#         for i in range(1, n - 1):
#             if motion[i] > motion[i - 1] and motion[i] > motion[i + 1]:
#                 idx.add(i)
#
#         idx.update(np.argsort(motion)[-k:])
#         idx = sorted(idx)
#
#         while len(idx) < k:
#             idx.append(idx[-1])
#
#         return idx[:k]
#
#     # -------------------------
#     # Feature extraction
#     # -------------------------
#     def _features(self, frames, motion):
#         selected = [frames[i] for i in self._keyframes(motion)]
#         selected = np.array(selected).reshape(len(selected), -1)
#
#         f = []
#         f.extend(selected.mean(axis=0))
#         f.extend(selected.std(axis=0))
#         f.extend(selected.min(axis=0))
#         f.extend(selected.max(axis=0))
#
#         return np.array(f)
#
#     # -------------------------
#     # Reset state
#     # -------------------------
#     def reset(self):
#         self.buffer = []
#         self.motion = []
#         self.prev = None
#         self.prev_time = None
#         self.motion_ema = 0.0
#         self.pred_buffer = []
#
#     # -------------------------
#     # MAIN PREDICTION
#     # -------------------------
#     def predict(self, frame):
#         now = time.time() * 1000  # REAL timestamp
#
#         pc = self._extract_landmarks(frame, now)
#         if pc is None:
#             return None
#
#         # skip invalid hand
#         has_hand = not np.all(np.array(pc) == 0.0)
#         if not has_hand:
#             return None
#
#         # time delta awareness (optional future use)
#         if self.prev_time is not None:
#             dt = now - self.prev_time
#         self.prev_time = now
#
#         self.buffer.append(pc)
#
#         # motion
#         if self.prev is None:
#             motion_val = 0.0
#         else:
#             motion_val = self._chamfer(self.prev, pc)
#
#         self.prev = pc
#
#         # EMA smoothing (IMPORTANT FIX)
#         self.motion_ema = (
#             self.ema_alpha * motion_val +
#             (1 - self.ema_alpha) * self.motion_ema
#         )
#
#         self.motion.append(self.motion_ema)
#
#         # sliding window
#         if len(self.buffer) > WINDOW_SIZE:
#             self.buffer.pop(0)
#             self.motion.pop(0)
#
#         if len(self.buffer) < KEYFRAME_K:
#             return None
#
#         # adaptive idle filter (LESS STRICT)
#         recent_motion = np.mean(self.motion[-10:])
#
#         if recent_motion < IDLE_MOTION_THRESHOLD * 0.7:
#             return None
#
#         # feature extraction
#         feat = self._features(self.buffer, self.motion)
#         feat = self.scaler.transform([feat])
#
#         print("FEATURE MEAN:", np.mean(feat), "STD:", np.std(feat))
#         print("UNIQUE BUFFER VARIANCE:", np.var(self.buffer))
#
#         probs = self.svm.predict_proba(feat)[0]
#         idx = np.argmax(probs)
#
#         label = self.svm.classes_[idx]
#         confidence = probs[idx]
#
#         print("CONFIDENCE:", float(confidence))
#         print("LABEL:", str(label))
#
#         # -------------------------
#         # simple prediction smoothing
#         # -------------------------
#         self.pred_buffer.append(label)
#         if len(self.pred_buffer) > 5:
#             self.pred_buffer.pop(0)
#
#         # majority vote (stability boost)
#         final_label = max(set(self.pred_buffer),
#                           key=self.pred_buffer.count)
#
#         return str(final_label)

# import numpy as np
# import cv2
# import os
# import joblib
# import mediapipe as mp
# import time
# from scipy.spatial.distance import cdist
# from dotenv import load_dotenv
#
# load_dotenv()
#
# KEYFRAME_K = int(os.getenv("HYPERPARAMETER_K", 30))
# WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 120))
# IDLE_MOTION_THRESHOLD = float(os.getenv("IDLE_MOTION_THRESHOLD", 0.01))
#
# # ── must mirror fsl_svm.py exactly ───────────────────────────────────────────
# _MCP_IDX = [1, 5, 9, 13, 17]
#
#
# def _normalise_hand(hand: np.ndarray) -> np.ndarray:
#     if np.all(hand == 0.0):
#         return hand
#     hand = hand - hand[0]
#     scale = np.linalg.norm(hand[_MCP_IDX], axis=1).mean()
#     if scale > 1e-6:
#         hand = hand / scale
#     return hand
#
#
# def _normalise_point_cloud(point_cloud: list) -> np.ndarray:
#     cloud = np.array(point_cloud, dtype=np.float32)
#     cloud[:21] = _normalise_hand(cloud[:21])
#     cloud[21:] = _normalise_hand(cloud[21:])
#     return cloud
# # ─────────────────────────────────────────────────────────────────────────────
#
#
# class FslSvmInfer:
#     def __init__(self, model_path):
#         data = joblib.load(model_path)
#         self.svm    = data["model"]
#         self.scaler = data["scaler"]
#         self.mp     = self._create_mp_task_video()
#
#         # buffers
#         self.buffer = []
#         self.motion = []
#         self.prev   = None
#
#         # real-time timing
#         self.prev_time = None
#
#         # EMA motion smoothing
#         self.motion_ema = 0.0
#         self.ema_alpha  = 0.3
#
#         # vote smoothing
#         self.pred_buffer = []
#
#     # ── MediaPipe setup ───────────────────────────────────────────────────────
#
#     def _create_mp_task_video(self):
#         BaseOptions           = mp.tasks.BaseOptions
#         HandLandmarker        = mp.tasks.vision.HandLandmarker
#         HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
#         VisionRunningMode     = mp.tasks.vision.RunningMode
#
#         options = HandLandmarkerOptions(
#             base_options=BaseOptions(model_asset_path="ml/hand_landmarker.task"),
#             running_mode=VisionRunningMode.VIDEO,
#             num_hands=2
#         )
#         return HandLandmarker.create_from_options(options)
#
#     # ── Landmark extraction ───────────────────────────────────────────────────
#
#     def _extract_landmarks(self, frame, timestamp_ms):
#         if frame is None or frame.size == 0:
#             return None
#
#         frame    = cv2.resize(frame, (320, 240))
#         frame    = cv2.flip(frame, 1)
#         rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#         result   = self.mp.detect_for_video(mp_image, int(timestamp_ms))
#
#         left  = [[0, 0, 0]] * 21
#         right = [[0, 0, 0]] * 21
#
#         if result and result.hand_landmarks:
#             for i, hand in enumerate(result.hand_landmarks):
#                 hand_type = result.handedness[i][0].category_name
#                 pts = [[lm.x, lm.y, lm.z] for lm in hand]
#                 if hand_type == "Left":
#                     left = pts
#                 else:
#                     right = pts
#
#         return left + right
#
#     # ── Chamfer distance ──────────────────────────────────────────────────────
#
#     def _chamfer(self, a, b):
#         a = np.array(a)
#         b = np.array(b)
#         return (
#             cdist(a, b).min(axis=1).mean() +
#             cdist(b, a).min(axis=1).mean()
#         )
#
#     # ── Keyframe selection ────────────────────────────────────────────────────
#
#     def _keyframes(self, motion):
#         k = KEYFRAME_K
#         n = len(motion)
#
#         if n < 2:
#             return list(range(n))
#
#         idx = {0, n - 1}
#         for i in range(1, n - 1):
#             if motion[i] > motion[i - 1] and motion[i] > motion[i + 1]:
#                 idx.add(i)
#
#         idx.update(np.argsort(motion)[-k:])
#         idx = sorted(idx)
#
#         while len(idx) < k:
#             idx.append(idx[-1])
#
#         return idx[:k]
#
#     # ── Feature extraction (mirrors training exactly) ─────────────────────────
#
#     def _features(self, frames, motion):
#         """
#         Normalise each keyframe's point cloud relative to the wrist, then
#         build pose + velocity blocks — identical to training.
#
#         Output length: (2K - 1) * 126
#         """
#         keyframes = self._keyframes(motion)
#
#         norm_frames = np.array([
#             _normalise_point_cloud(frames[i]).flatten()   # (126,)
#             for i in keyframes
#         ], dtype=np.float32)                              # (K, 126)
#
#         pose_block     = norm_frames.flatten()                   # K * 126
#         velocity_block = np.diff(norm_frames, axis=0).flatten()  # (K-1) * 126
#
#         return np.concatenate([pose_block, velocity_block])      # (2K-1) * 126
#
#     # ── Reset ─────────────────────────────────────────────────────────────────
#
#     def reset(self):
#         self.buffer      = []
#         self.motion      = []
#         self.prev        = None
#         self.prev_time   = None
#         self.motion_ema  = 0.0
#         self.pred_buffer = []
#
#     # ── Main prediction ───────────────────────────────────────────────────────
#
#     def predict(self, frame):
#         now = time.time() * 1000   # real-time timestamp for MediaPipe VIDEO mode
#
#         pc = self._extract_landmarks(frame, now)
#         if pc is None:
#             return None
#
#         if np.all(np.array(pc) == 0.0):
#             return None
#
#         if self.prev_time is not None:
#             dt = now - self.prev_time   # available for future use
#         self.prev_time = now
#
#         self.buffer.append(pc)
#
#         motion_val = self._chamfer(self.prev, pc) if self.prev is not None else 0.0
#         self.prev  = pc
#
#         self.motion_ema = (
#             self.ema_alpha * motion_val +
#             (1 - self.ema_alpha) * self.motion_ema
#         )
#         self.motion.append(self.motion_ema)
#
#         # sliding window
#         if len(self.buffer) > WINDOW_SIZE:
#             self.buffer.pop(0)
#             self.motion.pop(0)
#
#         if len(self.buffer) < KEYFRAME_K:
#             return None
#
#         # idle filter
#         if np.mean(self.motion[-10:]) < IDLE_MOTION_THRESHOLD * 0.7:
#             return None
#
#         feat  = self._features(self.buffer, self.motion)
#         feat  = self.scaler.transform([feat])
#
#         probs      = self.svm.predict_proba(feat)[0]
#         idx        = np.argmax(probs)
#         label      = self.svm.classes_[idx]
#         confidence = probs[idx]
#
#         print("CONFIDENCE:", float(confidence))
#         print("LABEL:", str(label))
#
#         # majority vote over last 5 predictions
#         self.pred_buffer.append(label)
#         if len(self.pred_buffer) > 5:
#             self.pred_buffer.pop(0)
#
#         return str(max(set(self.pred_buffer), key=self.pred_buffer.count))
#
import numpy as np
import cv2
import os
import joblib
import mediapipe as mp
from scipy.spatial.distance import cdist
from dotenv import load_dotenv

load_dotenv()

KEYFRAME_K            = int(os.getenv("HYPERPARAMETER_K", 30))
WINDOW_SIZE           = int(os.getenv("WINDOW_SIZE", 120))
IDLE_MOTION_THRESHOLD = float(os.getenv("IDLE_MOTION_THRESHOLD", 0.01))

# Must mirror fsl_svm.py exactly
_MCP_IDX = [1, 5, 9, 13, 17]

# How many consecutive low-motion frames = gesture is done
IDLE_PATIENCE = 12


def _normalise_hand(hand: np.ndarray) -> np.ndarray:
    if np.all(hand == 0.0):
        return hand
    hand = hand - hand[0]
    scale = np.linalg.norm(hand[_MCP_IDX], axis=1).mean()
    if scale > 1e-6:
        hand = hand / scale
    return hand


def _normalise_point_cloud(point_cloud: list) -> np.ndarray:
    cloud = np.array(point_cloud, dtype=np.float32)
    cloud[:21] = _normalise_hand(cloud[:21])
    cloud[21:] = _normalise_hand(cloud[21:])
    return cloud


# ── State machine states ───────────────────────────────────────────────────
_IDLE       = "IDLE"
_SIGNING    = "SIGNING"
_PREDICTING = "PREDICTING"


class FslSvmInfer:
    def __init__(self, model_path):
        data        = joblib.load(model_path)
        self.svm    = data["model"]
        self.scaler = data["scaler"]
        self.mp     = self._create_mp_task_video()

        # frame counter drives MediaPipe timestamps — mirrors training exactly
        self._frame_idx = 0
        self._fps       = 30                        # assumed; matches training default
        self._step_ms   = int(1000 / self._fps)

        # gesture buffers
        self.buffer = []
        self.motion = []
        self.prev   = None

        # EMA smoothing
        self.motion_ema = 0.0
        self.ema_alpha  = 0.3

        # state machine
        self.state        = _IDLE
        self.idle_counter = 0           # consecutive low-motion frames while SIGNING

    # ── MediaPipe setup ────────────────────────────────────────────────────

    def _create_mp_task_video(self):
        BaseOptions           = mp.tasks.BaseOptions
        HandLandmarker        = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode     = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="ml/hand_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2
        )
        return HandLandmarker.create_from_options(options)

    # ── Landmark extraction ────────────────────────────────────────────────

    def _extract_landmarks(self, frame):
        """
        Uses a frame-counter-based timestamp to match training,
        where timestamps were synthetic (frame_idx × step_ms).
        """
        if frame is None or frame.size == 0:
            return None

        timestamp_ms = self._frame_idx * self._step_ms
        self._frame_idx += 1

        frame    = cv2.resize(frame, (320, 240))
        frame    = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self.mp.detect_for_video(mp_image, int(timestamp_ms))

        left  = [[0, 0, 0]] * 21
        right = [[0, 0, 0]] * 21

        if result and result.hand_landmarks:
            for i, hand in enumerate(result.hand_landmarks):
                hand_type = result.handedness[i][0].category_name
                pts = [[lm.x, lm.y, lm.z] for lm in hand]
                if hand_type == "Left":
                    left = pts
                else:
                    right = pts

        return left + right

    # ── Chamfer distance ───────────────────────────────────────────────────

    def _chamfer(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return (
            cdist(a, b).min(axis=1).mean() +
            cdist(b, a).min(axis=1).mean()
        )

    # ── Keyframe selection ─────────────────────────────────────────────────

    def _keyframes(self, motion):
        k = KEYFRAME_K
        n = len(motion)
        if n < 2:
            return list(range(n))

        idx = {0, n - 1}
        for i in range(1, n - 1):
            if motion[i] > motion[i - 1] and motion[i] > motion[i + 1]:
                idx.add(i)

        idx.update(np.argsort(motion)[-k:])
        idx = sorted(idx)
        while len(idx) < k:
            idx.append(idx[-1])
        return idx[:k]

    # ── Feature extraction ─────────────────────────────────────────────────

    def _features(self, frames, motion):
        keyframes = self._keyframes(motion)
        norm_frames = np.array([
            _normalise_point_cloud(frames[i]).flatten()
            for i in keyframes
        ], dtype=np.float32)

        pose_block     = norm_frames.flatten()
        velocity_block = np.diff(norm_frames, axis=0).flatten()
        return np.concatenate([pose_block, velocity_block])

    # ── Reset buffers ──────────────────────────────────────────────────────

    def reset(self):
        self.buffer       = []
        self.motion       = []
        self.prev         = None
        self.motion_ema   = 0.0
        self.state        = _IDLE
        self.idle_counter = 0
        print("[INFO] Buffers reset, back to IDLE")

    # ── Main predict (call once per incoming frame) ────────────────────────

    def predict(self, frame):
        """
        Feed one frame. Returns a label string only when a complete gesture
        has been detected and classified. Returns None otherwise.
        """
        pc = self._extract_landmarks(frame)
        if pc is None:
            return None

        has_hand = not np.all(np.array(pc) == 0.0)

        # ── motion estimate ──
        motion_val = self._chamfer(self.prev, pc) if self.prev is not None else 0.0
        self.prev  = pc

        self.motion_ema = (
            self.ema_alpha * motion_val +
            (1 - self.ema_alpha) * self.motion_ema
        )
        is_moving = self.motion_ema > IDLE_MOTION_THRESHOLD

        # ── state machine ──────────────────────────────────────────────────
        if self.state == _IDLE:
            if has_hand and is_moving:
                self.state        = _SIGNING
                self.idle_counter = 0
                self.buffer       = [pc]
                self.motion       = [self.motion_ema]
                print("[INFO] State → SIGNING")

        elif self.state == _SIGNING:
            self.buffer.append(pc)
            self.motion.append(self.motion_ema)

            # cap buffer to WINDOW_SIZE
            if len(self.buffer) > WINDOW_SIZE:
                self.buffer.pop(0)
                self.motion.pop(0)

            if not is_moving or not has_hand:
                self.idle_counter += 1
            else:
                self.idle_counter = 0

            if self.idle_counter >= IDLE_PATIENCE:
                # gesture complete — move to prediction
                self.state = _PREDICTING
                print("[INFO] State → PREDICTING")

        if self.state == _PREDICTING:
            if len(self.buffer) < KEYFRAME_K:
                print(f"[WARN] Buffer too short ({len(self.buffer)} frames), discarding")
                self.reset()
                return None

            print("[INFO] Processing gesture...")
            feat  = self._features(self.buffer, self.motion)
            feat  = self.scaler.transform([feat])

            probs      = self.svm.predict_proba(feat)[0]
            idx        = np.argmax(probs)
            label      = self.svm.classes_[idx]
            confidence = float(probs[idx])

            print(f"[INFO] Prediction: {label} (confidence: {confidence:.2f})")

            self.reset()
            return label

        return None
