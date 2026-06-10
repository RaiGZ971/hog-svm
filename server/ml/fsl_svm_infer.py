import numpy as np
import cv2
import os
import joblib
import mediapipe as mp
import time
from scipy.spatial.distance import cdist
from dotenv import load_dotenv

load_dotenv()

KEYFRAME_K = int(os.getenv("HYPERPARAMETER_K", 30))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 120))
IDLE_MOTION_THRESHOLD = float(os.getenv("IDLE_MOTION_THRESHOLD", 0.01))


class FslSvmInfer:
    def __init__(self, model_path):
        data = joblib.load(model_path)

        self.svm = data["model"]
        self.scaler = data["scaler"]

        self.mp = self._create_mp_task_video()

        # buffers
        self.buffer = []
        self.motion = []
        self.prev = None

        # real-time timing (FIXED)
        self.prev_time = None

        # smoothing
        self.motion_ema = 0.0
        self.ema_alpha = 0.3

        # optional stability (vote smoothing)
        self.pred_buffer = []

    # -------------------------
    # MediaPipe setup
    # -------------------------
    def _create_mp_task_video(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="ml/hand_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2
        )

        return HandLandmarker.create_from_options(options)

    # -------------------------
    # Landmark extraction
    # -------------------------
    def _extract_landmarks(self, frame, timestamp_ms):
        if frame is None or frame.size == 0:
            return None

        frame = cv2.resize(frame, (320, 240))
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.mp.detect_for_video(mp_image, int(timestamp_ms))

        left = [[0, 0, 0]] * 21
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

    # -------------------------
    # Chamfer distance (motion)
    # -------------------------
    def _chamfer(self, a, b):
        a = np.array(a)
        b = np.array(b)

        return (
            cdist(a, b).min(axis=1).mean()
            + cdist(b, a).min(axis=1).mean()
        )

    # -------------------------
    # Keyframes
    # -------------------------
    def _keyframes(self, motion):
        k = KEYFRAME_K
        n = len(motion)

        if n < 2:
            return list(range(n))

        idx = set([0, n - 1])

        for i in range(1, n - 1):
            if motion[i] > motion[i - 1] and motion[i] > motion[i + 1]:
                idx.add(i)

        idx.update(np.argsort(motion)[-k:])
        idx = sorted(idx)

        while len(idx) < k:
            idx.append(idx[-1])

        return idx[:k]

    # -------------------------
    # Feature extraction
    # -------------------------
    def _features(self, frames, motion):
        selected = [frames[i] for i in self._keyframes(motion)]
        selected = np.array(selected).reshape(len(selected), -1)

        f = []
        f.extend(selected.mean(axis=0))
        f.extend(selected.std(axis=0))
        f.extend(selected.min(axis=0))
        f.extend(selected.max(axis=0))

        return np.array(f)

    # -------------------------
    # Reset state
    # -------------------------
    def reset(self):
        self.buffer = []
        self.motion = []
        self.prev = None
        self.prev_time = None
        self.motion_ema = 0.0
        self.pred_buffer = []

    # -------------------------
    # MAIN PREDICTION
    # -------------------------
    def predict(self, frame):
        now = time.time() * 1000  # REAL timestamp

        pc = self._extract_landmarks(frame, now)
        if pc is None:
            return None

        # skip invalid hand
        has_hand = not np.all(np.array(pc) == 0.0)
        if not has_hand:
            return None

        # time delta awareness (optional future use)
        if self.prev_time is not None:
            dt = now - self.prev_time
        self.prev_time = now

        self.buffer.append(pc)

        # motion
        if self.prev is None:
            motion_val = 0.0
        else:
            motion_val = self._chamfer(self.prev, pc)

        self.prev = pc

        # EMA smoothing (IMPORTANT FIX)
        self.motion_ema = (
            self.ema_alpha * motion_val +
            (1 - self.ema_alpha) * self.motion_ema
        )

        self.motion.append(self.motion_ema)

        # sliding window
        if len(self.buffer) > WINDOW_SIZE:
            self.buffer.pop(0)
            self.motion.pop(0)

        if len(self.buffer) < KEYFRAME_K:
            return None

        # adaptive idle filter (LESS STRICT)
        recent_motion = np.mean(self.motion[-10:])

        if recent_motion < IDLE_MOTION_THRESHOLD * 0.7:
            return None

        # feature extraction
        feat = self._features(self.buffer, self.motion)
        feat = self.scaler.transform([feat])

        print("FEATURE MEAN:", np.mean(feat), "STD:", np.std(feat))
        print("UNIQUE BUFFER VARIANCE:", np.var(self.buffer))

        probs = self.svm.predict_proba(feat)[0]
        idx = np.argmax(probs)

        label = self.svm.classes_[idx]
        confidence = probs[idx]

        print("CONFIDENCE:", float(confidence))
        print("LABEL:", str(label))

        # -------------------------
        # simple prediction smoothing
        # -------------------------
        self.pred_buffer.append(label)
        if len(self.pred_buffer) > 5:
            self.pred_buffer.pop(0)

        # majority vote (stability boost)
        final_label = max(set(self.pred_buffer),
                          key=self.pred_buffer.count)

        return str(final_label)
#
#
#
#
#
##
##
#import numpy as np
#import cv2
#import os
#import joblib
#import mediapipe as mp
#import time
#from scipy.spatial.distance import cdist
#from dotenv import load_dotenv
#
#load_dotenv()
#
#KEYFRAME_K = int(os.getenv("HYPERPARAMETER_K", 30))
#WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 120))
#IDLE_MOTION_THRESHOLD = float(os.getenv("IDLE_MOTION_THRESHOLD", 0.01))
#
#
#class FslSvmInfer:
#    def __init__(self, model_path):
#        data = joblib.load(model_path)
#
#        self.svm = data["model"]
#        self.scaler = data["scaler"]
#
#        self.mp = self._create_mp_task_video()
#
#        # buffers
#        self.buffer = []
#        self.motion = []
#        self.prev = None
#
#        # timing
#        self.prev_time = None
#
#        # smoothing
#        self.motion_ema = 0.0
#        self.ema_alpha = 0.3
#
#        # prediction smoothing
#        self.pred_buffer = []
#
#        # -------------------------
#        # GESTURE STATE MACHINE
#        # -------------------------
#        self.is_collecting = False
#        self.silence_counter = 0
#        self.max_silence = 8
#        self.min_motion_to_start = IDLE_MOTION_THRESHOLD * 1.2
#
#    # -------------------------
#    # MediaPipe setup
#    # -------------------------
#    def _create_mp_task_video(self):
#        BaseOptions = mp.tasks.BaseOptions
#        HandLandmarker = mp.tasks.vision.HandLandmarker
#        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
#        VisionRunningMode = mp.tasks.vision.RunningMode
#
#        options = HandLandmarkerOptions(
#            base_options=BaseOptions(model_asset_path="ml/hand_landmarker.task"),
#            running_mode=VisionRunningMode.VIDEO,
#            num_hands=2
#        )
#
#        return HandLandmarker.create_from_options(options)
#
#    # -------------------------
#    # Landmark extraction
#    # -------------------------
#    def _extract_landmarks(self, frame, timestamp_ms):
#        if frame is None or frame.size == 0:
#            return None
#
#        frame = cv2.resize(frame, (320, 240))
#        frame = cv2.flip(frame, 1)
#
#        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#
#        result = self.mp.detect_for_video(mp_image, int(timestamp_ms))
#
#        left = [[0, 0, 0]] * 21
#        right = [[0, 0, 0]] * 21
#
#        if result and result.hand_landmarks:
#            for i, hand in enumerate(result.hand_landmarks):
#                hand_type = result.handedness[i][0].category_name
#                pts = [[lm.x, lm.y, lm.z] for lm in hand]
#
#                if hand_type == "Left":
#                    left = pts
#                else:
#                    right = pts
#
#        return left + right
#
#    # -------------------------
#    # Chamfer motion distance
#    # -------------------------
#    def _chamfer(self, a, b):
#        a = np.array(a)
#        b = np.array(b)
#
#        return (
#            cdist(a, b).min(axis=1).mean()
#            + cdist(b, a).min(axis=1).mean()
#        )
#
#    # -------------------------
#    # Keyframes
#    # -------------------------
#    def _keyframes(self, motion):
#        k = KEYFRAME_K
#        n = len(motion)
#
#        if n < 2:
#            return list(range(n))
#
#        idx = set([0, n - 1])
#
#        for i in range(1, n - 1):
#            if motion[i] > motion[i - 1] and motion[i] > motion[i + 1]:
#                idx.add(i)
#
#        idx.update(np.argsort(motion)[-k:])
#        idx = sorted(idx)
#
#        while len(idx) < k:
#            idx.append(idx[-1])
#
#        return idx[:k]
#
#    # -------------------------
#    # Feature extraction
#    # -------------------------
#    def _features(self, frames, motion):
#        selected = [frames[i] for i in self._keyframes(motion)]
#        selected = np.array(selected).reshape(len(selected), -1)
#
#        f = []
#        f.extend(selected.mean(axis=0))
#        f.extend(selected.std(axis=0))
#        f.extend(selected.min(axis=0))
#        f.extend(selected.max(axis=0))
#
#        return np.array(f)
#
#    # -------------------------
#    # Reset state
#    # -------------------------
#    def reset(self):
#        self.buffer = []
#        self.motion = []
#        self.prev = None
#        self.prev_time = None
#        self.motion_ema = 0.0
#        self.pred_buffer = []
#
#        self.is_collecting = False
#        self.silence_counter = 0
#
#    # -------------------------
#    # MAIN PREDICTION (FIXED)
#    # -------------------------
#    def predict(self, frame):
#        now = time.time() * 1000
#
#        pc = self._extract_landmarks(frame, now)
#        if pc is None:
#            return None
#
#        has_hand = not np.all(np.array(pc) == 0.0)
#        if not has_hand:
#            return None
#
#        # motion
#        if self.prev is None:
#            motion_val = 0.0
#        else:
#            motion_val = self._chamfer(self.prev, pc)
#
#        self.prev = pc
#
#        # EMA smoothing
#        self.motion_ema = (
#            self.ema_alpha * motion_val +
#            (1 - self.ema_alpha) * self.motion_ema
#        )
#
#        self.motion.append(self.motion_ema)
#
#        # -------------------------
#        # WINDOW LIMIT (safety)
#        # -------------------------
#        if len(self.motion) > WINDOW_SIZE:
#            self.motion.pop(0)
#
#        recent_motion = np.mean(self.motion[-10:])
#
#        # -------------------------
#        # START GESTURE
#        # -------------------------
#        if not self.is_collecting:
#            if recent_motion > self.min_motion_to_start:
#                self.is_collecting = True
#                self.buffer = []
#                self.motion = []
#            else:
#                return None
#
#        # collect frames
#        self.buffer.append(pc)
#
#        # -------------------------
#        # END GESTURE DETECTION
#        # -------------------------
#        if recent_motion < IDLE_MOTION_THRESHOLD:
#            self.silence_counter += 1
#        else:
#            self.silence_counter = 0
#
#        # still signing → WAIT
#        if self.silence_counter < self.max_silence:
#            return None
#
#        # gesture ended
#        self.is_collecting = False
#        self.silence_counter = 0
#
#        # -------------------------
#        # CLASSIFICATION (ONLY HERE)
#        # -------------------------
#        if len(self.buffer) < KEYFRAME_K:
#            return None
#
#        feat = self._features(self.buffer, self.motion)
#        feat = self.scaler.transform([feat])
#
#        probs = self.svm.predict_proba(feat)[0]
#        idx = np.argmax(probs)
#
#        label = self.svm.classes_[idx]
#        confidence = probs[idx]
#
#        print("FINAL CONFIDENCE:", float(confidence))
#        print("LABEL:", label)
#
#        if confidence < 0.1:
#            return None
#
#        self.pred_buffer.append(label)
#        if len(self.pred_buffer) > 5:
#            self.pred_buffer.pop(0)
#
#        return max(set(self.pred_buffer), key=self.pred_buffer.count)
