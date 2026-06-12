import numpy as np
import os
import joblib
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

        # gesture buffers
        self.buffer = []
        self.motion = []
        self.prev   = None

        # EMA smoothing
        self.motion_ema = 0.0
        self.ema_alpha  = 0.3

        # state machine
        self.state        = _IDLE
        self.idle_counter = 0

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

    # ── Main predict (call with landmark array from frontend) ──────────────

    def predict(self, landmarks: list) -> str | None:
        """
        Feed one frame's landmark array (42 points × 3 coords = list of 42 [x,y,z]).
        landmarks[0:21]  = left hand
        landmarks[21:42] = right hand
        Returns a label string only when a complete gesture has been detected
        and classified. Returns None otherwise.
        """
        pc = landmarks  # already [42][3] from the frontend

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
