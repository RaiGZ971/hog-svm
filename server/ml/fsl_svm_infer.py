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
    """
    Mirrors fsl_svm.py _normalise_hand exactly.
    All-zero hands are returned unchanged.
    """
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


# State machine states
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

        # state machine
        self.state        = _IDLE
        self.idle_counter = 0

    def _chamfer(self, a, b):
        if a is None or b is None:
            return 0.0
        pc1 = np.array(a)
        pc2 = np.array(b)
        d1  = cdist(pc1, pc2).min(axis=1).mean()
        d2  = cdist(pc2, pc1).min(axis=1).mean()
        return d1 + d2

    def _keyframes(self, motion):
        """
        Select exactly KEYFRAME_K indices from motion.

        Priority order (highest motion moments win):
          1. Boundary frames {0, n-1}
          2. Local motion peaks
          3. Top-k highest motion frames

        If the union exceeds k we keep the k frames with the highest motion
        values — this is consistent with what training produced for short
        trimmed clips where the union never exceeded k, and prevents the
        feature vector from growing beyond (2k-1)*126 on longer live buffers.
        """
        k = KEYFRAME_K
        n = len(motion)
        motion_arr = np.array(motion)

        candidates = set()
        candidates.add(0)
        candidates.add(n - 1)

        for t in range(1, n - 1):
            if motion[t] > motion[t - 1] and motion[t] > motion[t + 1]:
                candidates.add(t)

        candidates.update(np.argsort(motion)[-k:].tolist())

        # If we have more than k candidates, keep the k with highest motion.
        # If fewer, pad by repeating the last index.
        candidates = sorted(candidates)
        if len(candidates) > k:
            candidates = sorted(
                candidates,
                key=lambda i: motion_arr[i],
                reverse=True
            )[:k]
            candidates = sorted(candidates)  # restore temporal order

        while len(candidates) < k:
            candidates.append(candidates[-1])

        return candidates

    def _features(self, frames, motion):
        keyframes = self._keyframes(motion)

        norm_frames = np.array([
            _normalise_point_cloud(frames[i]).flatten()   # (126,)
            for i in keyframes
        ], dtype=np.float32)                              # (K, 126)

        pose_block     = norm_frames.flatten()                  # K * 126
        velocity_block = np.diff(norm_frames, axis=0).flatten() # (K-1) * 126

        return np.concatenate([pose_block, velocity_block])     # (2K-1) * 126

    # Reset buffers
    def reset(self):
        self.buffer       = []
        self.motion       = []
        self.prev         = None
        self.state        = _IDLE
        self.idle_counter = 0
        print("[INFO] Buffers reset, back to IDLE")

    # Main predict

    def predict(self, landmarks: list) -> str | None:
        pc       = landmarks
        has_hand = not np.all(np.array(pc) == 0.0)

        motion_val = self._chamfer(self.prev, pc) if self.prev is not None else 0.0
        self.prev  = pc

        is_moving = motion_val > IDLE_MOTION_THRESHOLD

        # State machine 
        if self.state == _IDLE:
            if has_hand and is_moving:
                self.state        = _SIGNING
                self.idle_counter = 0
                self.buffer       = [pc]
                self.motion       = [motion_val]
                print("[INFO] State → SIGNING")

        elif self.state == _SIGNING:
            self.buffer.append(pc)
            self.motion.append(motion_val)

            # Cap buffer to WINDOW_SIZE (keeps most recent frames)
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

            print(f"[INFO] Processing gesture — {len(self.buffer)} frames in buffer...")
            feat  = self._features(self.buffer, self.motion)
            feat  = self.scaler.transform([feat])

            label      = self.svm.predict(feat)[0]
            probs      = self.svm.predict_proba(feat)[0]
            pred_idx   = np.where(self.svm.classes_ == label)[0][0]
            confidence = float(probs[pred_idx])

            print(f"[INFO] Prediction: {label} (confidence: {confidence:.2f})")

            self.reset()
            return label

        return None
