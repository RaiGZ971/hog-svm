# import numpy as np
# import pandas as pd
# from tqdm import tqdm
#
# import mediapipe as mp
# import cv2
# from scipy.spatial.distance import cdist
# import joblib
#
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
#
# import os
# from dotenv import load_dotenv
#
# load_dotenv()
#
# KEYFRAME_K = int(os.getenv("HYPERPARAMETER_K", 30))
#
#
# class FslSvm:
#     def __init__(self, training, testing):
#         self.raw_train_data = pd.read_csv(training)
#         self.raw_test_data = pd.read_csv(testing)
#
#         self.X_train, self.y_train = [], []
#         self.X_test, self.y_test = [], []
#
#         self.svm = None
#         self.scaler = StandardScaler()
#
#         self.model_path = os.path.join(
#             os.path.dirname(__file__),
#             "hand_landmarker.task"
#         )
#
#         self.mp = None 
#
#     def _create_mp_task_video(self):
#         BaseOptions = mp.tasks.BaseOptions
#         HandLandmarker = mp.tasks.vision.HandLandmarker
#         HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
#         VisionRunningMode = mp.tasks.vision.RunningMode
#
#         options = HandLandmarkerOptions(
#             base_options=BaseOptions(model_asset_path=self.model_path),
#             running_mode=VisionRunningMode.VIDEO,
#             num_hands=2
#         )
#
#         return HandLandmarker.create_from_options(options)
#
#     def _extract_landmarker_point_cloud(self, frame, timestamp_ms):
#         """
#         Extract 42-point (left + right hand) point cloud from a single frame.
#         Uses timestamp_ms that is local to the current video so MediaPipe VIDEO
#         mode always receives strictly-increasing timestamps.
#         """
#         left_hand_points = [[0.0, 0.0, 0.0]] * 21
#         right_hand_points = [[0.0, 0.0, 0.0]] * 21
#
#         frame = cv2.resize(frame, (320, 240))
#
#         frame = cv2.flip(frame, 1)
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#
#         results = self.mp.detect_for_video(mp_image, int(timestamp_ms))
#
#         for idx, hand in enumerate(results.hand_landmarks):
#             hand_type = results.handedness[idx][0].category_name
#             points = [[lm.x, lm.y, lm.z] for lm in hand]
#
#             if hand_type == "Left":
#                 left_hand_points = points
#             else:
#                 right_hand_points = points
#
#         return left_hand_points + right_hand_points
#
#     def _chamfer_distance(self, prev_point_cloud, pres_point_cloud):
#         if not prev_point_cloud or not pres_point_cloud:
#             return 0.0
#
#         pc1 = np.array(prev_point_cloud)
#         pc2 = np.array(pres_point_cloud)
#
#         d1 = cdist(pc1, pc2).min(axis=1).mean()
#         d2 = cdist(pc2, pc1).min(axis=1).mean()
#
#         return d1 + d2
#
#     def _select_keyframes(self, motion):
#         """
#         Select keyframes and always return exactly KEYFRAME_K indices so
#         every feature vector has the same length.
#         """
#         k = KEYFRAME_K
#         n = len(motion)
#
#         keyframes = set()
#         keyframes.add(0)
#         keyframes.add(n - 1)
#
#         for t in range(1, n - 1):
#             if motion[t] > motion[t - 1] and motion[t] > motion[t + 1]:
#                 keyframes.add(t)
#
#         top_k = np.argsort(motion)[-k:]
#         keyframes.update(top_k)
#
#         keyframes = sorted(keyframes)
#
#         # FIX #3 – pad by repeating the last frame, or truncate, to exactly k
#         while len(keyframes) < k:
#             keyframes.append(keyframes[-1])
#
#         return keyframes[:k]
#
#     def _extract_svm_features(self, keyframes, point_clouds):
#         """
#         Always produces a feature vector of fixed length:
#             KEYFRAME_K * 42 * 3 * 4 = KEYFRAME_K * 504
#         """
#         selected = [point_clouds[i] for i in keyframes]
#         selected = np.array(selected)               # (k, 42, 3)
#         flat = selected.reshape(len(selected), -1)  # (k, 126)
#
#         features = []
#         features.extend(flat.mean(axis=0))   # average pose
#         features.extend(flat.std(axis=0))    # motion variation
#         features.extend(flat.min(axis=0))    # extreme positions (min)
#         features.extend(flat.max(axis=0))    # extreme positions (max)
#
#         return np.array(features)            # always length k*126*4 = k*504
#
#     def _video_to_features(self, video_path):
#         self.mp = self._create_mp_task_video()
#
#         cap = cv2.VideoCapture(video_path)
#
#         if not cap.isOpened():
#             print(f"[WARN] Could not open: {video_path}")
#             return None
#
#         fps = cap.get(cv2.CAP_PROP_FPS) or 30
#         step_ms = int(1000 / fps)
#
#         frame_timestamp_ms = 0
#
#         motion = []
#         point_clouds = []
#         hand_detected = []
#         prev_point_cloud = None
#
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             pres_point_cloud = self._extract_landmarker_point_cloud(
#                 frame, frame_timestamp_ms
#             )
#
#             if pres_point_cloud is not None:
#                 flat = np.array(pres_point_cloud)
#                 has_hand = not np.all(flat == 0.0)
#                 hand_detected.append(has_hand)
#
#                 point_clouds.append(pres_point_cloud)
#
#                 if prev_point_cloud is not None:
#                     m = self._chamfer_distance(prev_point_cloud, pres_point_cloud)
#                     motion.append(m)
#                 else:
#                     motion.append(0.0)
#
#                 prev_point_cloud = pres_point_cloud
#
#             frame_timestamp_ms += step_ms
#
#         cap.release()
#
#         if any(hand_detected):
#             first = next(i for i, v in enumerate(hand_detected) if v)
#             last  = len(hand_detected) - 1 - next(
#                 i for i, v in enumerate(reversed(hand_detected)) if v
#             )
#             point_clouds = point_clouds[first:last + 1]
#             motion       = motion[first:last + 1]
#             print(f"[INFO] Hand present in frames {first}–{last} "
#                   f"({last - first + 1} frames kept out of {len(hand_detected)})")
#         else:
#             print(f"[WARN] No hand detected in any frame: {video_path}")
#             return None
#
#         if len(point_clouds) < KEYFRAME_K:
#             print(f"[WARN] Not enough hand frames in: {video_path} "
#                   f"(got {len(point_clouds)}, need {KEYFRAME_K})")
#             return None
#
#         keyframes = self._select_keyframes(motion)
#         features = self._extract_svm_features(keyframes, point_clouds)
#
#         return features
#
#     def _build_Xy(self):
#         train_features, train_labels = [], []
#         test_features, test_labels = [], []
#
#         for row in tqdm(self.raw_train_data.itertuples(),
#                         total=len(self.raw_train_data),
#                         desc="Train"):
#             feature = self._video_to_features(row.vid_path)
#             if feature is not None:
#                 train_features.append(feature)
#                 train_labels.append(row.label)
#
#         for row in tqdm(self.raw_test_data.itertuples(),
#                         total=len(self.raw_test_data),
#                         desc="Test"):
#             feature = self._video_to_features(row.vid_path)
#             if feature is not None:
#                 test_features.append(feature)
#                 test_labels.append(row.label)
#
#         self.X_train = np.array(train_features)
#         self.y_train = np.array(train_labels)
#         self.X_test = np.array(test_features)
#         self.y_test = np.array(test_labels)
#
#         print("Train shape:", self.X_train.shape)
#         print("Test shape: ", self.X_test.shape)
#
#         self.X_train = self.scaler.fit_transform(self.X_train)
#         self.X_test = self.scaler.transform(self.X_test)
#
#     def train_svm_model(self):
#         if len(self.X_train) == 0:
#             self._build_Xy()
#
#         param_grid = {
#             "C": [0.1, 1, 10, 100],
#             "gamma": ["scale", 0.01, 0.001, 0.0001],
#             "kernel": ["rbf"]
#         }
#
#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
#         base_model = SVC(probability=True)
#
#         grid = GridSearchCV(
#             estimator=base_model,
#             param_grid=param_grid,
#             cv=cv,
#             scoring="accuracy",
#             n_jobs=-1,
#             verbose=1
#         )
#
#         grid.fit(self.X_train, self.y_train)
#
#         self.svm = grid.best_estimator_
#
#         print("SVM trained successfully.")
#         print("Best parameters:", grid.best_params_)
#         print("Best CV accuracy:", grid.best_score_)
#
#     def evaluate_svm_model(self):
#         if self.svm is None:
#             return
#
#         y_pred = self.svm.predict(self.X_test)
#
#         print(classification_report(self.y_test, y_pred, zero_division=0))
#
#         cm = confusion_matrix(self.y_test, y_pred)
#         ConfusionMatrixDisplay(cm, display_labels=self.svm.classes_).plot(
#             xticks_rotation="vertical"
#         )
#
#         plt.title("FSL-SVM (Optimized Landmark Model)")
#         plt.tight_layout()
#         plt.show()
#
#         print("Train Accuracy:", self.svm.score(self.X_train, self.y_train))
#         print("Test Accuracy: ", self.svm.score(self.X_test, self.y_test))
#
#     def run_svm_model(self, video_path):
#         feat = self._video_to_features(video_path)
#
#         print("FEATURE:", feat)
#         feat = self.scaler.transform([feat])
#
#         probabilities = self.svm.predict_proba(feat)[0]
#         max_prob_index = np.argmax(probabilities)
#         confidence = probabilities[max_prob_index]
#
#         pred_label = self.svm.classes_[max_prob_index]
#
#         print("CONFIDENCE: ", float(confidence))
#         print("LABEL: ", pred_label)
#
#     def save_svm_model(self, path="./models/fsl-svm-2-catv2.pkl"):
#         joblib.dump({"model": self.svm, "scaler": self.scaler}, path)
#
#     def load_svm_model(self, path="./models/fsl-svm-2-catv2.pkl"):
#         data = joblib.load(path)
#         self.svm = data["model"]
#         self.scaler = data["scaler"]
#


import numpy as np
import pandas as pd
from tqdm import tqdm

import mediapipe as mp
import cv2
from scipy.spatial.distance import cdist
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv

load_dotenv()

KEYFRAME_K = int(os.getenv("HYPERPARAMETER_K", 30))

_MCP_IDX = [1, 5, 9, 13, 17]   # knuckle indices per hand


def _normalise_hand(hand: np.ndarray) -> np.ndarray:
    """
    Normalise a single (21, 3) hand point cloud relative to its wrist.
    All-zero hands (not detected) are returned unchanged so the SVM can
    learn the absence of a hand as its own feature pattern.
    """
    if np.all(hand == 0.0):
        return hand

    hand = hand - hand[0]                                  # centre on wrist

    scale = np.linalg.norm(hand[_MCP_IDX], axis=1).mean() # avg knuckle dist
    if scale > 1e-6:
        hand = hand / scale

    return hand


def _normalise_point_cloud(point_cloud: list) -> np.ndarray:
    """
    Normalise the full 42-point cloud (left + right hand) independently.
    Returns a (42, 3) float32 array.
    """
    cloud = np.array(point_cloud, dtype=np.float32)
    cloud[:21] = _normalise_hand(cloud[:21])
    cloud[21:] = _normalise_hand(cloud[21:])
    return cloud

def _extract_svm_features(keyframes: list, point_clouds: list) -> np.ndarray:
    """
    Build a fixed-length feature vector from the selected keyframes.

    Parameters
    ----------
    keyframes    : list[int], length K — indices into point_clouds
    point_clouds : list of raw 42-point clouds (one per video frame)

    Returns
    -------
    np.ndarray, shape ((2K-1) * 126,)
    """
    norm_frames = np.array([
        _normalise_point_cloud(point_clouds[i]).flatten()   # (126,)
        for i in keyframes
    ], dtype=np.float32)                                    # (K, 126)

    pose_block     = norm_frames.flatten()                  # K * 126
    velocity_block = np.diff(norm_frames, axis=0).flatten() # (K-1) * 126

    return np.concatenate([pose_block, velocity_block])     # (2K-1) * 126


class FslSvm:
    def __init__(self, training, testing):
        self.raw_train_data = pd.read_csv(training)
        self.raw_test_data  = pd.read_csv(testing)

        self.X_train, self.y_train = [], []
        self.X_test,  self.y_test  = [], []

        self.svm    = None
        self.scaler = StandardScaler()

        self.model_path = os.path.join(
            os.path.dirname(__file__),
            "hand_landmarker.task"
        )

        self.mp = None

    def _create_mp_task_video(self):
        BaseOptions           = mp.tasks.BaseOptions
        HandLandmarker        = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode     = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2
        )
        return HandLandmarker.create_from_options(options)

    def _extract_landmarker_point_cloud(self, frame, timestamp_ms):
        """
        Extract 42-point (left + right hand) point cloud from a single frame.
        Uses timestamp_ms local to the current video so MediaPipe VIDEO mode
        always receives strictly-increasing timestamps.
        """
        left_hand_points  = [[0.0, 0.0, 0.0]] * 21
        right_hand_points = [[0.0, 0.0, 0.0]] * 21

        frame    = cv2.resize(frame, (320, 240))
        frame    = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results  = self.mp.detect_for_video(mp_image, int(timestamp_ms))

        for idx, hand in enumerate(results.hand_landmarks):
            hand_type = results.handedness[idx][0].category_name
            points    = [[lm.x, lm.y, lm.z] for lm in hand]
            if hand_type == "Left":
                left_hand_points  = points
            else:
                right_hand_points = points

        return left_hand_points + right_hand_points

    def _chamfer_distance(self, prev_point_cloud, pres_point_cloud):
        if not prev_point_cloud or not pres_point_cloud:
            return 0.0

        pc1 = np.array(prev_point_cloud)
        pc2 = np.array(pres_point_cloud)
        d1  = cdist(pc1, pc2).min(axis=1).mean()
        d2  = cdist(pc2, pc1).min(axis=1).mean()
        return d1 + d2

    def _select_keyframes(self, motion):
        """
        Select keyframes and always return exactly KEYFRAME_K indices so
        every feature vector has the same length.
        """
        k = KEYFRAME_K
        n = len(motion)

        keyframes = {0, n - 1}

        for t in range(1, n - 1):
            if motion[t] > motion[t - 1] and motion[t] > motion[t + 1]:
                keyframes.add(t)

        top_k = np.argsort(motion)[-k:]
        keyframes.update(top_k)
        keyframes = sorted(keyframes)

        while len(keyframes) < k:
            keyframes.append(keyframes[-1])

        return keyframes[:k]

    def _video_to_features(self, video_path):
        self.mp = self._create_mp_task_video()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Could not open: {video_path}")
            return None

        fps      = cap.get(cv2.CAP_PROP_FPS) or 30
        step_ms  = int(1000 / fps)
        frame_ts = 0

        motion        = []
        point_clouds  = []
        hand_detected = []
        prev_pc       = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pc = self._extract_landmarker_point_cloud(frame, frame_ts)

            if pc is not None:
                flat     = np.array(pc)
                has_hand = not np.all(flat == 0.0)
                hand_detected.append(has_hand)
                point_clouds.append(pc)

                m = self._chamfer_distance(prev_pc, pc) if prev_pc is not None else 0.0
                motion.append(m)
                prev_pc = pc

            frame_ts += step_ms

        cap.release()

        if any(hand_detected):
            first = next(i for i, v in enumerate(hand_detected) if v)
            last  = len(hand_detected) - 1 - next(
                i for i, v in enumerate(reversed(hand_detected)) if v
            )
            point_clouds = point_clouds[first:last + 1]
            motion       = motion[first:last + 1]
            print(f"[INFO] Hand present in frames {first}–{last} "
                  f"({last - first + 1} frames kept out of {len(hand_detected)})")
        else:
            print(f"[WARN] No hand detected in any frame: {video_path}")
            return None

        if len(point_clouds) < KEYFRAME_K:
            print(f"[WARN] Not enough hand frames in: {video_path} "
                  f"(got {len(point_clouds)}, need {KEYFRAME_K})")
            return None

        keyframes = self._select_keyframes(motion)
        features  = _extract_svm_features(keyframes, point_clouds)

        return features

    def _build_Xy(self):
        train_features, train_labels = [], []
        test_features,  test_labels  = [], []

        for row in tqdm(self.raw_train_data.itertuples(),
                        total=len(self.raw_train_data), desc="Train"):
            feat = self._video_to_features(row.vid_path)
            if feat is not None:
                train_features.append(feat)
                train_labels.append(row.label)

        for row in tqdm(self.raw_test_data.itertuples(),
                        total=len(self.raw_test_data), desc="Test"):
            feat = self._video_to_features(row.vid_path)
            if feat is not None:
                test_features.append(feat)
                test_labels.append(row.label)

        self.X_train = np.array(train_features)
        self.y_train = np.array(train_labels)
        self.X_test  = np.array(test_features)
        self.y_test  = np.array(test_labels)

        print("Train shape:", self.X_train.shape)
        print("Test shape: ", self.X_test.shape)

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test  = self.scaler.transform(self.X_test)

    def train_svm_model(self):
        if len(self.X_train) == 0:
            self._build_Xy()

        param_grid = {
            "C":      [0.1, 1, 10, 100],
            "gamma":  ["scale", 0.01, 0.001, 0.0001],
            "kernel": ["rbf"]
        }

        cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            estimator  = SVC(probability=True),
            param_grid = param_grid,
            cv         = cv,
            scoring    = "accuracy",
            n_jobs     = -1,
            verbose    = 1
        )
        grid.fit(self.X_train, self.y_train)

        self.svm = grid.best_estimator_
        print("SVM trained successfully.")
        print("Best parameters:", grid.best_params_)
        print("Best CV accuracy:", grid.best_score_)

    def evaluate_svm_model(self):
        if self.svm is None:
            return

        y_pred = self.svm.predict(self.X_test)
        print(classification_report(self.y_test, y_pred, zero_division=0))

        cm = confusion_matrix(self.y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=self.svm.classes_).plot(
            xticks_rotation="vertical"
        )
        plt.title("FSL-SVM (Normalised Point Cloud + Velocity)")
        plt.tight_layout()
        plt.show()

        print("Train Accuracy:", self.svm.score(self.X_train, self.y_train))
        print("Test Accuracy: ", self.svm.score(self.X_test,  self.y_test))

    def run_svm_model(self, video_path):
        feat = self._video_to_features(video_path)
        if feat is None:
            print("[WARN] Could not extract features from video.")
            return

        feat_scaled   = self.scaler.transform([feat])
        probabilities = self.svm.predict_proba(feat_scaled)[0]
        max_idx       = np.argmax(probabilities)
        confidence    = probabilities[max_idx]
        pred_label    = self.svm.classes_[max_idx]

        print("CONFIDENCE:", float(confidence))
        print("LABEL:     ", pred_label)

    def save_svm_model(self, path="./models/fsl-svm-2-catv2.pkl"):
        joblib.dump({"model": self.svm, "scaler": self.scaler}, path)

    def load_svm_model(self, path="./models/fsl-svm-2-catv2.pkl"):
        data        = joblib.load(path)
        self.svm    = data["model"]
        self.scaler = data["scaler"]
