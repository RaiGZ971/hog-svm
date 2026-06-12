import numpy as np
import pandas as pd
from tqdm import tqdm

import mediapipe as mp
import cv2
from scipy.spatial.distance import cdist
import joblib

from svm.stratified_kfold import StratifiedKFold
from svm.grid_search_cv import GridSearchCV
from svm.standard_scaler import StandardScaler

from evaluation.metrics import compute_accuracy, compute_class_metrics
from evaluation.confusion import build_confusion_matrix, plot_confusion_matrix
from evaluation.report import print_classification_report

import os
from dotenv import load_dotenv

load_dotenv()

KEYFRAME_K = int(os.getenv("HYPERPARAMETER_K", 30))

TRAINING_FPS = int(os.getenv("TRAINING_FPS", 30))

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


def _select_keyframes(motion: list) -> list:
    """
    Select exactly KEYFRAME_K frame indices from a motion signal.

    Priority order:
      1. Boundary frames {0, n-1}
      2. Local motion peaks
      3. Top-k highest motion frames

    If the union of the above exceeds k, the k candidates with the highest
    motion values are kept (then restored to temporal order).
    If fewer than k candidates exist, the last index is repeated to pad.

    Always returns exactly KEYFRAME_K indices so every feature vector has
    the same length: (2*KEYFRAME_K - 1) * 126.
    """
    k          = KEYFRAME_K
    n          = len(motion)
    motion_arr = np.array(motion)

    candidates = {0, n - 1}

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
        candidates = sorted(candidates)   # restore temporal order

    while len(candidates) < k:
        candidates.append(candidates[-1])

    return candidates


def _extract_svm_features(keyframes: list, point_clouds: list) -> np.ndarray:
    """
    Build a fixed-length feature vector from the selected keyframes.

    Parameters
    ----------
    keyframes    : list[int], length K — indices into point_clouds
    point_clouds : list of raw 42-point clouds (one per sampled frame)

    Returns
    -------
    np.ndarray, shape ((2K-1) * 126,)  where K = KEYFRAME_K
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
        timestamp_ms must be strictly increasing within the current video.
        """
        left_hand_points  = [[0.0, 0.0, 0.0]] * 21
        right_hand_points = [[0.0, 0.0, 0.0]] * 21

        frame    = cv2.resize(frame, (320, 240))
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

    def _video_to_features(self, video_path):
        """
        Extract a fixed-length SVM feature vector from a single video.

        The video is downsampled to TRAINING_FPS (default 30) before landmark
        extraction so that the temporal density matches the ~30 fps stream
        received from the browser at inference time.
        """
        self.mp = self._create_mp_task_video()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Could not open: {video_path}")
            return None

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_fps = float(TRAINING_FPS)

        frame_step = max(1, round(native_fps / target_fps))
        step_ms = 1000.0 / target_fps

        print(f"[INFO] {video_path}: native={native_fps:.1f}fps "
              f"target={target_fps:.0f}fps step={frame_step}")

        motion        = []
        point_clouds  = []
        hand_detected = []
        prev_pc       = None
        frame_idx     = 0   # native frame counter
        sampled_idx   = 0   # sampled frame counter (drives timestamp)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every frame_step-th native frame
            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            timestamp_ms = sampled_idx * step_ms
            sampled_idx += 1
            frame_idx   += 1

            pc = self._extract_landmarker_point_cloud(frame, timestamp_ms)

            if pc is not None:
                flat     = np.array(pc)
                has_hand = not np.all(flat == 0.0)
                hand_detected.append(has_hand)
                point_clouds.append(pc)

                m = self._chamfer_distance(prev_pc, pc) if prev_pc is not None else 0.0
                motion.append(m)
                prev_pc = pc

        cap.release()

        if any(hand_detected):
            first = next(i for i, v in enumerate(hand_detected) if v)
            last  = len(hand_detected) - 1 - next(
                i for i, v in enumerate(reversed(hand_detected)) if v
            )
            point_clouds = point_clouds[first:last + 1]
            motion       = motion[first:last + 1]
            print(f"[INFO] Hand present in frames {first}–{last} "
                  f"({last - first + 1} sampled frames kept)")
        else:
            print(f"[WARN] No hand detected in any frame: {video_path}")
            return None

        if len(point_clouds) < KEYFRAME_K:
            print(f"[WARN] Not enough hand frames in: {video_path} "
                  f"(got {len(point_clouds)}, need {KEYFRAME_K})")
            return None

        keyframes = _select_keyframes(motion)
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

    def train_svm_model(self):
        if len(self.X_train) == 0:
            self._build_Xy()
            self.store_Xy()

        param_grid = {
            "C":      [0.1, 1, 10, 100],
            "gamma":  ["scale", 0.01, 0.001, 0.0001],
            "max_passes": [10]
        }

        cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(param_grid=param_grid, cv=cv, verbose=1)
        grid.fit(self.X_train, self.y_train)

        self.svm    = grid.best_estimator_
        self.scaler = grid._best_scaler

        print("SVM trained successfully.")
        print("Best parameters:", grid.best_params_)
        print("Best CV accuracy:", grid.best_score_)

    def evaluate_svm_model(self):
        """
        Evaluate the trained SVM on the test set (and report train accuracy).

        Prints:
            - Classification report  (precision / recall / F1 / support per class)
            - Train and test accuracy
            - Plot confusion matrix
        """
        if self.svm is None:
            print("No trained SVM found. Skipping evaluation.")
            return

        X_trained_scaled = self.scaler.transform(self.X_train)
        X_test_scaled    = self.scaler.transform(self.X_test)

        y_pred       = self.svm.predict(X_test_scaled)
        y_train_pred = self.svm.predict(X_trained_scaled)

        y_true_list = list(self.y_test)
        y_pred_list = list(y_pred)

        print_classification_report(y_true_list, y_pred_list)

        train_acc = compute_accuracy(list(self.y_train), list(y_train_pred))
        test_acc  = compute_accuracy(y_true_list, y_pred_list)

        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test  Accuracy: {test_acc:.4f}\n")

        classes, _ = compute_class_metrics(y_true_list, y_pred_list)
        cm = build_confusion_matrix(y_true_list, y_pred_list, classes)
        plot_confusion_matrix(cm, classes)

    def run_svm_model(self, video_path):
        feat = self._video_to_features(video_path)
        if feat is None:
            print("[WARN] Could not extract features from video.")
            return

        feat_scaled = self.scaler.transform(feat.reshape(1, -1))
        
        pred_label  = self.svm.predict(feat_scaled)[0]
        
        probabilities = self.svm.predict_proba(feat_scaled)[0]
        pred_idx      = np.where(self.svm.classes_ == pred_label)[0][0]
        confidence    = probabilities[pred_idx]

        print("CONFIDENCE:", float(confidence))
        print("LABEL:     ", pred_label)

    def save_svm_model(self, path="./models/fsl-svm-2-catv6.pkl"):
        joblib.dump({"model": self.svm, "scaler": self.scaler}, path)

    def load_svm_model(
        self,
        path="./models/fsl-svm-2-catv6.pkl",
        train_features=None,
        train_labels=None,
        test_features=None,
        test_labels=None
    ):
        data = joblib.load(path)

        self.svm    = data["model"]
        self.scaler = data["scaler"]

        if train_features:
            self.X_train = joblib.load(train_features)
        if train_labels:
            self.y_train = joblib.load(train_labels)
        if test_features:
            self.X_test = joblib.load(test_features)
        if test_labels:
            self.y_test = joblib.load(test_labels)

    def store_Xy(self):
        joblib.dump(self.X_train, "./data/train_features_v3.npy")
        joblib.dump(self.y_train, "./data/train_labels_v3.npy")
        joblib.dump(self.X_test,  "./data/test_features_v3.npy")
        joblib.dump(self.y_test,  "./data/test_labels_v3.npy")

    def append_dataset(
        self,
        new_train_csv=None,
        new_test_csv=None,
        train_features_path="./data/train_features_v2.npy",
        train_labels_path="./data/train_labels_v2.npy",
        test_features_path="./data/test_features_v2.npy",
        test_labels_path="./data/test_labels_v2.npy",
    ):
        # Load old features and labels
        X_train_old = joblib.load(train_features_path)
        y_train_old = joblib.load(train_labels_path)

        X_test_old = joblib.load(test_features_path)
        y_test_old = joblib.load(test_labels_path)


        # Extract features and labels from new CSVs
        new_train_features = []
        new_train_labels = []

        if new_train_csv is not None:
            train_df = pd.read_csv(new_train_csv)

            for row in tqdm(
                train_df.itertuples(),
                total=len(train_df),
                desc="New Train"
            ):
                feat = self._video_to_features(row.vid_path)

                if feat is not None:
                    new_train_features.append(feat)
                    new_train_labels.append(row.label)

        new_test_features = []
        new_test_labels = []

        if new_test_csv is not None:
            test_df = pd.read_csv(new_test_csv)

            for row in tqdm(
                test_df.itertuples(),
                total=len(test_df),
                desc="New Test"
            ):
                feat = self._video_to_features(row.vid_path)

                if feat is not None:
                    new_test_features.append(feat)
                    new_test_labels.append(row.label)

        # Convert to numpy arrays
        new_train_features = np.array(new_train_features)
        new_train_labels = np.array(new_train_labels)

        new_test_features = np.array(new_test_features)
        new_test_labels = np.array(new_test_labels)

        # Merge old and new data
        if len(new_train_features) > 0:
            self.X_train = np.concatenate(
                [X_train_old, new_train_features],
                axis=0
            )
            self.y_train = np.concatenate(
                [y_train_old, new_train_labels],
                axis=0
            )
        else:
            self.X_train = X_train_old
            self.y_train = y_train_old

        if len(new_test_features) > 0:
            self.X_test = np.concatenate(
                [X_test_old, new_test_features],
                axis=0
            )
            self.y_test = np.concatenate(
                [y_test_old, new_test_labels],
                axis=0
            )
        else:
            self.X_test = X_test_old
            self.y_test = y_test_old

        print("Merged train shape:", self.X_train.shape)
        print("Merged test shape :", self.X_test.shape)

        self.store_Xy()
        