import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from dataset_utils import pad_or_trim, normalize_sequence

class SequenceBuilder:
    def __init__(self):
        self.mp = self._init_mp()
        self.global_ts = 0

    def _init_mp(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path="ml/hand_landmarker.task"
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2
        )
        return HandLandmarker.create_from_options(options)

    def extract(self, path):
        cap = cv2.VideoCapture(path)
        seq = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.global_ts += 1
            ts = self.global_ts * 33
            frame = cv2.resize(frame, (320, 240))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
            result = self.mp.detect_for_video(mp_image, int(ts))
            left = [[0, 0, 0]] * 21
            right = [[0, 0, 0]] * 21
            if result and result.hand_landmarks:
                for i, hand in enumerate(result.hand_landmarks):
                    pts = [[lm.x, lm.y, lm.z] for lm in hand]
                    if result.handedness[i][0].category_name == "Left":
                        left = pts
                    else:
                        right = pts
            seq.append(left + right)
        cap.release()
        return np.array(seq)


def build_dataset(csv_path, out_dir, T=60):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    builder = SequenceBuilder()
    X = []
    y = []
    print(f"Processing {len(df)} videos...")
    for i, row in df.iterrows():
        path = row["vid_path"]
        label = row["label"]
        print(f"[{i+1}/{len(df)}] {label}")
        seq = builder.extract(path)
        seq = normalize_sequence(seq)
        seq = pad_or_trim(seq, T=T)
        X.append(seq)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    np.save(os.path.join(out_dir, "sequences.npy"), X)
    np.save(os.path.join(out_dir, "labels.npy"), y)
    print("\n✅ DONE")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Saved to:", out_dir)


if __name__ == "__main__":
    build_dataset(
        "./csvs/train_filtered.csv",
        "./data",
        T=60
    )
