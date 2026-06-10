import cv2
import mediapipe as mp
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.mp = self._init()
        self.global_ts = 0  # same fix as build_sequences.py

    def _init(self):
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

    def extract(self, frame):
        self.global_ts += 1
        ts = self.global_ts * 33

        frame = cv2.resize(frame, (320, 240))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mp.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int(ts)
        )

        left = [[0, 0, 0]] * 21
        right = [[0, 0, 0]] * 21
        if result.hand_landmarks:
            for i, hand in enumerate(result.hand_landmarks):
                pts = [[lm.x, lm.y, lm.z] for lm in hand]
                if result.handedness[i][0].category_name == "Left":
                    left = pts
                else:
                    right = pts
        return np.array(left + right)
