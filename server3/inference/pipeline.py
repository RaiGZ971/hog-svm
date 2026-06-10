import numpy as np
import joblib
from lstm.encoder import Encoder
from inference.feature_extractor import FeatureExtractor
from preprocessing.dataset_utils import normalize_sequence, pad_or_trim

class Pipeline:
    def __init__(self):
        self.fe = FeatureExtractor()
        self.encoder = Encoder("lstm/encoder.pth")
        self.svm = joblib.load("svm/model.pkl")
        self.buffer = []

    def predict(self, frame):
        lm = self.fe.extract(frame)
        self.buffer.append(lm)

        if len(self.buffer) > 60:
            self.buffer.pop(0)

        if len(self.buffer) < 60:
            return None

        seq = np.array(self.buffer)
        seq = normalize_sequence(seq)
        seq = pad_or_trim(seq, T=60)
        emb = self.encoder.encode(seq)
        probs = self.svm.predict_proba([emb])[0]
        idx = np.argmax(probs)
        pred = self.svm.classes_[idx]

        # 🔥 DEBUG
        print(f"[PRED] {pred} | confidence: {probs[idx]:.3f} | all: {dict(zip(self.svm.classes_, probs.round(3)))}")

        return pred
