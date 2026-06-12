import torch
import json
from lstm.model import LSTMEncoder

class Encoder:
    def __init__(self, path="/home/code871/Git/fsl-svm/server3/lstm/encoder.pth"):
        self.model = LSTMEncoder()
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()

    def encode(self, seq):
        x = seq.reshape(1, seq.shape[0], -1)
        x = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            emb = self.model(x)

        return emb.numpy()[0]
