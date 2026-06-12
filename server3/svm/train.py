import numpy as np
import joblib
from sklearn.svm import SVC
from lstm.encoder import Encoder

X = np.load("/home/code871/Git/fsl-svm/server3/data/sequences.npy")
y = np.load("/home/code871/Git/fsl-svm/server3/data/labels.npy")


encoder = Encoder("/home/code871/Git/fsl-svm/server3/lstm/encoder.pth")

embeddings = []

for seq in X:
    embeddings.append(encoder.encode(seq))

embeddings = np.array(embeddings)

svm = SVC(kernel="rbf", probability=True)
svm.fit(embeddings, y)

joblib.dump(svm, "svm/model.pkl")

print("SVM trained on LSTM embeddings")
