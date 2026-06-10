import numpy as np
import torch
import torch.nn as nn
from model import LSTMEncoder

X = np.load("/home/code871/Git/fsl-svm/server3/data/sequences.npy")
y = np.load("/home/code871/Git/fsl-svm/server3/data/labels.npy")

X = X.reshape(X.shape[0], X.shape[1], -1)

# label encoding
classes = sorted(list(set(y)))
label_map = {c: i for i, c in enumerate(classes)}
y = np.array([label_map[i] for i in y])

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

model = LSTMEncoder(input_size=126)
classifier = nn.Linear(128, len(classes))

opt = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(25):
    emb = model(X)
    logits = classifier(emb)

    loss = loss_fn(logits, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"Epoch {epoch} Loss {loss.item()}")

torch.save(model.state_dict(), "lstm/encoder.pth")

import json
with open("/home/code871/Git/fsl-svm/server3/data/label_map.json", "w") as f:
    json.dump(label_map, f)
