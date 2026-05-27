import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------
# DATA LOAD
# --------------------
df = pd.read_csv(
    "../dataset/cache_dataset.csv",
    header=None,
    names=["latency_ns", "buffer_size", "label"]
)

df["ratio"] = df["latency_ns"] / df["buffer_size"]

X = df[["latency_ns", "buffer_size", "ratio"]].values
y_true = df["label"].values

X = np.log1p(X)
X = torch.tensor(X, dtype=torch.float32)

# --------------------
# MODEL (same structure)
# --------------------
class CacheClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)

model = CacheClassifier()
model.load_state_dict(torch.load("cache_model.pth"))

# --------------------
# PREDICT
# --------------------
with torch.no_grad():
    y_pred = torch.argmax(model(X), dim=1).numpy()

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))