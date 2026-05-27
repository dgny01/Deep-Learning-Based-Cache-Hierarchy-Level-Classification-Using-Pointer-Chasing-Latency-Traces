import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --------------------
# DATA LOAD
# --------------------
df = pd.read_csv(
    "../dataset/cache_dataset.csv",
    header=None,
    names=["latency_ns", "buffer_size", "label"]
)

# --------------------
# FEATURE ENGINEERING
# --------------------
df["ratio"] = df["latency_ns"] / df["buffer_size"]

X = df[["latency_ns", "buffer_size", "ratio"]].values
y = df["label"].values

# log transform (çok önemli)
X = np.log1p(X)

# tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------
# MODEL
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

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# --------------------
# TRAIN
# --------------------
for epoch in range(100):
    logits = model(X_train)
    loss = loss_fn(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("epoch", epoch, "loss", loss.item())

# save
torch.save(model.state_dict(), "cache_model.pth")
print("Model saved")