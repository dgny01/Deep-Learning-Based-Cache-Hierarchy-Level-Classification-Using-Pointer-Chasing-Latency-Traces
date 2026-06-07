import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from python.model import CacheClassifier

def run_training(dataset_path="dataset/cache_dataset.csv", model_save_path="python/cache_model.pth"):
    """
    Loads dataset, trains the PyTorch model, and saves the trained weights.
    """
    print(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(
            dataset_path,
            header=None,
            names=["latency_ns", "buffer_size", "label"]
        )
    except FileNotFoundError:
        print(f"Error: {dataset_path} not found. Run benchmark first.")
        return False

    # Feature Engineering
    df["ratio"] = df["latency_ns"] / df["buffer_size"]
    X = df[["latency_ns", "buffer_size", "ratio"]].values
    y = df["label"].values

    # Log transform
    X = np.log1p(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Initializing model...")
    model = CacheClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    print("Starting training...")
    for epoch in range(100):
        logits = model(X_train)
        loss = loss_fn(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")
    return True

if __name__ == "__main__":
    run_training()