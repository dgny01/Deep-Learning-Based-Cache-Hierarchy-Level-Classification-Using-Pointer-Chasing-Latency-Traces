import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from python.model import CacheClassifier

def run_evaluation(dataset_path="dataset/cache_dataset.csv", model_load_path="python/cache_model.pth"):
    """
    Loads dataset and model, evaluates the model, and prints accuracy and confusion matrix.
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

    df["ratio"] = df["latency_ns"] / df["buffer_size"]
    X = df[["latency_ns", "buffer_size", "ratio"]].values
    y_true = df["label"].values

    X = np.log1p(X)
    X = torch.tensor(X, dtype=torch.float32)

    print(f"Loading model from {model_load_path}...")
    model = CacheClassifier()
    try:
        model.load_state_dict(torch.load(model_load_path))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model weights not found at {model_load_path}. Run training first.")
        return False

    print("Running evaluation...")
    with torch.no_grad():
        y_pred = torch.argmax(model(X), dim=1).numpy()

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("--------------------------\n")
    return True

if __name__ == "__main__":
    run_evaluation()