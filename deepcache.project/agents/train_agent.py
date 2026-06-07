from python.train import run_training

class TrainAgent:
    """
    Antigravity TrainAgent.
    Responsible for loading the generated CSV dataset, training the PyTorch model, and saving the weights.
    """
    def __init__(self, dataset_path="dataset/cache_dataset.csv", model_save_path="python/cache_model.pth"):
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path

    def run(self):
        print("\n[TrainAgent] Starting training process...")
        success = run_training(self.dataset_path, self.model_save_path)
        if success:
            print("[TrainAgent] Training completed successfully.")
            return True
        else:
            print("[TrainAgent] Training failed.")
            return False
