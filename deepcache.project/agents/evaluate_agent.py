from python.evaluate import run_evaluation

class EvaluateAgent:
    """
    Antigravity EvaluateAgent.
    Responsible for loading the trained model, running evaluation, and printing the accuracy + confusion matrix.
    """
    def __init__(self, dataset_path="dataset/cache_dataset.csv", model_load_path="python/cache_model.pth"):
        self.dataset_path = dataset_path
        self.model_load_path = model_load_path

    def run(self):
        print("\n[EvaluateAgent] Starting evaluation process...")
        success = run_evaluation(self.dataset_path, self.model_load_path)
        if success:
            print("[EvaluateAgent] Evaluation completed successfully.")
            return True
        else:
            print("[EvaluateAgent] Evaluation failed.")
            return False
