from agents.benchmark_agent import BenchmarkAgent
from agents.train_agent import TrainAgent
from agents.evaluate_agent import EvaluateAgent

def run_pipeline():
    """
    Orchestrates the DeepCacheScope agentic pipeline.
    Chains BenchmarkAgent, TrainAgent, and EvaluateAgent sequentially.
    """
    print("="*50)
    print(" DeepCacheScope - Antigravity Agent Pipeline")
    print("="*50)

    # 1. Benchmark Agent
    benchmark_agent = BenchmarkAgent()
    if not benchmark_agent.run():
        print("Pipeline aborted at Benchmark phase.")
        return

    # 2. Train Agent
    train_agent = TrainAgent()
    if not train_agent.run():
        print("Pipeline aborted at Training phase.")
        return

    # 3. Evaluate Agent
    evaluate_agent = EvaluateAgent()
    if not evaluate_agent.run():
        print("Pipeline aborted at Evaluation phase.")
        return

    print("\n" + "="*50)
    print(" Pipeline completed successfully!")
    print("="*50)

if __name__ == "__main__":
    run_pipeline()
