import subprocess
import os

class BenchmarkAgent:
    """
    Antigravity BenchmarkAgent.
    Responsible for compiling and running the C++ benchmark to generate the latency dataset.
    """
    def __init__(self, cpp_dir="cpp", dataset_dir="dataset"):
        self.cpp_dir = cpp_dir
        self.dataset_dir = dataset_dir
        self.executable = os.path.join(self.cpp_dir, "cachescope")
        self.output_csv = os.path.join(self.dataset_dir, "cache_dataset.csv")

    def run(self):
        print("[BenchmarkAgent] Starting benchmark process...")
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            
        # Compile if executable doesn't exist
        if not os.path.exists(self.executable):
            print(f"[BenchmarkAgent] Compiling C++ benchmark...")
            compile_cmd = [
                "g++", "-O3",
                os.path.join(self.cpp_dir, "main.cpp"),
                os.path.join(self.cpp_dir, "cache_scope.cpp"),
                os.path.join(self.cpp_dir, "exporter.cpp"),
                "-o", self.executable
            ]
            try:
                subprocess.run(compile_cmd, check=True)
                print("[BenchmarkAgent] Compilation successful.")
            except subprocess.CalledProcessError as e:
                print(f"[BenchmarkAgent] Compilation failed: {e}")
                return False

        # Run benchmark
        print(f"[BenchmarkAgent] Running benchmark to generate {self.output_csv}...")
        try:
            # Note: Assuming the C++ executable writes to dataset/cache_dataset.csv relative to cwd.
            # But wait, looking at exporter.cpp, where does it write?
            # Let's run it from root.
            subprocess.run([self.executable], check=True)
            print("[BenchmarkAgent] Benchmark complete.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[BenchmarkAgent] Benchmark execution failed: {e}")
            return False
