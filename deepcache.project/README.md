# DeepCacheScope

DeepCacheScope is a high-performance cache hierarchy level classifier designed to detect and categorize memory access latency. It combines a low-level C++ benchmarking engine that performs pointer-chasing latency measurements with a PyTorch-based Deep Neural Network (DNN) to classify memory accesses into **L1 Cache**, **L2 Cache**, **L3 Cache**, or **RAM (System Memory)**.

The entire workflow is automated using an **Agentic AI Architecture** where each step (benchmarking, training, and evaluation) is encapsulated as a specialized agent orchestrated by a central pipeline.

---

## 📂 Project Structure

```text
DeepCacheScope/
├── cpp/
│   ├── cache_scope.cpp      # Latency measurement implementation
│   ├── cache_scope.hpp      # Latency measurement header
│   ├── main.cpp             # Benchmark runner driving memory allocations
│   └── exporter.cpp         # Exports benchmark results to CSV
├── python/
│   ├── model.py             # PyTorch Neural Network architecture definition
│   ├── train.py             # Model training routines
│   └── evaluate.py          # Model evaluation and metrics generation
├── dataset/                 # Generated datasets (Git ignored)
├── agents/
│   ├── benchmark_agent.py   # Compiles & runs C++ benchmark to generate CSV
│   ├── train_agent.py       # Loads CSV, trains PyTorch model, and saves .pth
│   └── evaluate_agent.py    # Loads model, runs evaluation, and prints metrics
├── pipeline.py              # Main orchestrator chaining all agents sequentially
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignored files & directories
└── README.md                # Project documentation
```

---

## 🚀 How to Set Up and Run

Follow these step-by-step instructions to set up a Python virtual environment, install dependencies, compile the benchmark, and run the pipeline.

### 1. Clone the Project
First, navigate to your project directory:
```bash
cd /home/doganay/Masaüstü/deepcache.project
```

### 2. Set Up a Virtual Environment (Recommended)
Creating an isolated virtual environment prevents conflicts with system packages.

**On Linux/macOS:**
```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

*(You will know it is active when `(venv)` appears at the beginning of your terminal prompt).*

### 3. Install Python Dependencies
Install all required libraries (PyTorch, Pandas, Scikit-learn, etc.) listed in `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Pipeline
With the virtual environment active and dependencies installed, you can run the entire workflow end-to-end with a single command:
```bash
python pipeline.py
```

---

## 🤖 Agent Workflow Details

When you execute `pipeline.py`, the system coordinates the following agents:

### 1. ⚙️ Benchmark Agent (`BenchmarkAgent`)
- Checks for the compiled C++ executable (`cpp/cachescope`).
- If not compiled, it invokes `g++ -O3` to compile the C++ benchmarking source code.
- Runs the benchmark which performs pointer-chasing latency measurements across various buffer sizes mapping to L1, L2, L3, and RAM.
- Outputs the raw latency data to `dataset/cache_dataset.csv`.

### 2. 🧠 Train Agent (`TrainAgent`)
- Loads the generated `dataset/cache_dataset.csv`.
- Performs feature engineering (logarithmic transform and latency/buffer-size ratios).
- Trains the `CacheClassifier` PyTorch model for 100 epochs.
- Saves the trained model weights to `python/cache_model.pth`.

### 3. 📊 Evaluate Agent (`EvaluateAgent`)
- Loads the trained model weights (`python/cache_model.pth`).
- Evaluates model performance on the test subset.
- Prints the classification **Accuracy** and computes a **Confusion Matrix** to inspect classification distribution across hierarchy levels.

---

## 🛠️ Requirements & Prerequisites
- **Compiler**: `g++` supporting C++11 or newer.
- **Python**: Version `3.8` or newer.
- **Hardware**: Runs on both CPU and CUDA-enabled GPU (automatically defaults to CPU for small-scale tabular classification).