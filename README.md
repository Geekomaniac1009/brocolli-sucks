# Repository
The code is provided **solely for the purpose of peer review** and is intended to support **full reproducibility of the experimental results** reported in the submitted manuscript.
All author-identifying information has been deliberately removed to preserve **double-blind review**.

---

## 1. Overview

This repository implements the experimental framework, benchmarks, and evaluation data used in the paper.  
The focus is on evaluating algorithmic behavior under varying **task demand**, **power availability**, **battery configurations**, **grid costs**, and **service-level constraints**.

The repository is intentionally kept **flat (no subdirectories)** to simplify reviewer access.

---

## 2. Repository Structure (Flat Layout)

The repository consists of the following categories of files:

### 2.1 Executable / Core Experiment
- `exp3`  
  Compiled executable used to run the main experiments reported in the paper.

---

### 2.2 Benchmark Files (`benchmark_*`)

Files prefixed with `benchmark_` correspond to **benchmark scenarios** used for comparative evaluation.

These include benchmarks across:
- Battery capacity levels  
  - `benchmark_battery1` … `benchmark_battery5`
- Efficiency settings  
  - `benchmark_efficiency1` … `benchmark_efficiency5`
- SLA constraints  
  - `benchmark_sla1` … `benchmark_sla5`

Each benchmark file encodes a specific experimental configuration described in the paper.

---

### 2.3 Grid Cost Model
- `grid_cost`  

Defines the **grid pricing / cost model** used in the experiments, including cost variations across usage levels.

---

### 2.4 Real-World Traces
These files correspond to **realistic task demand and power availability traces**:

- `task_reallife`  
- `task_reallife_1440`  
- `power_reallife`  

They represent real or realistically generated traces used to validate performance beyond synthetic benchmarks.

---

### 2.5 Task and Power Variations

The remaining files represent **controlled variations** of task and power profiles used for sensitivity analysis:

#### Task variations
- `tasks_5_deviation`
- `tasks_10_deviation`
- `tasks_20_deviation`
- `results_10_robust_tasks`

#### Power variations
- `results_5_power_deviation`
- `results_10_power_deviation`
- `results_20_power_deviation`
- `results_10_robust_power`

These files are used to evaluate robustness under increasing uncertainty and deviation levels.

---

### 2.6 Logs and Detailed Outputs
- `detailed_log_greedy`
- `detailed_log_optimal`
- `detailed_log_sdc`
- `grid_usage_log_1` … `grid_usage_log_5`

These logs capture fine-grained execution details for different algorithms and configurations.

---

## 3. System Requirements

- Linux / macOS / Windows (WSL2)
- C++17-compatible compiler (`g++ ≥ 9.0`)

Optional (for analysis):
- Python ≥ 3.8
- NumPy, Pandas, Matplotlib

---

## 4. Running the Experiments

Ensure the executable has run permissions:

```bash
chmod +x exp3
```

Run an experiment using a benchmark file:

```bash
./exp3 benchmark_battery3
```

Run with real-world traces:

```bash
./exp3 task_reallife power_reallife
```

(Exact invocation parameters are consistent with those described in the paper.)

---

## 5. Mapping to the Paper

| Paper Component | Files |
|----------------|------|
| Benchmark evaluation | `benchmark_*` |
| Grid pricing model | `grid_cost` |
| Real-world validation | `task_reallife*`, `power_reallife` |
| Robustness analysis | `tasks_*_deviation`, `results_*_power_deviation` |
| Algorithm comparison | `detailed_log_*` |

---

## 6. Reproducibility Notes

- All experiments are deterministic unless explicitly stated
- Default parameters correspond to those used in the paper
- Output logs are generated in-place

---

## 7. Anonymity Statement

To preserve double-blind review:
- No author names or affiliations appear in this repository
- Commit metadata is anonymized
- File naming is intentionally generic

Please do not attempt to infer authorship from this repository.

---

## 8. License

This repository is released **only for anonymous peer review**.

Upon acceptance, the repository will be updated with:
- Author information
- Documentation
- Licensing details
- Citation instructions
