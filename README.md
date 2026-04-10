# Hybrid Quantum Machine Learning — Wildfire Risk Prediction

**Deloitte Quantum Sustainability Challenge 2026 | Task 1A & 1B**

---

## Overview

This solution addresses **Tasks 1A and 1B** of the [Deloitte Quantum Sustainability Challenge 2026](https://us.ekipa.de/deloitte-quantum-2026), which asks participants to apply quantum machine learning to predict future wildfire risk at the California ZIP-code level.

- **Task 1A:** Build a hybrid QML model that predicts which ZIP codes will experience a wildfire in 2023, trained on historical data (2018–2022). Run it on a quantum simulator and report resource requirements.
- **Task 1B:** Evaluate the model against classical baselines, discussing the advantages and limitations of the quantum approach.

A wildfire is defined as an **unplanned, uncontrolled wildland fire** (`OBJECTIVE == 1` with `FIRE_NAME` present), in line with the competition specification.

---

## Architecture

```
Raw CSV (wildfire_weather.csv)
  └── Temporal feature engineering (rolling 3-year windows, no data leakage)
        └── PCA → MinMaxScaler → quantum angle encoding
              └── Entangled quantum circuit (RY · RZ + CZ ring/cross topology)
                    └── 8 Pauli observables (4 single-qubit Z + 4 two-qubit ZZ)
                          └── Hybrid features (PCA angles + quantum expectation values)
                                ├── LogisticRegression (primary classifier)
                                ├── QSVC — ZZFeatureMap kernel (quantum subset)
                                ├── VQC — RealAmplitudes ansatz (quantum subset)
                                └── RandomForest, GradientBoosting, SVM (classical baselines)
                                      └── Ensemble risk score + PR-curve threshold tuning
                                            └── Per-ZIP risk categories + resource estimation
```

---

## Dataset

This solution uses **Wildfire Dataset 2** from the competition:

| File | Source |
|------|--------|
| `abfap7bci2UF6CTY_wildfire_weather.csv` | [Download from competition page](https://us.ekipa.de/deloitte-quantum-2026) |

Place the downloaded CSV at the path specified by `--dataset` (see [Usage](#usage)).

---

## Features

Ten features are engineered per `(zip, target_year)` row, all strictly lagged to prevent data leakage:

| Feature | Description |
|---------|-------------|
| `fire_count_prev1` | Wildfire count in the immediately preceding year |
| `fire_count_prev3` | Wildfire count summed over the prior 3 years |
| `acres_prev1_log` | Log-transformed burned acres in the preceding year |
| `acres_prev3_log` | Log-transformed burned acres over the prior 3 years |
| `avg_tmax_prev_window` | Mean daily max temperature over the weather window |
| `avg_tmin_prev_window` | Mean daily min temperature over the weather window |
| `total_prcp_prev_window` | Average annual precipitation over the weather window |
| `prcp_std_prev_window` | Standard deviation of precipitation (rainfall variability) |
| `heat_stress_prev` | Derived: `avg_tmax − avg_tmin` |
| `hot_dry_idx_prev` | Derived: `avg_tmax − 0.1 × total_prcp` |

Weather data is available through 2021 in the provided dataset; features for the 2023 test year use the 2019–2021 window.

---

## Quantum Circuit Design

The encoding circuit uses **4 qubits** with a two-layer entangling structure:

1. **Layer 1:** `RY(θᵢ) · RZ(θᵢ² / π)` on each qubit — encodes both linear and quadratic angle terms.
2. **Entanglement 1:** CZ gates in a ring topology `(0→1, 1→2, 2→3, 3→0)`.
3. **Layer 2:** `RX(0.5 · (θᵢ + θᵢ₊₁))` — encodes pairwise feature sums.
4. **Entanglement 2:** CZ cross `(0→2, 1→3)` for long-range qubit correlations.

**8 Pauli observables** are measured: `{ZIII, IZII, IIZI, IIIZ}` (marginal expectations) and `{ZZII, IZZI, IIZZ, ZIIZ}` (pairwise ZZ correlators capturing entanglement structure).

---

## Models

| Model | Type | Notes |
|-------|------|-------|
| Hybrid LogReg | Primary | PCA angles + 8 quantum expectation values → LogisticRegression |
| QSVC | Quantum | ZZFeatureMap fidelity kernel; trained on balanced 120-sample subset |
| VQC | Quantum | RealAmplitudes ansatz, COBYLA optimizer, 100 iterations |
| Random Forest | Classical baseline | Full training set |
| Gradient Boosting | Classical baseline | Full training set |
| SVM (RBF) | Classical baseline | Full training set |

The final **risk score** blends hybrid LogReg probability (60%) with QSVC/VQC agreement (40%) for the top-100 highest-risk ZIPs identified by the Random Forest.

Risk categories:

| Score | Category |
|-------|----------|
| ≥ 0.70 | CRITICAL |
| ≥ 0.45 | HIGH |
| ≥ 0.20 | MODERATE |
| < 0.20 | LOW |

---

## Resource Estimation

Resources are estimated via Qiskit Aer transpilation (`optimization_level=1`) on a statevector simulator. The reported metrics include circuit depth, gate counts (1-qubit and 2-qubit), and projected shot count for QASM-based execution.

The 4-qubit circuit is designed to be **feasible on near-term hardware** (e.g., IBM Eagle / Heron processors).

---

## Installation

```bash
pip install numpy pandas scikit-learn matplotlib \
            qiskit qiskit-aer qiskit-algorithms qiskit-machine-learning
```

Python 3.9+ is recommended.

---

## Usage

```bash
python solution.py --dataset /path/to/wildfire_weather.csv --output-dir results
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | (hardcoded path) | Path to the competition wildfire/weather CSV |
| `--output-dir` | `results/` | Directory where all outputs are saved |

---

## Outputs

All files are written to `--output-dir`:

| File | Description |
|------|-------------|
| `predictions_2023.csv` | Per-ZIP risk scores, categories, and predicted labels for 2023 |
| `modeling_dataset.csv` | Full engineered feature table (train + test years) |
| `results_summary.json` | Metrics, resource estimates, and model performance summary |
| `*.png` | Visualisation plots (risk distribution, model comparison, VQC training curve, feature importance, PCA loadings) |

---

## Evaluation Summary (Task 1B)

### Advantages of the Hybrid QML Approach

- Temporal train/test split (2019–2022 train, 2023 test) ensures no future data leakage.
- Log-transformed fire acres prevent label leakage from extreme outlier events.
- Entangled CZ ring + cross topology encodes **pairwise feature interactions** via two-qubit ZZ correlators, going beyond the classical `cos(θ)` approximation of plain RY circuits.
- QSVC operates in Hilbert space via the ZZFeatureMap kernel — a genuine quantum kernel trick.
- `class_weight="balanced"` combined with PR-curve threshold tuning addresses the heavily imbalanced wildfire label distribution.
- Only **4 qubits** are required — feasible on today's IBM hardware.
- Full suite of classical baselines enables fair Task 1B comparison.
- Comprehensive metrics reported: ROC-AUC, PR-AUC, Brier score, F1, balanced accuracy, and top-10% precision/recall.

### Limitations

- QSVC and VQC are trained on ~120 samples due to simulator runtime constraints.
- No proven quantum speedup at this dataset scale.
- Statevector simulation is **noiseless** — real QPU execution is expected to degrade performance by roughly 5–15%.
- Barren plateau risk exists with deeper VQC circuits (mitigated here by COBYLA optimiser and `reps=2`).
- Weather data in the provided CSV is only available through 2021.

---

## Competition Context

| Property | Value |
|----------|-------|
| Competition | Deloitte Quantum Sustainability Challenge 2026 |
| Host | ekipa / Deloitte |
| Task | 1A (QML model + resource report) and 1B (evaluation vs. classical) |
| Submission deadline | April 10, 2026 |
| Quantum platform | IBM Qiskit / Qiskit Aer (simulator) |

---

## License

This code is submitted as a competition entry for the Deloitte Quantum Sustainability Challenge 2026. All rights reserved by the author(s).
