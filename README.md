# Energy Storage Bidding Strategies using Approximate Dynamic Programming

This repository provides two approaches for **optimal pumped hydro storage (PHS) bidding** in **day-ahead** and **intraday** electricity markets:

1. **BADP (Backward Approximate Dynamic Programming)**  
2. **NFQCA (Neural Fitted Q-Continuous Action)**

The **BADP** approach builds upon the work of Finnah et al. to approximate the dynamic program for day-ahead and intraday decisions. The **NFQCA** approach uses an offline Reinforcement Learning (RL) pipeline with neural networks to handle continuous-action optimization and constraint enforcement.

## Contents

- [Energy Storage Bidding Strategies using Approximate Dynamic Programming](#energy-storage-bidding-strategies-using-approximate-dynamic-programming)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Installation \& Dependencies](#installation--dependencies)
  - [Quick Start](#quick-start)
  - [Detailed Workflow](#detailed-workflow)
  - [Key Files](#key-files)
  - [References \& Further Reading](#references--further-reading)

---

## Overview

**Motivation**  
- Electricity markets operate at two time-scales: **day-ahead** (hourly) and **intraday** (15-min), requiring a strategy to handle both.  
- **Pumped Hydro Storage (PHS)** can exploit price fluctuations by buying electricity when cheap (pumping) and selling when expensive (turbine generation).  
- Directly solving the full dynamic program is intractable for large state-spaces (up to 842/890-dimensional states).  
- Hence, we use:
  - **BADP**: A baseline that discretizes key state variables (reservoir level, net flow) and uses scenario-based approximate DP.  
  - **NFQCA**: A hierarchical RL method that learns separate day-ahead and intraday policies from offline data.

**Key Contributions**  
- Demonstration of how approximate DP can generate a baseline policy and produce **offline data**.  
- Introduction of a **neural fitted Q** approach for day-ahead and intraday scheduling with penalty-based constraint enforcement.  
- Full end-to-end pipeline to compare classical approximate DP with a modern RL approach.

---

## Project Structure

```
.
├─ data/
│   ├─ offline_samples/
│   │   ├─ offline_DA.pkl        # Offline data: Day-ahead
│   │   └─ offline_ID.pkl        # Offline data: Intraday
│   ├─ test_data/                # Contains .mat or .npy test price data
│   └─ ...
├─ src/
│   ├─ BADP/
│   │   ├─ badp_main.py
│   │   ├─ train.py
│   │   ├─ eval.py
│   │   └─ gen_offline_samples.py
│   ├─ Sequential_NFQCA/
│   │   ├─ nfqca_main.py
│   │   ├─ models/
│   │   │   ├─ q_network.py
│   │   │   ├─ policy_da.py
│   │   │   └─ policy_id.py
│   │   ├─ training/
│   │   │   ├─ trainer.py
│   │   │   └─ optimizers.py
│   │   └─ evaluation/
│   │       └─ evaluation.py
│   ├─ config.py
│   ├─ main.py                    # Orchestration script
│   └─ utils/                     # Helper functions (data_loader, etc.)
└─ README.md
```

- **`src/BADP/`**: Implements the Backward Approximate Dynamic Programming approach.  
- **`src/Sequential_NFQCA/`**: Implements the Neural Fitted Q approach with continuous actions.  
- **`main.py`**: A top-level script that first calls BADP to generate offline data, then calls NFQCA to train and evaluate with that data.

---

## Installation & Dependencies

1. **Python 3.8+** (or a recent version).  
2. **MATLAB** + `matlab.engine` if you plan to run the MILP problems in the BADP approach (it calls `intlinprog`).  
3. **Python Libraries** (install via pip or conda):
   - `numpy`, `pandas`, `jax`, `flax`, `matplotlib`, `scipy`, `optax`, `tqdm`  
   - Additional optional packages for file I/O (e.g., `h5py`, `pickle`).

**Example**:
```bash
conda create -n energy-bidding python=3.9
conda activate energy-bidding
pip install numpy pandas jax flax optax matplotlib scipy tqdm
```

4. **Environment Setup**:
   - If your code uses `.env`, ensure that `PYTHONPATH` is set to include `src/`.  
   - Example of `.env`:
     ```
     PYTHONPATH=./src
     ```
   - Then `main.py` should be able to import `src.BADP.badp_main` and `src.Sequential_NFQCA.nfqca_main`.

---

## Quick Start

1. **Clone** the repository:

   ```bash
   git clone https://github.com/Pkay449/Self-Schedule_Bidding.git
   ```

2. **Install** dependencies as described above.

3. **Run** the top-level pipeline:

   ```bash
   python src/main.py
   ```

4. This will:
   - **Train BADP** approach (backward pass to approximate value function),  
   - **Generate offline samples** for day-ahead (`offline_DA.pkl`) and intraday (`offline_ID.pkl`),  
   - **Evaluate** BADP policy on test data,  
   - Then **Train NFQCA** using the offline samples,  
   - **Evaluate** NFQCA policy on test data.

5. **Check** outputs:
   - BADP results are saved in `src/BADP/objects/` (e.g., `model_state` and backtest results).  
   - NFQCA results are saved in `Results/NFQCA/` (e.g., `BACKTEST_R_path.npy`, etc.).  
   - Offline data is in `data/offline_samples/offline_DA.pkl` and `data/offline_samples/offline_ID.pkl`.

---

## Detailed Workflow

1. **BADP**  
   - `badp_main.py` calls:
     1. `train_policy(...)` –  runs a scenario-based backward DP approach.  
     2. `generate_offline_data(...)` – forward simulation, storing `(state, action, reward, next_state)` in `offline_DA.pkl` & `offline_ID.pkl`.  
     3. `evaluate_policy(...)` – test set evaluation, logs performance metrics.

2. **NFQCA**  
   - `nfqca_main.py`:
     1. **Loads** `offline_DA.pkl` and `offline_ID.pkl`.  
     2. **Builds** day-ahead and intraday Q-networks + policy networks.  
     3. **Trains** them via a supervised RL approach (batch updates from offline data).  
     4. **Evaluates** final policy on test data, logs rewards, and saves time-series paths.

3. **Visualization & Backtesting**  
   - Both approaches generate `.npy` arrays for backtest paths.  
   - Each pipeline has plotting routines to show storage levels, pump/turbine usage, price trajectories, etc.

---

## Key Files

- **`src/main.py`**: Master orchestrator; calls `badp_main()` then `nfqca_main()`.  
- **`src/BADP/badp_main.py`**: Main entry for the Backward Approximate DP pipeline.  
- **`src/Sequential_NFQCA/nfqca_main.py`**: Main entry for the NFQCA pipeline.  
- **`src/BADP/gen_offline_samples.py`**: Generates offline datasets for training NFQCA.  
- **`src/Sequential_NFQCA/training/trainer.py`**: Defines the NFQCA class, handles Q-updates, policy updates, constraints.  
- **`src/config.py`**: `SimulationParams` and `TrainingParams` dataclasses for user-defined parameters (time horizon T, seeds, learning rates, etc.).  

---

## References & Further Reading

- **Finnah et al.** on Approximate Dynamic Programming for day-ahead and intraday bidding.  
- **Powell, W. B. (2019)**: A comprehensive reference on Approximate Dynamic Programming.  
- **Ziel, F.** and **Weron, R.**: Price forecasting approaches for day-ahead and intraday electricity markets.  
- **Glasserman, P.** (2003): Monte Carlo Methods in Financial Engineering (for scenario-based backward pass ideas).  

For questions or contributions, feel free to open an issue or pull request!
