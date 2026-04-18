# Jbeil: Reproducing Experiments

This codebase reproduces the experiments from:

> **Jbeil: Temporal Graph-Based Inductive Learning to Infer Lateral Movement in Evolving Enterprise Networks**  
> Khoury et al., IEEE S&P 2024

The implementation adapts the [TGN](https://github.com/twitter-research/tgn) framework for lateral movement detection on authentication logs, following the architecture described in Section 3 of the paper.

---

## Prerequisites

```bash
pip install torch numpy pandas scikit-learn
```

Hardware used in the paper: GPU VM with 32 vCPUs, 125GB RAM, NVIDIA GPU (PyTorch 1.11+CUDA 11.3).

---

## Directory Structure

```
jbeil_reproduce/
├── README.md
├── train_jbeil.py              # Main training script (all experiments)
├── run_all_experiments.sh       # Shell script to run all experiments
├── model/
│   └── tgn.py                  # TGN model adapted for Jbeil
├── modules/
│   └── modules.py              # Memory, attention, embedding modules
├── evaluation/
│   └── evaluation.py           # AUC, AP, Precision, Recall, G-mean threshold
├── utils/
│   ├── preprocess_lanl.py      # Step 1-4: LANL auth log preprocessing
│   ├── data_processing.py      # Data loading with inductive node masking
│   ├── augment_threats.py      # Threat sample augmentation (3 scenarios)
│   └── utils.py                # Neighbor finder, samplers, early stopping
└── data/                       # Processed data files (created by preprocessing)
```

---

## Step 1: Obtain the LANL Dataset

Download from https://csr.lanl.gov/data/cyber1/ and place in `./lanl_data/`:

```
lanl_data/
├── auth.txt.gz        # Authentication logs (~12GB compressed)
└── redteam.txt.gz     # Red team labels
```

---

## Step 2: Preprocess the Data

This implements the 4-step pipeline from Section 3.1 (Figure 4):
1. Parse authentication log attributes (timestamp, src, usr, dst)
2. Extract graph maps (in/out-degree dictionaries, Algorithm 1)
3. Calculate 9 graph features per edge
4. Produce temporal graph representation

```bash
python utils/preprocess_lanl.py --data_dir ./lanl_data --output_dir ./data
```

Output files in `./data/`:
- `ml_lanl.csv` — sorted edge list (u, i, ts, label)
- `ml_lanl.npy` — normalized edge features (N×9)
- `ml_lanl_node.npy` — node features (initialized to zeros)

---

## Step 3: Run Experiments

### Experiment 1 — 30% Node Masking (Table 2, Row 1)

**Paper target:** AUC 99.82% (transductive), 99.73% (inductive)

```bash
python train_jbeil.py \
    --data lanl \
    --mask_ratio 0.3 \
    --use_memory \
    --n_epoch 10 \
    --lr 0.005 \
    --patience 5 \
    --prefix jbeil_exp1
```

### Experiment 2 — 40% Node Masking (Table 2, Row 2)

**Paper target:** AUC 94.76% (transductive), 94.59% (inductive)

```bash
python train_jbeil.py \
    --data lanl \
    --mask_ratio 0.4 \
    --use_memory \
    --n_epoch 10 \
    --lr 0.005 \
    --patience 5 \
    --prefix jbeil_exp2
```

### Experiment 3 — 50% Node Masking (Table 2, Row 3)

**Paper target:** AUC 75.62% (transductive), 74.55% (inductive)

```bash
python train_jbeil.py \
    --data lanl \
    --mask_ratio 0.5 \
    --use_memory \
    --n_epoch 10 \
    --lr 0.005 \
    --patience 5 \
    --prefix jbeil_exp3
```

### Run All Three Experiments

```bash
bash run_all_experiments.sh
```

---

## Step 4 (Optional): Augmented Threat Scenarios (Table 4)

Generate the three attack scenarios from Section 4.2.2:

```bash
python utils/augment_threats.py --data_dir ./data --dataset lanl
```

This creates `ml_lanl_scenario1.csv`, `ml_lanl_scenario2.csv`, `ml_lanl_scenario3.csv`.

Then train on each scenario:

```bash
# Scenario 1 (695 attacks, limited-knowledge attacker)
python train_jbeil.py --data lanl_scenario1 --mask_ratio 0.3 --use_memory --prefix jbeil_s1

# Scenario 2 (606 attacks, full-topology attacker)
python train_jbeil.py --data lanl_scenario2 --mask_ratio 0.3 --use_memory --prefix jbeil_s2

# Scenario 3 (500 attacks, credential-aware attacker)
python train_jbeil.py --data lanl_scenario3 --mask_ratio 0.3 --use_memory --prefix jbeil_s3
```

**Paper targets (Table 4):** All three scenarios achieve ~99.5% AUC in both transductive and inductive settings.

---

## Expected Results (Table 2)

| Experiment | Training Nodes | Transductive AUC | Inductive AUC |
|------------|---------------|------------------|---------------|
| 1 (30%)    | 9,886         | 99.82%           | 99.73%        |
| 2 (40%)    | 8,423         | 94.76%           | 94.59%        |
| 3 (50%)    | 6,943         | 75.62%           | 74.55%        |

Key observation: inductive performance closely matches transductive, demonstrating Jbeil's ability to generalize to unseen nodes.

---

## Hyperparameters (from paper)

| Parameter        | Value |
|------------------|-------|
| Epochs           | 10    |
| Learning rate    | 0.005 |
| Early stopping   | 5     |
| Batch size       | 200   |
| Neighbors        | 10    |
| Attention heads  | 2     |
| Graph layers     | 1     |
| Memory updater   | GRU   |
| Message function | identity |
| Aggregator       | last  |
| Memory dimension | 172   |

---

## Architecture Summary

```
Authentication Logs
       │
       ▼
┌─────────────────────┐
│  Pre-processing     │  Graph maps → 9 features per edge
│  (Section 3.1)      │  + Threat augmentation (BFS-based)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Temporal Graph     │  Continuous-time dynamic graph
│  Representation     │  G = {e_t0, e_t1, ...}
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Memory Module      │  GRU-based node memory (Eq. 1-2)
│  (Section 3.2.1)    │  Stores interaction history per node
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Embedding Module   │  Multi-head temporal attention (Eq. 3)
│  (Section 3.2.2)    │  Aggregates neighbor memories + features
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Decoder (MLP)      │  Edge probability → LM link prediction
│                     │  Self-supervised BCE loss
└─────────────────────┘
```

---

## Notes

- The official implementation is at https://github.com/LMscope/Jbeil
- This reproduction follows the paper's description and adapts the TGN codebase
- The LANL dataset has 15,610 nodes and ~49M edges spanning 58 days
- Preprocessing is the most time-consuming step (~1-2 hours for full LANL data)
- Training Experiment 1 takes ~5-15 minutes depending on GPU
