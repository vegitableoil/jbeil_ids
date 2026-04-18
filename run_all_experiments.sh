#!/bin/bash
# ============================================================
# Jbeil: Run All Inductive Reasoning Experiments (Table 2)
# ============================================================
# This script reproduces the three experiments from Table 2:
#   Experiment 1: 30% masking → AUC ~99.73% (inductive)
#   Experiment 2: 40% masking → AUC ~94.59% (inductive)
#   Experiment 3: 50% masking → AUC ~74.55% (inductive)
#
# Prerequisites:
#   1. LANL data preprocessed: python utils/preprocess_lanl.py
#   2. Dependencies installed: pip install torch numpy pandas scikit-learn
# ============================================================

set -e

echo "============================================================"
echo "Jbeil: Temporal Graph-Based Inductive LM Detection"
echo "Reproducing Table 2 Experiments"
echo "============================================================"

# Common hyperparameters (from Section 4.4.1)
COMMON_ARGS="--data lanl --use_memory --n_epoch 10 --lr 0.005 --patience 5 \
    --n_degree 10 --n_head 2 --n_layer 1 --memory_updater gru \
    --message_function identity --aggregator last --memory_dim 172 --bs 200"

echo ""
echo "============================================================"
echo "Experiment 1: 30% Node Masking (9,886 training nodes)"
echo "Target: AUC 99.82% (trans) / 99.73% (induct)"
echo "============================================================"
python train_jbeil.py $COMMON_ARGS --mask_ratio 0.3 --prefix jbeil_exp1

echo ""
echo "============================================================"
echo "Experiment 2: 40% Node Masking (8,423 training nodes)"
echo "Target: AUC 94.76% (trans) / 94.59% (induct)"
echo "============================================================"
python train_jbeil.py $COMMON_ARGS --mask_ratio 0.4 --prefix jbeil_exp2

echo ""
echo "============================================================"
echo "Experiment 3: 50% Node Masking (6,943 training nodes)"
echo "Target: AUC 75.62% (trans) / 74.55% (induct)"
echo "============================================================"
python train_jbeil.py $COMMON_ARGS --mask_ratio 0.5 --prefix jbeil_exp3

echo ""
echo "============================================================"
echo "All experiments complete!"
echo "Results saved in ./results/"
echo "Models saved in ./saved_models/"
echo "Logs saved in ./log/"
echo "============================================================"
