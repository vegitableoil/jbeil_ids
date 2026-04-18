"""
Jbeil: Training Script for Self-Supervised Inductive LM Detection.

Reproduces the experiments from Section 4.4 of the paper:
  - Experiment 1: 30% node masking (Table 2, row 1)
  - Experiment 2: 40% node masking (Table 2, row 2)
  - Experiment 3: 50% node masking (Table 2, row 3)

Training configuration (from paper):
  - 10 epochs
  - Learning rate: 0.005
  - Patience: 5
  - Self-supervised link prediction with both benign and malicious data

Usage:
    # Experiment 1 (30% masking - main result)
    python train_jbeil.py --data lanl --mask_ratio 0.3 --use_memory --prefix jbeil_exp1

    # Experiment 2 (40% masking)
    python train_jbeil.py --data lanl --mask_ratio 0.4 --use_memory --prefix jbeil_exp2

    # Experiment 3 (50% masking)
    python train_jbeil.py --data lanl --mask_ratio 0.5 --use_memory --prefix jbeil_exp3

    # Transductive only (0% masking, for comparison)
    python train_jbeil.py --data lanl --mask_ratio 0.0 --use_memory --prefix jbeil_trans
"""

import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation.evaluation import eval_edge_prediction, eval_lm_detection
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('Jbeil: Temporal Graph-Based Inductive LM Detection')

# Data
parser.add_argument('-d', '--data', type=str, default='lanl',
                    help='Dataset name (e.g., lanl)')
parser.add_argument('--data_dir', type=str, default='./data',
                    help='Directory containing processed data')

# Inductive settings
parser.add_argument('--mask_ratio', type=float, default=0.3,
                    help='Fraction of nodes to mask for inductive evaluation (0.0-0.5)')

# Model architecture
parser.add_argument('--n_degree', type=int, default=10,
                    help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2,
                    help='Number of attention heads')
parser.add_argument('--n_layer', type=int, default=1,
                    help='Number of graph attention layers')
parser.add_argument('--node_dim', type=int, default=100,
                    help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100,
                    help='Dimensions of the time embedding')
parser.add_argument('--message_dim', type=int, default=100,
                    help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172,
                    help='Dimensions of the memory')

# Training
parser.add_argument('--bs', type=int, default=200, help='Batch size')
parser.add_argument('--n_epoch', type=int, default=10,
                    help='Number of epochs (paper uses 10)')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Learning rate (paper uses 0.005)')
parser.add_argument('--patience', type=int, default=5,
                    help='Patience for early stopping (paper uses 5)')
parser.add_argument('--n_runs', type=int, default=1,
                    help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1,
                    help='Dropout probability')
parser.add_argument('--backprop_every', type=int, default=1,
                    help='Backprop frequency')

# Model options
parser.add_argument('--use_memory', action='store_true',
                    help='Use node memory (required for Jbeil)')
parser.add_argument('--embedding_module', type=str, default='graph_attention',
                    choices=['graph_attention', 'identity'],
                    help='Type of embedding module')
parser.add_argument('--message_function', type=str, default='identity',
                    choices=['mlp', 'identity'],
                    help='Type of message function')
parser.add_argument('--memory_updater', type=str, default='gru',
                    choices=['gru', 'rnn'],
                    help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default='last',
                    choices=['last', 'mean'],
                    help='Type of message aggregator')

# Other
parser.add_argument('--prefix', type=str, default='jbeil',
                    help='Prefix for saved files')
parser.add_argument('--gpu', type=int, default=0, help='GPU index')
parser.add_argument('--uniform', action='store_true',
                    help='Uniform temporal neighbor sampling')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# ============================================================
# Configuration
# ============================================================
BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
MASK_RATIO = args.mask_ratio

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
Path("./results/").mkdir(parents=True, exist_ok=True)
Path("./log/").mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

# ============================================================
# Logging
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'log/{args.prefix}_{time.time()}.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

logger.info("=" * 80)
logger.info("Jbeil: Temporal Graph-Based Inductive LM Detection")
logger.info("=" * 80)
logger.info(f"Configuration: {args}")
logger.info(f"Mask ratio: {MASK_RATIO} ({MASK_RATIO*100:.0f}% nodes masked)")

# ============================================================
# Load Data
# ============================================================
logger.info("Loading data...")
node_features, edge_features, full_data, train_data, val_data, test_data, \
    new_node_val_data, new_node_test_data = get_data(
        DATA, data_dir=args.data_dir, mask_ratio=MASK_RATIO
    )

# Initialize neighbor finders
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(
    new_node_val_data.sources, new_node_val_data.destinations, seed=1
)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(
    new_node_test_data.sources, new_node_test_data.destinations, seed=3
)

# Device
device_string = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
logger.info(f"Using device: {device}")

# Time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

# ============================================================
# Training Loop
# ============================================================
for run in range(args.n_runs):
    results_path = f"results/{args.prefix}_{run}.pkl" if run > 0 else f"results/{args.prefix}.pkl"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Run {run + 1}/{args.n_runs}")
    logger.info(f"{'='*60}")
    
    # Initialize Model
    tgn = TGN(
        neighbor_finder=train_ngh_finder,
        node_features=node_features,
        edge_features=edge_features,
        device=device,
        n_layers=NUM_LAYER,
        n_heads=NUM_HEADS,
        dropout=DROP_OUT,
        use_memory=USE_MEMORY,
        message_dimension=MESSAGE_DIM,
        memory_dimension=MEMORY_DIM,
        memory_update_at_start=True,
        embedding_module_type=args.embedding_module,
        message_function=args.message_function,
        aggregator_type=args.aggregator,
        memory_updater_type=args.memory_updater,
        n_neighbors=NUM_NEIGHBORS,
        mean_time_shift_src=mean_time_shift_src,
        std_time_shift_src=std_time_shift_src,
        mean_time_shift_dst=mean_time_shift_dst,
        std_time_shift_dst=std_time_shift_dst,
    )
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
    tgn = tgn.to(device)
    
    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)
    
    logger.info(f"Training instances: {num_instance}")
    logger.info(f"Batches per epoch: {num_batch}")
    
    val_aps = []
    new_nodes_val_aps = []
    epoch_times = []
    train_losses = []
    
    early_stopper = EarlyStopMonitor(max_round=args.patience)
    
    for epoch in range(NUM_EPOCH):
        start_epoch = time.time()
        
        # ---- Training ----
        if USE_MEMORY:
            tgn.memory.__init_memory__()
        
        tgn.set_neighbor_finder(train_ngh_finder)
        m_loss = []
        
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCH}")
        
        for k in range(0, num_batch, args.backprop_every):
            loss = 0
            optimizer.zero_grad()
            
            for j in range(args.backprop_every):
                batch_idx = k + j
                if batch_idx >= num_batch:
                    continue
                
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)
                
                sources_batch = train_data.sources[start_idx:end_idx]
                destinations_batch = train_data.destinations[start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]
                
                size = len(sources_batch)
                _, negatives_batch = train_rand_sampler.sample(size)
                
                with torch.no_grad():
                    pos_label = torch.ones(size, dtype=torch.float, device=device)
                    neg_label = torch.zeros(size, dtype=torch.float, device=device)
                
                tgn = tgn.train()
                pos_prob, neg_prob = tgn.compute_edge_probabilities(
                    sources_batch, destinations_batch, negatives_batch,
                    timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS
                )
                
                loss += criterion(pos_prob.squeeze(), pos_label) + \
                        criterion(neg_prob.squeeze(), neg_label)
            
            loss /= args.backprop_every
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())
            
            if USE_MEMORY:
                tgn.memory.detach_memory()
        
        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)
        
        # ---- Validation ----
        tgn.set_neighbor_finder(full_ngh_finder)
        
        if USE_MEMORY:
            train_memory_backup = tgn.memory.backup_memory()
        
        # Transductive validation
        val_ap, val_auc = eval_edge_prediction(
            model=tgn, negative_edge_sampler=val_rand_sampler,
            data=val_data, n_neighbors=NUM_NEIGHBORS
        )
        
        if USE_MEMORY:
            val_memory_backup = tgn.memory.backup_memory()
            tgn.memory.restore_memory(train_memory_backup)
        
        # Inductive validation (unseen nodes)
        nn_val_ap, nn_val_auc = eval_edge_prediction(
            model=tgn, negative_edge_sampler=nn_val_rand_sampler,
            data=new_node_val_data, n_neighbors=NUM_NEIGHBORS
        )
        
        if USE_MEMORY:
            tgn.memory.restore_memory(val_memory_backup)
        
        val_aps.append(val_ap)
        new_nodes_val_aps.append(nn_val_ap)
        train_losses.append(np.mean(m_loss))
        
        logger.info(f"  Epoch {epoch+1} | Time: {epoch_time:.1f}s | "
                     f"Loss: {np.mean(m_loss):.4f}")
        logger.info(f"  Transductive Val | AUC: {val_auc:.4f} | AP: {val_ap:.4f}")
        logger.info(f"  Inductive Val    | AUC: {nn_val_auc:.4f} | AP: {nn_val_ap:.4f}")
        
        # Early stopping
        if early_stopper.early_stop_check(val_ap):
            logger.info(f"  Early stopping at epoch {epoch+1}")
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            tgn.load_state_dict(torch.load(best_model_path))
            tgn.eval()
            break
        else:
            torch.save(tgn.state_dict(), get_checkpoint_path(epoch))
    
    # ============================================================
    # Testing
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    
    if USE_MEMORY:
        val_memory_backup = tgn.memory.backup_memory()
    
    tgn.embedding_module.neighbor_finder = full_ngh_finder
    
    # --- Transductive Test ---
    trans_results = eval_lm_detection(
        model=tgn, negative_edge_sampler=test_rand_sampler,
        data=test_data, n_neighbors=NUM_NEIGHBORS
    )
    
    logger.info(f"\nTransductive Test Results (seen nodes):")
    logger.info(f"  Nodes:     {test_data.n_unique_nodes}")
    logger.info(f"  AUC:       {trans_results['auc']*100:.2f}%")
    logger.info(f"  AP:        {trans_results['ap']*100:.2f}%")
    logger.info(f"  Precision: {trans_results['precision']*100:.2f}%")
    logger.info(f"  Recall:    {trans_results['recall']*100:.2f}%")
    
    if USE_MEMORY:
        tgn.memory.restore_memory(val_memory_backup)
    
    # --- Inductive Test ---
    induct_results = eval_lm_detection(
        model=tgn, negative_edge_sampler=nn_test_rand_sampler,
        data=new_node_test_data, n_neighbors=NUM_NEIGHBORS
    )
    
    logger.info(f"\nInductive Test Results (unseen nodes):")
    logger.info(f"  Nodes:     {new_node_test_data.n_unique_nodes}")
    logger.info(f"  AUC:       {induct_results['auc']*100:.2f}%")
    logger.info(f"  AP:        {induct_results['ap']*100:.2f}%")
    logger.info(f"  Precision: {induct_results['precision']*100:.2f}%")
    logger.info(f"  Recall:    {induct_results['recall']*100:.2f}%")
    
    # ============================================================
    # Summary (matching Table 2 format)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info(f"SUMMARY (Mask Ratio: {MASK_RATIO*100:.0f}%)")
    logger.info(f"{'='*60}")
    logger.info(f"{'Metric':<15} {'Transductive':>15} {'Inductive':>15}")
    logger.info(f"{'-'*45}")
    logger.info(f"{'Nodes #':<15} {test_data.n_unique_nodes:>15} {new_node_test_data.n_unique_nodes:>15}")
    logger.info(f"{'Precision (%)':<15} {trans_results['precision']*100:>15.2f} {induct_results['precision']*100:>15.2f}")
    logger.info(f"{'Recall (%)':<15} {trans_results['recall']*100:>15.2f} {induct_results['recall']*100:>15.2f}")
    logger.info(f"{'AP (%)':<15} {trans_results['ap']*100:>15.2f} {induct_results['ap']*100:>15.2f}")
    logger.info(f"{'AUC (%)':<15} {trans_results['auc']*100:>15.2f} {induct_results['auc']*100:>15.2f}")
    
    # Save results
    pickle.dump({
        "val_aps": val_aps,
        "new_nodes_val_aps": new_nodes_val_aps,
        "train_losses": train_losses,
        "epoch_times": epoch_times,
        "transductive_results": trans_results,
        "inductive_results": induct_results,
        "mask_ratio": MASK_RATIO,
        "args": vars(args)
    }, open(results_path, "wb"))
    
    # Save model
    torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"\nModel saved to {MODEL_SAVE_PATH}")
    logger.info(f"Results saved to {results_path}")
