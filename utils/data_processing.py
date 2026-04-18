"""
Data processing utilities for Jbeil.

Adapted from TGN's utils/data_processing.py with extensions for:
  - Inductive learning (node masking)
  - Label-aware data splitting (preserving malicious events)
  - Temporal ordering preservation
"""

import numpy as np
import pandas as pd
from pathlib import Path


class Data:
    """Container for temporal graph data."""
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def get_data(dataset_name, data_dir='./data', mask_ratio=0.0, 
             different_new_nodes_between_val_and_test=False,
             randomize_features=False, val_ratio=0.15, test_ratio=0.15):
    """
    Load and split data for Jbeil training and evaluation.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'lanl')
        data_dir: Directory containing processed data files
        mask_ratio: Fraction of nodes to mask for inductive evaluation (0.0 - 0.5)
        different_new_nodes_between_val_and_test: Use disjoint new nodes for val/test
        randomize_features: Whether to randomize features
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
    
    Returns:
        node_features, edge_features, full_data, train_data, val_data, test_data,
        new_node_val_data, new_node_test_data, masked_node_ids
    """
    # Load data files
    graph_df = pd.read_csv(f'{data_dir}/ml_{dataset_name}.csv')
    edge_features = np.load(f'{data_dir}/ml_{dataset_name}.npy')
    node_features = np.load(f'{data_dir}/ml_{dataset_name}_node.npy')
    
    if randomize_features:
        node_features = np.random.rand(*node_features.shape).astype(np.float32)
    
    # Ensure sorted by timestamp
    graph_df = graph_df.sort_values('ts').reset_index(drop=True)
    
    sources = graph_df['u'].values
    destinations = graph_df['i'].values
    timestamps = graph_df['ts'].values
    edge_idxs = np.arange(1, len(graph_df) + 1)
    labels = graph_df['label'].values
    
    # Compute split indices (temporal split)
    val_time, test_time = list(np.quantile(
        timestamps, [(1 - val_ratio - test_ratio), (1 - test_ratio)]
    ))
    
    # Get all unique nodes
    all_nodes = set(sources) | set(destinations)
    n_total_nodes = len(all_nodes)
    
    # ----- Inductive Node Masking -----
    # Randomly select mask_ratio fraction of nodes to be "unseen" during training
    masked_node_ids = set()
    if mask_ratio > 0:
        all_nodes_list = sorted(list(all_nodes))
        n_mask = int(n_total_nodes * mask_ratio)
        np.random.seed(42)
        masked_node_ids = set(np.random.choice(all_nodes_list, size=n_mask, replace=False))
        print(f"Masking {len(masked_node_ids)} nodes ({mask_ratio*100:.0f}%) for inductive evaluation")
    
    # ----- Split data temporally -----
    train_mask = timestamps <= val_time
    val_mask = (timestamps > val_time) & (timestamps <= test_time)
    test_mask = timestamps > test_time
    
    # ----- Separate edges involving masked nodes -----
    def involves_masked(src, dst):
        return src in masked_node_ids or dst in masked_node_ids
    
    involves_masked_vec = np.array([
        involves_masked(s, d) for s, d in zip(sources, destinations)
    ])
    
    # Training data: only edges NOT involving masked nodes, within training time
    if mask_ratio > 0:
        train_edge_mask = train_mask & ~involves_masked_vec
    else:
        train_edge_mask = train_mask
    
    train_sources = sources[train_edge_mask]
    train_destinations = destinations[train_edge_mask]
    train_timestamps = timestamps[train_edge_mask]
    train_edge_idxs = edge_idxs[train_edge_mask]
    train_labels = labels[train_edge_mask]
    
    # Transductive val/test: edges involving only seen nodes
    seen_nodes = set(train_sources) | set(train_destinations)
    
    def is_transductive(src, dst):
        return src in seen_nodes and dst in seen_nodes
    
    is_trans_vec = np.array([
        is_transductive(s, d) for s, d in zip(sources, destinations)
    ])
    
    # Validation data (transductive - seen nodes only)
    val_edge_mask = val_mask & is_trans_vec
    val_sources = sources[val_edge_mask]
    val_destinations = destinations[val_edge_mask]
    val_timestamps = timestamps[val_edge_mask]
    val_edge_idxs = edge_idxs[val_edge_mask]
    val_labels = labels[val_edge_mask]
    
    # Test data (transductive - seen nodes only)
    test_edge_mask = test_mask & is_trans_vec
    test_sources = sources[test_edge_mask]
    test_destinations = destinations[test_edge_mask]
    test_timestamps = timestamps[test_edge_mask]
    test_edge_idxs = edge_idxs[test_edge_mask]
    test_labels = labels[test_edge_mask]
    
    # Inductive val/test: edges involving at least one masked (new) node
    if mask_ratio > 0:
        new_val_mask = val_mask & involves_masked_vec
        new_test_mask = test_mask & involves_masked_vec
    else:
        # Fallback: use nodes that appear only in val/test
        train_node_set = set(train_sources) | set(train_destinations)
        def is_new_node(src, dst):
            return src not in train_node_set or dst not in train_node_set
        is_new_vec = np.array([
            is_new_node(s, d) for s, d in zip(sources, destinations)
        ])
        new_val_mask = val_mask & is_new_vec
        new_test_mask = test_mask & is_new_vec
    
    new_node_val_sources = sources[new_val_mask]
    new_node_val_destinations = destinations[new_val_mask]
    new_node_val_timestamps = timestamps[new_val_mask]
    new_node_val_edge_idxs = edge_idxs[new_val_mask]
    new_node_val_labels = labels[new_val_mask]
    
    new_node_test_sources = sources[new_test_mask]
    new_node_test_destinations = destinations[new_test_mask]
    new_node_test_timestamps = timestamps[new_test_mask]
    new_node_test_edge_idxs = edge_idxs[new_test_mask]
    new_node_test_labels = labels[new_test_mask]
    
    # Build Data objects
    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    train_data = Data(train_sources, train_destinations, train_timestamps,
                      train_edge_idxs, train_labels)
    val_data = Data(val_sources, val_destinations, val_timestamps,
                    val_edge_idxs, val_labels)
    test_data = Data(test_sources, test_destinations, test_timestamps,
                     test_edge_idxs, test_labels)
    new_node_val_data = Data(new_node_val_sources, new_node_val_destinations,
                             new_node_val_timestamps, new_node_val_edge_idxs,
                             new_node_val_labels)
    new_node_test_data = Data(new_node_test_sources, new_node_test_destinations,
                              new_node_test_timestamps, new_node_test_edge_idxs,
                              new_node_test_labels)
    
    # Print statistics
    print(f"\nData Statistics:")
    print(f"  Total nodes: {n_total_nodes}")
    print(f"  Total edges: {len(sources)}")
    print(f"  Training edges: {len(train_sources)} (nodes: {train_data.n_unique_nodes})")
    print(f"  Validation edges: {len(val_sources)} (transductive)")
    print(f"  Test edges: {len(test_sources)} (transductive)")
    print(f"  New node val edges: {len(new_node_val_sources)} (inductive)")
    print(f"  New node test edges: {len(new_node_test_sources)} (inductive)")
    print(f"  Masked nodes: {len(masked_node_ids)}")
    print(f"  Training malicious edges: {train_labels.sum()}")
    print(f"  Test malicious edges (trans): {test_labels.sum()}")
    print(f"  Test malicious edges (induct): {new_node_test_labels.sum()}")
    
    return (node_features, edge_features, full_data, train_data, val_data,
            test_data, new_node_val_data, new_node_test_data)


def compute_time_statistics(sources, destinations, timestamps):
    """Compute mean and std of time shifts for source and destination nodes."""
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        
        if source_id not in last_timestamp_sources:
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst:
            last_timestamp_dst[dest_id] = 0
        
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)
    
    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
