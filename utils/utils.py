"""
Utility classes for Jbeil: neighbor finding, edge sampling, early stopping.
Adapted from TGN's utils/utils.py.
"""

import numpy as np
import torch


class MergeLayer(torch.nn.Module):
    """MLP layer for computing edge probabilities from node embeddings."""
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class EarlyStopMonitor:
    """Monitor for early stopping based on validation metric."""
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round


class RandEdgeSampler:
    """Random edge sampler for generating negative samples."""
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = seed
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = np.random.RandomState()

    def sample(self, size):
        src_index = self.random_state.randint(0, len(self.src_list), size)
        dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


class NeighborFinder:
    """
    Finds temporal neighbors for nodes in the graph.
    Maintains adjacency lists sorted by timestamp for efficient temporal queries.
    """
    def __init__(self, adj_list, uniform=False, seed=None):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []
        
        for neighbors in adj_list:
            # Neighbors is a list of (neighbor, edge_idx, timestamp)
            sorted_neighbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighbors]))
        
        self.uniform = uniform
        self.seed = seed
        self.random_state = np.random.RandomState(seed)

    def find_before(self, src_idx, cut_time):
        """Find neighbors of src_idx before cut_time."""
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
        return (self.node_to_neighbors[src_idx][:i],
                self.node_to_edge_idxs[src_idx][:i],
                self.node_to_edge_timestamps[src_idx][:i])

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        """
        Get temporal neighbors for a batch of source nodes.
        
        Args:
            source_nodes: array of source node IDs
            timestamps: array of timestamps (find neighbors before these times)
            n_neighbors: number of neighbors to sample
        
        Returns:
            neighbors, edge_idxs, edge_times: arrays of shape (len(source_nodes), n_neighbors)
        """
        assert len(source_nodes) == len(timestamps)
        
        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(np.int32)
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(np.float32)
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(np.int32)
        
        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_node = int(source_node)
            
            if source_node >= len(self.node_to_neighbors):
                continue
                
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(
                source_node, timestamp)
            
            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:
                    sampled_idx = self.random_state.randint(
                        0, len(source_neighbors), n_neighbors)
                else:
                    # Take the most recent n_neighbors
                    sampled_idx = source_neighbors.shape[0] - 1 - np.arange(
                        min(n_neighbors, len(source_neighbors)))
                
                neighbors[i, :len(sampled_idx)] = source_neighbors[sampled_idx]
                edge_idxs[i, :len(sampled_idx)] = source_edge_idxs[sampled_idx]
                edge_times[i, :len(sampled_idx)] = source_edge_times[sampled_idx]
        
        return neighbors, edge_idxs, edge_times


def get_neighbor_finder(data, uniform, max_node_idx=None):
    """Build a NeighborFinder from interaction data."""
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    
    for source, destination, edge_idx, timestamp in zip(
        data.sources, data.destinations, data.edge_idxs, data.timestamps
    ):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))
    
    return NeighborFinder(adj_list, uniform=uniform)
