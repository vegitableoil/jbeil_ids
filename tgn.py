"""
Jbeil: Temporal Graph Network for Lateral Movement Detection.

This is the main model class that integrates:
  - Memory module (Section 3.2.1): stores node interaction history
  - Embedding module (Section 3.2.2): computes temporal node embeddings
  - Decoder: performs LM link prediction via edge probability computation
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

from modules.modules import (
    Memory, TimeEncode, MergeLayer,
    get_memory_updater, get_message_aggregator,
    get_message_function, get_embedding_module
)


class TGN(nn.Module):
    """
    Temporal Graph Network adapted for Jbeil's LM detection.
    
    Architecture:
        1. For each interaction event at time t:
           - Compute messages from source and destination memories
           - Update memories using GRU
        2. For link prediction:
           - Compute temporal embeddings via attention over neighbors
           - Use MLP decoder to predict edge probability
    """
    def __init__(self, neighbor_finder, node_features, edge_features, device,
                 n_layers=2, n_heads=2, dropout=0.1, use_memory=True,
                 memory_update_at_start=True, message_dimension=100,
                 memory_dimension=172, embedding_module_type="graph_attention",
                 message_function="mlp", aggregator_type="last",
                 memory_updater_type="gru", n_neighbors=None,
                 mean_time_shift_src=0, std_time_shift_src=1,
                 mean_time_shift_dst=0, std_time_shift_dst=1,
                 use_destination_embedding_in_message=False,
                 use_source_embedding_in_message=False,
                 dyrep=False):
        super().__init__()
        
        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.use_memory = use_memory
        self.memory_update_at_start = memory_update_at_start
        self.n_neighbors = n_neighbors
        
        self.n_node_features = node_features.shape[1]
        self.n_edge_features = edge_features.shape[1]
        self.n_nodes = node_features.shape[0]
        self.embedding_dimension = self.n_node_features
        self.n_time_features = self.n_node_features
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message
        self.dyrep = dyrep
        
        # Convert features to tensors on device
        self.node_raw_features = torch.from_numpy(
            node_features.astype(np.float32)
        ).to(device)
        self.edge_raw_features = torch.from_numpy(
            edge_features.astype(np.float32)
        ).to(device)
        
        # Time encoding
        self.time_encoder = TimeEncode(dimension=self.n_time_features)
        
        # Memory
        self.memory_dimension = memory_dimension
        self.memory = None
        self.message_aggregator = None
        self.message_function = None
        self.memory_updater = None
        
        if self.use_memory:
            raw_message_dim = 2 * memory_dimension + self.n_edge_features + self.time_encoder.dimension
            message_dimension = message_dimension if message_function != "identity" else raw_message_dim
            
            self.memory = Memory(
                n_nodes=self.n_nodes,
                memory_dimension=memory_dimension,
                input_dimension=self.n_node_features,
                message_dimension=message_dimension,
                device=device
            )
            
            self.message_aggregator = get_message_aggregator(aggregator_type, device)
            self.message_function = get_message_function(
                message_function, raw_message_dim, message_dimension
            )
            self.memory_updater = get_memory_updater(
                memory_updater_type, self.memory, message_dimension,
                memory_dimension, device
            )
        
        # Embedding module
        self.embedding_module_type = embedding_module_type
        self.embedding_module = get_embedding_module(
            module_type=embedding_module_type,
            node_features=self.node_raw_features,
            edge_features=self.edge_raw_features,
            memory_dimension=memory_dimension,
            neighbor_finder=neighbor_finder,
            time_encoder=self.time_encoder,
            n_layers=n_layers,
            n_node_features=self.n_node_features,
            n_edge_features=self.n_edge_features,
            n_time_features=self.n_time_features,
            embedding_dimension=self.embedding_dimension,
            device=device,
            n_heads=n_heads,
            dropout=dropout,
            use_memory=use_memory
        )
        
        # Decoder: MLP for link prediction
        self.affinity_score = MergeLayer(
            self.n_node_features, self.n_node_features,
            self.n_node_features, 1
        )

    def compute_temporal_embeddings(self, source_nodes, destination_nodes,
                                     negative_nodes, edge_times, edge_idxs,
                                     n_neighbors=20):
        """
        Compute temporal embeddings for source, destination, and negative nodes.
        Also updates node memories.
        """
        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times, edge_times])
        
        memory = None
        time_diffs = None
        
        if self.use_memory:
            if self.memory_update_at_start:
                # Update memory BEFORE computing embeddings
                memory, last_update = self.get_updated_memory(
                    list(range(self.n_nodes)),
                    self.memory.messages
                )
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
        
        # Compute embeddings
        node_embedding = self.embedding_module.compute_embedding(
            memory, nodes, timestamps, self.n_layers, n_neighbors
        )
        
        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples:2*n_samples]
        negative_node_embedding = node_embedding[2*n_samples:]
        
        if self.use_memory:
            if self.memory_update_at_start:
                # Store messages for future memory updates
                self.update_memory(source_nodes, destination_nodes, edge_times, edge_idxs)
                # Memory has already been updated at the start
                assert torch.allclose(
                    memory[positives],
                    self.memory.get_memory(positives),
                    atol=1e-5
                ), "Memory not updated correctly"
            else:
                # Update memory AFTER computing embeddings
                self.update_memory(source_nodes, destination_nodes, edge_times, edge_idxs)
                self.get_updated_memory(
                    list(range(self.n_nodes)),
                    self.memory.messages
                )
        
        return source_node_embedding, destination_node_embedding, negative_node_embedding

    def compute_edge_probabilities(self, source_nodes, destination_nodes,
                                    negative_nodes, edge_times, edge_idxs,
                                    n_neighbors=20):
        """
        Compute probabilities for positive and negative edges.
        This is the decoder component of Jbeil.
        """
        source_embedding, destination_embedding, negative_embedding = \
            self.compute_temporal_embeddings(
                source_nodes, destination_nodes, negative_nodes,
                edge_times, edge_idxs, n_neighbors
            )
        
        score = self.affinity_score(
            torch.cat([source_embedding, source_embedding], dim=0),
            torch.cat([destination_embedding, negative_embedding])
        ).squeeze(dim=0)
        
        pos_score = score[:len(source_nodes)]
        neg_score = score[len(source_nodes):]
        
        return pos_score.sigmoid(), neg_score.sigmoid()

    def update_memory(self, source_nodes, destination_nodes, edge_times, edge_idxs):
        """Create and store messages for memory update."""
        unique_sources, source_id_to_messages = self.get_raw_messages(
            source_nodes, destination_nodes, edge_times, edge_idxs
        )
        unique_destinations, destination_id_to_messages = self.get_raw_messages(
            destination_nodes, source_nodes, edge_times, edge_idxs
        )
        
        if self.memory_update_at_start:
            self.memory.store_raw_messages(unique_sources, source_id_to_messages)
            self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
        else:
            self.memory.store_raw_messages(unique_sources, source_id_to_messages)
            self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)

    def get_raw_messages(self, source_nodes, destination_nodes, edge_times, edge_idxs):
        """Compute raw messages from interactions (Eq. 1 and 2)."""
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
        edge_features = self.edge_raw_features[edge_idxs]
        
        source_memory = self.memory.get_memory(source_nodes)
        destination_memory = self.memory.get_memory(destination_nodes)
        
        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(
            source_time_delta.unsqueeze(dim=1)
        ).view(len(source_nodes), -1)
        
        # Message = concat(source_memory, destination_memory, edge_features, time_encoding)
        source_message = torch.cat([
            source_memory, destination_memory,
            edge_features, source_time_delta_encoding
        ], dim=1)
        
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)
        
        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append(
                (source_message[i], edge_times[i])
            )
        
        return unique_sources, messages

    def get_updated_memory(self, nodes, messages):
        """Aggregate messages and update memory for given nodes."""
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(nodes, messages)
        
        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)
        
        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(
            unique_nodes, unique_messages, unique_timestamps
        )
        
        return updated_memory, updated_last_update

    def set_neighbor_finder(self, neighbor_finder):
        """Set neighbor finder for the model and embedding module."""
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder
