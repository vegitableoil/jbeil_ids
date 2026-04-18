"""
Jbeil Model Modules.

Implements the core components of the TGN architecture adapted for Jbeil:
  - TimeEncode: Temporal encoding using learnable Fourier features
  - Memory: Stores long-term node states (Section 3.2.1)
  - MessageAggregator: Aggregates messages for batch processing
  - MessageFunction: Computes messages from interactions
  - EmbeddingModule: Computes temporal node embeddings (Section 3.2.2)
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class TimeEncode(nn.Module):
    """Time encoding using learnable parameters based on Bochner's theorem."""
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
            .float().reshape(dimension, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        t = t.unsqueeze(dim=2)
        output = torch.cos(self.w(t))
        return output


class Memory(nn.Module):
    """
    Node memory module (Section 3.2.1).
    
    Maintains a memory vector for each node that captures its interaction history.
    Updated via GRU whenever a node participates in an interaction.
    """
    def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
                 device="cpu", combination_method='sum'):
        super().__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension
        self.device = device
        
        self.__init_memory__()

    def __init_memory__(self):
        """Initialize all node memories to zero vectors."""
        self.memory = nn.Parameter(
            torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
            requires_grad=False
        )
        self.last_update = nn.Parameter(
            torch.zeros(self.n_nodes).to(self.device),
            requires_grad=False
        )
        self.messages = defaultdict(list)

    def store_raw_messages(self, nodes, node_id_to_messages):
        for node in nodes:
            self.messages[node].extend(node_id_to_messages[node])

    def get_memory(self, node_idxs):
        return self.memory[node_idxs, :]

    def set_memory(self, node_idxs, values):
        self.memory[node_idxs, :] = values

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        messages_clone = {}
        for k, v in self.messages.items():
            messages_clone[k] = [(x[0].clone(), x[1]) for x in v]
        return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()
        self.messages = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.messages[k] = [(x[0].clone(), x[1]) for x in v]

    def detach_memory(self):
        self.memory.detach_()

    def clear_messages(self, nodes):
        for node in nodes:
            self.messages[node] = []


class MemoryUpdater(nn.Module):
    """Updates node memory using GRU (Eq. 1 and 2 in the paper)."""
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super().__init__()
        self.memory = memory
        self.layer_norm = nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.device = device

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), \
            "Temporal constraint violated"

        memory = self.memory.get_memory(unique_node_ids)
        self.memory.last_update.data[unique_node_ids] = timestamps

        updated_memory = self.memory_updater(unique_messages, memory)
        updated_memory = self.layer_norm(updated_memory)

        self.memory.set_memory(unique_node_ids, updated_memory)

        return self.memory.memory.data.clone(), self.memory.last_update.data.clone()


class GRUMemoryUpdater(MemoryUpdater):
    """GRU-based memory updater."""
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super().__init__(memory, message_dimension, memory_dimension, device)
        self.memory_updater = nn.GRUCell(
            input_size=message_dimension, hidden_size=memory_dimension
        )


class RNNMemoryUpdater(MemoryUpdater):
    """RNN-based memory updater."""
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super().__init__(memory, message_dimension, memory_dimension, device)
        self.memory_updater = nn.RNNCell(
            input_size=message_dimension, hidden_size=memory_dimension
        )


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
    if module_type == "gru":
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "rnn":
        return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)


class MessageAggregator(nn.Module):
    """Aggregates messages for nodes that have multiple interactions in a batch."""
    def __init__(self, device):
        super().__init__()
        self.device = device

    def aggregate(self, node_ids, messages):
        raise NotImplementedError


class LastMessageAggregator(MessageAggregator):
    """Keep only the last message for each node."""
    def aggregate(self, node_ids, messages):
        unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []
        
        to_update_node_ids = []
        
        for node_id in unique_node_ids:
            if len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_messages.append(messages[node_id][-1][0])
                unique_timestamps.append(messages[node_id][-1][1])
        
        unique_node_ids = np.array(to_update_node_ids)
        
        if len(unique_messages) > 0:
            unique_messages = torch.stack(unique_messages)
            unique_timestamps = torch.stack(unique_timestamps)
        else:
            unique_messages = torch.tensor([]).to(self.device)
            unique_timestamps = torch.tensor([]).to(self.device)
        
        return unique_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
    """Average all messages for each node."""
    def aggregate(self, node_ids, messages):
        unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []
        
        to_update_node_ids = []
        
        for node_id in unique_node_ids:
            if len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_messages.append(
                    torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0)
                )
                unique_timestamps.append(messages[node_id][-1][1])
        
        unique_node_ids = np.array(to_update_node_ids)
        
        if len(unique_messages) > 0:
            unique_messages = torch.stack(unique_messages)
            unique_timestamps = torch.stack(unique_timestamps)
        else:
            unique_messages = torch.tensor([]).to(self.device)
            unique_timestamps = torch.tensor([]).to(self.device)
        
        return unique_node_ids, unique_messages, unique_timestamps


def get_message_aggregator(aggregator_type, device):
    if aggregator_type == "last":
        return LastMessageAggregator(device=device)
    elif aggregator_type == "mean":
        return MeanMessageAggregator(device=device)
    else:
        raise ValueError(f"Unknown aggregator: {aggregator_type}")


class MessageFunction(nn.Module):
    """Computes messages from source/destination memories and edge features."""
    pass


class MLPMessageFunction(MessageFunction):
    """MLP-based message function."""
    def __init__(self, raw_message_dimension, message_dimension):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_message_dimension, raw_message_dimension // 2),
            nn.ReLU(),
            nn.Linear(raw_message_dimension // 2, message_dimension),
        )

    def compute_message(self, raw_messages):
        return self.mlp(raw_messages)


class IdentityMessageFunction(MessageFunction):
    """Identity message function - passes through raw messages."""
    def __init__(self, raw_message_dimension, message_dimension):
        super().__init__()

    def compute_message(self, raw_messages):
        return raw_messages


def get_message_function(module_type, raw_message_dimension, message_dimension):
    if module_type == "mlp":
        return MLPMessageFunction(raw_message_dimension, message_dimension)
    elif module_type == "identity":
        return IdentityMessageFunction(raw_message_dimension, message_dimension)


class GraphAttentionEmbedding(nn.Module):
    """
    Graph attention-based embedding module (Section 3.2.2).
    
    Computes temporal node embeddings by attending over temporal neighbors
    using multi-head attention (Eq. 3 in the paper).
    """
    def __init__(self, node_features, edge_features, memory_dimension,
                 neighbor_finder, time_encoder, n_layers, n_node_features,
                 n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.embedding_dimension = embedding_dimension
        self.device = device
        self.use_memory = use_memory
        self.n_heads = n_heads
        self.dropout = dropout
        
        self.attention_models = nn.ModuleList([
            TemporalAttentionLayer(
                n_node_features=n_node_features,
                n_neighbors_features=n_node_features,
                n_edge_features=n_edge_features,
                time_dim=n_time_features,
                n_head=n_heads,
                dropout=dropout,
                output_dimension=n_node_features
            ) for _ in range(n_layers)
        ])

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers,
                          n_neighbors=20):
        """Compute temporal embedding for source_nodes at given timestamps."""
        assert n_layers >= 0
        
        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(
            torch.from_numpy(timestamps).float().to(self.device), dim=1
        )
        
        # Get node features
        source_node_features = self.node_features[source_nodes_torch, :]
        
        if self.use_memory:
            source_node_features = memory[source_nodes] + source_node_features
        
        if n_layers == 0:
            return source_node_features
        else:
            # Get temporal neighbors
            neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
                source_nodes, timestamps, n_neighbors=n_neighbors
            )
            
            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
            edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
            edge_deltas = timestamps[:, np.newaxis] - edge_times
            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
            
            # Recursively compute neighbor embeddings
            neighbors_flat = neighbors.flatten()
            neighbor_timestamps = np.repeat(timestamps, n_neighbors)
            
            neighbor_embeddings = self.compute_embedding(
                memory, neighbors_flat, neighbor_timestamps,
                n_layers=n_layers - 1, n_neighbors=n_neighbors
            )
            neighbor_embeddings = neighbor_embeddings.view(
                len(source_nodes), n_neighbors, -1
            )
            
            # Get edge features
            edge_features = self.edge_features[edge_idxs, :]
            
            # Time encoding
            source_time_encoding = self.time_encoder(
                torch.zeros_like(timestamps_torch)
            )
            neighbor_time_encoding = self.time_encoder(edge_deltas_torch)
            
            # Attention
            source_embedding = self.attention_models[n_layers - 1](
                source_node_features,
                source_time_encoding,
                neighbor_embeddings,
                neighbor_time_encoding,
                edge_features,
                neighbors_torch
            )
            
            return source_embedding


class TemporalAttentionLayer(nn.Module):
    """Multi-head temporal attention layer."""
    def __init__(self, n_node_features, n_neighbors_features, n_edge_features,
                 time_dim, n_head=2, dropout=0.1, output_dimension=None):
        super().__init__()
        
        self.n_head = n_head
        self.feat_dim = n_node_features
        self.time_dim = time_dim
        
        self.query_dim = n_node_features + time_dim
        self.key_dim = n_neighbors_features + time_dim + n_edge_features
        
        self.merger = MergeLayer(self.query_dim, n_node_features, n_node_features, output_dimension)
        
        self.multi_head_target = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            kdim=self.key_dim,
            vdim=self.key_dim,
            num_heads=n_head,
            dropout=dropout,
        )

    def forward(self, src_node_features, src_time_features, neighbor_node_features,
                neighbor_time_features, edge_features, neighbor_mask):
        
        # Query: source node [features || time_encoding]
        src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)
        query = torch.cat([src_node_features_unrolled, src_time_features], dim=2)
        
        # Key/Value: neighbor [features || time_encoding || edge_features]
        key = torch.cat([neighbor_node_features, neighbor_time_features, edge_features], dim=2)
        
        # Mask for padding (zero neighbor IDs)
        mask = neighbor_mask == 0
        mask = mask.to(self.merger.fc1.weight.device)
        
        # Transpose for nn.MultiheadAttention: (seq_len, batch, features)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        
        # Pad key to match query dimension if needed
        if key.shape[2] < query.shape[2]:
            key = torch.nn.functional.pad(key, (0, query.shape[2] - key.shape[2]))
        elif key.shape[2] > query.shape[2]:
            query = torch.nn.functional.pad(query, (0, key.shape[2] - query.shape[2]))
        
        attn_output, _ = self.multi_head_target(
            query=query, key=key, value=key,
            key_padding_mask=mask
        )
        
        attn_output = attn_output.squeeze(0)
        
        # Merge with source features
        output = self.merger(attn_output, src_node_features)
        
        return output


class MergeLayer(nn.Module):
    """Merge two feature vectors through MLP."""
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, dim3)
        self.fc2 = nn.Linear(dim3, dim4)
        self.act = nn.ReLU()
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class IdentityEmbedding(nn.Module):
    """Identity embedding - uses memory directly as embedding."""
    def __init__(self, node_features, edge_features, memory_dimension,
                 neighbor_finder, time_encoder, n_layers, n_node_features,
                 n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super().__init__()
        self.node_features = node_features
        self.device = device
        self.use_memory = use_memory
        self.neighbor_finder = neighbor_finder

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers,
                          n_neighbors=20):
        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        source_node_features = self.node_features[source_nodes_torch, :]
        if self.use_memory:
            source_node_features = memory[source_nodes] + source_node_features
        return source_node_features


def get_embedding_module(module_type, node_features, edge_features, memory_dimension,
                         neighbor_finder, time_encoder, n_layers, n_node_features,
                         n_edge_features, n_time_features, embedding_dimension, device,
                         n_heads=2, dropout=0.1, use_memory=True):
    if module_type == "graph_attention":
        return GraphAttentionEmbedding(
            node_features, edge_features, memory_dimension, neighbor_finder,
            time_encoder, n_layers, n_node_features, n_edge_features,
            n_time_features, embedding_dimension, device, n_heads, dropout, use_memory
        )
    elif module_type == "identity":
        return IdentityEmbedding(
            node_features, edge_features, memory_dimension, neighbor_finder,
            time_encoder, n_layers, n_node_features, n_edge_features,
            n_time_features, embedding_dimension, device, n_heads, dropout, use_memory
        )
    else:
        raise ValueError(f"Unknown embedding module: {module_type}")
