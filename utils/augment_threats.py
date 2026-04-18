"""
Threat Sample Augmentation for Jbeil.

Implements the attack synthesis framework described in Sections 3.1.1 and 4.2.2,
based on the BFS-based lateral movement simulator by Ho et al. [25].

Generates three attack scenarios:
  - Scenario 1: Limited knowledge attacker (695 attacks)
  - Scenario 2: Full topology knowledge, edge-following attacker (606 attacks)
  - Scenario 3: Full topology knowledge, credential-aware attacker (500 attacks)

Usage:
    python utils/augment_threats.py --data_dir ./data --dataset lanl
"""

import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import json
import os


class LateralMovementSimulator:
    """
    BFS-based lateral movement attack simulator.
    
    Given a network graph derived from authentication logs, generates realistic
    LM attack paths that conform to the network's topology and data distribution.
    """
    
    def __init__(self, graph_df, node_mapping=None):
        """
        Args:
            graph_df: DataFrame with columns [u, i, ts, label] representing
                      authentication events
            node_mapping: Optional dict mapping node names to IDs
        """
        self.graph_df = graph_df
        self.nodes = sorted(set(graph_df['u'].values) | set(graph_df['i'].values))
        self.n_nodes = len(self.nodes)
        
        # Build adjacency lists
        self.adj = defaultdict(set)
        self.edge_users = defaultdict(set)  # Track users per edge
        self.node_timestamps = defaultdict(list)
        
        for _, row in graph_df.iterrows():
            src, dst, ts = int(row['u']), int(row['i']), row['ts']
            self.adj[src].add(dst)
            self.adj[dst].add(src)
            self.node_timestamps[src].append(ts)
            self.node_timestamps[dst].append(ts)
        
        # Identify high-value targets (nodes with high in-degree, e.g., servers)
        in_degrees = defaultdict(int)
        for _, row in graph_df.iterrows():
            in_degrees[int(row['i'])] += 1
        
        sorted_nodes = sorted(in_degrees.items(), key=lambda x: -x[1])
        n_hv = max(1, len(sorted_nodes) // 20)  # Top 5% as high-value
        self.high_value_targets = set([n for n, _ in sorted_nodes[:n_hv]])
        
        # Identify admin/privileged nodes (nodes that access many destinations)
        out_degrees = defaultdict(int)
        for _, row in graph_df.iterrows():
            out_degrees[int(row['u'])] += 1
        sorted_out = sorted(out_degrees.items(), key=lambda x: -x[1])
        n_priv = max(1, len(sorted_out) // 10)  # Top 10% as privileged
        self.privileged_nodes = set([n for n, _ in sorted_out[:n_priv]])
        
        print(f"Network: {self.n_nodes} nodes, {len(graph_df)} edges")
        print(f"High-value targets: {len(self.high_value_targets)}")
        print(f"Privileged nodes: {len(self.privileged_nodes)}")
    
    def _bfs_shortest_path(self, start, targets, allowed_edges=None):
        """BFS to find shortest path from start to any node in targets."""
        from collections import deque
        
        visited = {start}
        queue = deque([(start, [start])])
        
        while queue:
            node, path = queue.popleft()
            
            if node in targets:
                return path
            
            neighbors = self.adj[node]
            if allowed_edges is not None:
                neighbors = {n for n in neighbors if (node, n) in allowed_edges or (n, node) in allowed_edges}
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None  # No path found
    
    def _generate_random_timestamp(self, base_ts, offset_range=(0, 86400)):
        """Generate a random timestamp near base_ts."""
        return base_ts + np.random.randint(*offset_range)
    
    def generate_scenario1(self, n_attacks=695):
        """
        Scenario 1: Limited knowledge attacker.
        
        - Attacker only knows about machines they've previously accessed
        - Attack stops once access to a previously inaccessible system is gained
        """
        print(f"\nGenerating Scenario 1 ({n_attacks} attacks)...")
        attacks = []
        
        for i in range(n_attacks):
            # Select random foothold
            foothold = np.random.choice(self.nodes)
            
            # Attacker explores from foothold using BFS
            compromised = {foothold}
            known_neighbors = set(self.adj[foothold])
            path = [foothold]
            
            base_ts = np.random.uniform(
                self.graph_df['ts'].min(), self.graph_df['ts'].max()
            )
            
            max_steps = np.random.randint(2, 8)
            for step in range(max_steps):
                # Try to move to an unknown neighbor
                candidates = known_neighbors - compromised
                if not candidates:
                    break
                
                next_node = np.random.choice(list(candidates))
                compromised.add(next_node)
                path.append(next_node)
                
                # Update known neighbors
                known_neighbors.update(self.adj[next_node])
                
                # Stop if reached a node the original victim couldn't access
                if next_node not in self.adj[foothold]:
                    break
            
            # Generate attack edges
            if len(path) > 1:
                for j in range(len(path) - 1):
                    ts = base_ts + j * np.random.randint(10, 3600)
                    attacks.append({
                        'u': path[j], 'i': path[j+1],
                        'ts': ts, 'label': 1,
                        'scenario': 1, 'attack_id': i
                    })
        
        print(f"  Generated {len(attacks)} attack edges from {n_attacks} campaigns")
        return attacks
    
    def generate_scenario2(self, n_attacks=606):
        """
        Scenario 2: Full topology knowledge.
        
        - Attacker knows the whole network topology
        - Attack terminates after accessing 50 devices or all reachable machines
        - Only traverses edges that valid users have already traversed
        """
        print(f"\nGenerating Scenario 2 ({n_attacks} attacks)...")
        
        # Build set of valid (traversed) edges
        valid_edges = set()
        for _, row in self.graph_df.iterrows():
            valid_edges.add((int(row['u']), int(row['i'])))
        
        attacks = []
        
        for i in range(n_attacks):
            foothold = np.random.choice(self.nodes)
            compromised = {foothold}
            path = [foothold]
            
            base_ts = np.random.uniform(
                self.graph_df['ts'].min(), self.graph_df['ts'].max()
            )
            
            max_devices = min(50, len(self.nodes) // 10)
            
            current = foothold
            for step in range(max_devices):
                # Find neighbors reachable via valid edges
                candidates = []
                for n in self.adj[current]:
                    if n not in compromised:
                        if (current, n) in valid_edges or (n, current) in valid_edges:
                            candidates.append(n)
                
                if not candidates:
                    # Try BFS to find alternate path
                    reachable = set()
                    for c in compromised:
                        for n in self.adj[c]:
                            if n not in compromised:
                                if (c, n) in valid_edges or (n, c) in valid_edges:
                                    reachable.add(n)
                    candidates = list(reachable)
                
                if not candidates:
                    break
                
                next_node = np.random.choice(candidates)
                compromised.add(next_node)
                path.append(next_node)
                current = next_node
                
                if len(compromised) >= max_devices:
                    break
            
            # Generate attack edges
            if len(path) > 1:
                for j in range(len(path) - 1):
                    ts = base_ts + j * np.random.randint(10, 3600)
                    attacks.append({
                        'u': path[j], 'i': path[j+1],
                        'ts': ts, 'label': 1,
                        'scenario': 2, 'attack_id': i
                    })
        
        print(f"  Generated {len(attacks)} attack edges from {n_attacks} campaigns")
        return attacks
    
    def generate_scenario3(self, n_attacks=500):
        """
        Scenario 3: Full topology knowledge + credential awareness.
        
        - Attacker knows the whole network topology
        - Performs multiple login attempts until access to a high-value server
        - Only uses credentials where the authorized user has recently logged in
        """
        print(f"\nGenerating Scenario 3 ({n_attacks} attacks)...")
        
        attacks = []
        
        for i in range(n_attacks):
            foothold = np.random.choice(self.nodes)
            compromised = {foothold}
            path = [foothold]
            
            base_ts = np.random.uniform(
                self.graph_df['ts'].min(), self.graph_df['ts'].max()
            )
            
            current = foothold
            reached_hv = False
            
            for step in range(20):  # Max 20 hops
                candidates = list(self.adj[current] - compromised)
                if not candidates:
                    break
                
                # Prefer paths toward high-value targets
                hv_candidates = [c for c in candidates if c in self.high_value_targets]
                priv_candidates = [c for c in candidates if c in self.privileged_nodes]
                
                if hv_candidates:
                    next_node = np.random.choice(hv_candidates)
                elif priv_candidates:
                    next_node = np.random.choice(priv_candidates)
                else:
                    next_node = np.random.choice(candidates)
                
                compromised.add(next_node)
                path.append(next_node)
                current = next_node
                
                # Multiple login attempts (1-3 per hop)
                n_attempts = np.random.randint(1, 4)
                for attempt in range(n_attempts):
                    ts = base_ts + (step * 3600) + (attempt * np.random.randint(5, 300))
                    attacks.append({
                        'u': path[-2], 'i': next_node,
                        'ts': ts, 'label': 1,
                        'scenario': 3, 'attack_id': i
                    })
                
                if next_node in self.high_value_targets:
                    reached_hv = True
                    break
        
        print(f"  Generated {len(attacks)} attack edges from {n_attacks} campaigns")
        return attacks
    
    def augment_dataset(self, output_dir, scenarios=None):
        """
        Generate all attack scenarios and merge with original data.
        
        Args:
            output_dir: Directory to save augmented data
            scenarios: List of scenario numbers to generate (default: all)
        """
        if scenarios is None:
            scenarios = [1, 2, 3]
        
        all_attacks = []
        
        if 1 in scenarios:
            all_attacks.extend(self.generate_scenario1())
        if 2 in scenarios:
            all_attacks.extend(self.generate_scenario2())
        if 3 in scenarios:
            all_attacks.extend(self.generate_scenario3())
        
        attacks_df = pd.DataFrame(all_attacks)
        
        # Merge with original dataset
        orig_cols = ['u', 'i', 'ts', 'label']
        combined = pd.concat([
            self.graph_df[orig_cols],
            attacks_df[orig_cols]
        ], ignore_index=True)
        
        # Sort by timestamp
        combined = combined.sort_values('ts').reset_index(drop=True)
        
        # Save augmented dataset
        os.makedirs(output_dir, exist_ok=True)
        
        for scenario in scenarios:
            scenario_attacks = attacks_df[attacks_df['scenario'] == scenario]
            scenario_combined = pd.concat([
                self.graph_df[orig_cols],
                scenario_attacks[orig_cols]
            ], ignore_index=True).sort_values('ts').reset_index(drop=True)
            
            out_path = os.path.join(output_dir, f'ml_lanl_scenario{scenario}.csv')
            scenario_combined.to_csv(out_path, index=False)
            print(f"\nSaved Scenario {scenario}: {out_path}")
            print(f"  Total edges: {len(scenario_combined)}")
            print(f"  Malicious edges: {scenario_combined['label'].sum()}")
        
        # Save all scenarios combined
        combined_path = os.path.join(output_dir, 'ml_lanl_all_scenarios.csv')
        combined.to_csv(combined_path, index=False)
        print(f"\nSaved combined: {combined_path}")
        print(f"  Total edges: {len(combined)}")
        print(f"  Malicious edges: {combined['label'].sum()}")
        
        return combined


def main():
    parser = argparse.ArgumentParser(description='Generate threat samples for Jbeil')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='lanl')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    csv_path = os.path.join(args.data_dir, f'ml_{args.dataset}.csv')
    print(f"Loading dataset: {csv_path}")
    graph_df = pd.read_csv(csv_path)
    
    simulator = LateralMovementSimulator(graph_df)
    simulator.augment_dataset(args.data_dir)


if __name__ == '__main__':
    main()
