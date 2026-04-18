"""
Jbeil Pre-processing Pipeline for LANL Authentication Logs.

This script implements the 4-step pre-processing pipeline described in Section 3.1:
  Step 1: Parse and format authentication logs (timestamp, src, usr, dst)
  Step 2: Extract graph maps (in-degree and out-degree dictionaries)
  Step 3: Calculate graph features from graph maps
  Step 4: Produce temporal graph representation (Time, Src, Dst, Features, Label)

Usage:
    python utils/preprocess_lanl.py --data_dir ./lanl_data --output_dir ./data
"""

import argparse
import csv
import json
import os
import numpy as np
from collections import defaultdict
from pathlib import Path


def parse_auth_logs(auth_file, redteam_file=None):
    """
    Step 1: Parse LANL authentication logs.
    
    LANL auth.txt format: time,src_user@src_domain,dst_user@dst_domain,src_computer,dst_computer,auth_type,logon_type,auth_orient,success/failure
    LANL redteam.txt format: time,src_user@src_domain,src_computer,dst_computer
    
    Returns:
        events: list of (timestamp, src_host, user, dst_host, label)
    """
    # Load red team events for labeling
    redteam_set = set()
    if redteam_file and os.path.exists(redteam_file):
        print("Loading red team events...")
        with open(redteam_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    t, user, src, dst = parts[0], parts[1], parts[2], parts[3]
                    redteam_set.add((t, user, src, dst))
        print(f"  Loaded {len(redteam_set)} red team events.")

    print("Parsing authentication logs...")
    events = []
    node_set = set()
    
    with open(auth_file, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
            
            timestamp = int(parts[0])
            src_user = parts[1].split('@')[0] if '@' in parts[1] else parts[1]
            dst_user = parts[2].split('@')[0] if '@' in parts[2] else parts[2]
            src_host = parts[3]
            dst_host = parts[4]
            success = parts[8]
            
            # Use dst_user as the "user" for graph feature extraction
            user = dst_user
            
            # Label: 1 if this event matches a red team event, else 0
            label = 0
            if (parts[0], parts[1], src_host, dst_host) in redteam_set:
                label = 1
            
            events.append((timestamp, src_host, user, dst_host, label))
            node_set.add(src_host)
            node_set.add(dst_host)
            
            if (i + 1) % 5_000_000 == 0:
                print(f"  Processed {i+1} lines...")
    
    print(f"  Total events: {len(events)}")
    print(f"  Unique nodes: {len(node_set)}")
    print(f"  Malicious events: {sum(1 for e in events if e[4] == 1)}")
    
    return events, node_set


def extract_graph_maps(events):
    """
    Step 2: Extract graph maps from authentication logs.
    
    Implements Algorithm 1 from the paper.
    Produces 6 dictionaries:
      In-degree maps:  InUsrMap, InSrcMap, InUsrSrcMap
      Out-degree maps: OutUsrMap, OutDstMap, OutUsrDstMap
    """
    print("Extracting graph maps...")
    
    # In-degree maps: keyed by destination host
    InUsrMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    InSrcMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    InUsrSrcMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # Out-degree maps: keyed by source host
    OutUsrMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    OutDstMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    OutUsrDstMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for ts, src, usr, dst, label in events:
        day = ts // 86400  # Convert epoch to day
        
        # In-degree maps (destination-centric)
        InUsrMap[dst][usr][day] += 1
        InSrcMap[dst][src][day] += 1
        InUsrSrcMap[dst][(usr, src)][day] += 1
        
        # Out-degree maps (source-centric)
        OutUsrMap[src][usr][day] += 1
        OutDstMap[src][dst][day] += 1
        OutUsrDstMap[src][(usr, dst)][day] += 1
    
    print("  Graph maps extracted.")
    return InUsrMap, InSrcMap, InUsrSrcMap, OutUsrMap, OutDstMap, OutUsrDstMap


def calculate_graph_features(events, InUsrMap, InSrcMap, InUsrSrcMap,
                              OutUsrMap, OutDstMap, OutUsrDstMap):
    """
    Step 3: Calculate graph features using graph maps.
    
    For each event, calculate 9 features:
      In-degree features (for dst):
        1. In_Unique_Usr:    # unique users targeting dst
        2. In_Unique_Src:    # unique src hosts targeting dst
        3. In_Unique_UsrSrc: # unique (usr, src) combinations targeting dst
      Out-degree features (for src):
        4. Out_Unique_Usr:    # unique users from src
        5. Out_Unique_Dst:    # unique dst hosts from src
        6. Out_Unique_UsrDst: # unique (usr, dst) combinations from src
        7. Out_Day_Avg_Usr:    avg daily user interactions from src
        8. Out_Day_Avg_Dst:    avg daily dst interactions from src
        9. Out_Day_Avg_UsrDst: avg daily (usr, dst) interactions from src
    """
    print("Calculating graph features...")
    
    features_list = []
    
    for idx, (ts, src, usr, dst, label) in enumerate(events):
        # In-degree features for destination
        in_unique_usr = len(InUsrMap[dst])
        in_unique_src = len(InSrcMap[dst])
        in_unique_usr_src = len(InUsrSrcMap[dst])
        
        # Out-degree features for source
        out_unique_usr = len(OutUsrMap[src])
        out_unique_dst = len(OutDstMap[src])
        out_unique_usr_dst = len(OutUsrDstMap[src])
        
        # Daily averages for out-degrees
        # Average daily interactions across all users from src
        usr_days = set()
        usr_total = 0
        for u in OutUsrMap[src]:
            for d in OutUsrMap[src][u]:
                usr_days.add(d)
                usr_total += OutUsrMap[src][u][d]
        out_day_avg_usr = usr_total / max(len(usr_days), 1)
        
        dst_days = set()
        dst_total = 0
        for d_host in OutDstMap[src]:
            for d in OutDstMap[src][d_host]:
                dst_days.add(d)
                dst_total += OutDstMap[src][d_host][d]
        out_day_avg_dst = dst_total / max(len(dst_days), 1)
        
        usr_dst_days = set()
        usr_dst_total = 0
        for ud in OutUsrDstMap[src]:
            for d in OutUsrDstMap[src][ud]:
                usr_dst_days.add(d)
                usr_dst_total += OutUsrDstMap[src][ud][d]
        out_day_avg_usr_dst = usr_dst_total / max(len(usr_dst_days), 1)
        
        features = [
            in_unique_usr, in_unique_src, in_unique_usr_src,
            out_unique_usr, out_unique_dst, out_unique_usr_dst,
            out_day_avg_usr, out_day_avg_dst, out_day_avg_usr_dst
        ]
        features_list.append(features)
        
        if (idx + 1) % 5_000_000 == 0:
            print(f"  Processed features for {idx+1} events...")
    
    print("  Graph features calculated.")
    return features_list


def build_temporal_graph(events, features_list, output_dir):
    """
    Step 4: Produce final temporal graph representation.
    
    Output format: CSV with columns:
        timestamp, src_id, dst_id, label, feature_1, ..., feature_9
    
    Also produces:
        - Node ID mapping
        - Edge features as .npy
    """
    print("Building temporal graph representation...")
    
    # Build node ID mapping
    node_to_id = {}
    current_id = 0
    for ts, src, usr, dst, label in events:
        if src not in node_to_id:
            node_to_id[src] = current_id
            current_id += 1
        if dst not in node_to_id:
            node_to_id[dst] = current_id
            current_id += 1
    
    num_nodes = len(node_to_id)
    print(f"  Total unique nodes: {num_nodes}")
    
    # Write processed data
    os.makedirs(output_dir, exist_ok=True)
    
    # Save node mapping
    with open(os.path.join(output_dir, 'node_mapping.json'), 'w') as f:
        json.dump(node_to_id, f)
    
    # Save in TGN-compatible format: u, i, ts, label, idx
    # Plus edge features as separate .npy file
    sources = []
    destinations = []
    timestamps = []
    labels = []
    edge_features = []
    
    for idx, (ts, src, usr, dst, label) in enumerate(events):
        sources.append(node_to_id[src])
        destinations.append(node_to_id[dst])
        timestamps.append(ts)
        labels.append(label)
        edge_features.append(features_list[idx])
    
    # Write CSV (TGN format)
    csv_path = os.path.join(output_dir, 'lanl.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['u', 'i', 'ts', 'label'] + [f'f{j}' for j in range(9)])
        for idx in range(len(sources)):
            writer.writerow([
                sources[idx], destinations[idx], timestamps[idx], labels[idx]
            ] + features_list[idx])
    
    # Save edge features as npy (TGN format)
    edge_features_np = np.array(edge_features, dtype=np.float32)
    np.save(os.path.join(output_dir, 'lanl_edge_features.npy'), edge_features_np)
    
    # Save empty node features (initialized to zero, as per paper)
    node_features = np.zeros((num_nodes + 1, 9), dtype=np.float32)
    np.save(os.path.join(output_dir, 'lanl_node_features.npy'), node_features)
    
    print(f"  Saved {len(sources)} edges to {csv_path}")
    print(f"  Edge features shape: {edge_features_np.shape}")
    print(f"  Node features shape: {node_features.shape}")
    
    return num_nodes


def preprocess_data_tgn_format(data_dir, output_dir):
    """
    Convert the preprocessed CSV into the format expected by the TGN-based
    training script (matching utils/data_processing.py expectations).
    
    Produces:
        ml_lanl.csv          - sorted edges
        ml_lanl.npy          - edge features
        ml_lanl_node.npy     - node features
    """
    print("\nConverting to TGN-compatible format...")
    
    csv_path = os.path.join(output_dir, 'lanl.csv')
    
    # Read and sort by timestamp
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Re-index nodes to be 1-based (TGN convention)
    all_nodes = set(df['u'].values) | set(df['i'].values)
    node_map = {old: new + 1 for new, old in enumerate(sorted(all_nodes))}
    df['u'] = df['u'].map(node_map)
    df['i'] = df['i'].map(node_map)
    
    # Save CSV
    out_csv = os.path.join(output_dir, 'ml_lanl.csv')
    df[['u', 'i', 'ts', 'label']].to_csv(out_csv, index=False)
    
    # Save features
    feat_cols = [c for c in df.columns if c.startswith('f')]
    edge_feats = df[feat_cols].values.astype(np.float32)
    
    # Normalize features
    mean = edge_feats.mean(axis=0, keepdims=True)
    std = edge_feats.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    edge_feats = (edge_feats - mean) / std
    
    np.save(os.path.join(output_dir, 'ml_lanl.npy'), edge_feats)
    
    num_nodes = max(node_map.values())
    node_feats = np.zeros((num_nodes + 1, edge_feats.shape[1]), dtype=np.float32)
    np.save(os.path.join(output_dir, 'ml_lanl_node.npy'), node_feats)
    
    print(f"  Saved ml_lanl.csv with {len(df)} edges")
    print(f"  Saved ml_lanl.npy with shape {edge_feats.shape}")
    print(f"  Saved ml_lanl_node.npy with shape {node_feats.shape}")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  Malicious edges: {df['label'].sum()}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess LANL data for Jbeil')
    parser.add_argument('--data_dir', type=str, default='./lanl_data',
                        help='Directory containing auth.txt.gz and redteam.txt.gz')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for processed data')
    args = parser.parse_args()
    
    auth_file = os.path.join(args.data_dir, 'auth.txt')
    redteam_file = os.path.join(args.data_dir, 'redteam.txt')
    
    # Check for gzipped versions
    import gzip
    if not os.path.exists(auth_file) and os.path.exists(auth_file + '.gz'):
        print("Decompressing auth.txt.gz...")
        with gzip.open(auth_file + '.gz', 'rb') as f_in:
            with open(auth_file, 'wb') as f_out:
                for chunk in iter(lambda: f_in.read(1024*1024), b''):
                    f_out.write(chunk)
    
    if not os.path.exists(redteam_file) and os.path.exists(redteam_file + '.gz'):
        print("Decompressing redteam.txt.gz...")
        with gzip.open(redteam_file + '.gz', 'rb') as f_in:
            with open(redteam_file, 'wb') as f_out:
                f_out.write(f_in.read())
    
    # Step 1: Parse authentication logs
    events, node_set = parse_auth_logs(auth_file, redteam_file)
    
    # Step 2: Extract graph maps
    maps = extract_graph_maps(events)
    
    # Step 3: Calculate graph features
    features_list = calculate_graph_features(events, *maps)
    
    # Step 4: Build temporal graph representation
    num_nodes = build_temporal_graph(events, features_list, args.output_dir)
    
    # Convert to TGN format
    preprocess_data_tgn_format(args.data_dir, args.output_dir)
    
    print("\nPreprocessing complete!")


if __name__ == '__main__':
    main()
