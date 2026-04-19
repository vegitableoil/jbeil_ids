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


class AuthLogStream:
    """
    An iterable that streams the massive auth.txt file line-by-line.
    This prevents OOM errors by never loading the full dataset into RAM.
    """
    def __init__(self, auth_file, redteam_set):
        self.auth_file = auth_file
        self.redteam_set = redteam_set

    def __iter__(self):
        with open(self.auth_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    continue
                
                timestamp = int(parts[0])
                src_user = parts[1].split('@')[0] if '@' in parts[1] else parts[1]
                dst_user = parts[2].split('@')[0] if '@' in parts[2] else parts[2]
                src_host = parts[3]
                dst_host = parts[4]
                
                user = dst_user
                label = 1 if (parts[0], parts[1], src_host, dst_host) in self.redteam_set else 0
                
                yield (timestamp, src_host, user, dst_host, label)


def setup_event_stream(auth_file, redteam_file=None):
    """
    Step 1: Setup LANL authentication logs streaming.
    """
    redteam_set = set()
    if redteam_file and os.path.exists(redteam_file):
        print("Loading red team events...")
        with open(redteam_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    redteam_set.add((parts[0], parts[1], parts[2], parts[3]))
        print(f"  Loaded {len(redteam_set)} red team events.")
        
    return AuthLogStream(auth_file, redteam_set)


def extract_graph_maps(events_stream):
    """
    Step 2: Extract graph maps from authentication logs.
    """
    print("Extracting graph maps (Pass 1)...")
    
    InUsrMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    InSrcMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    InUsrSrcMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    OutUsrMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    OutDstMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    OutUsrDstMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    node_set = set()
    total_events = 0
    
    for ts, src, usr, dst, label in events_stream:
        day = ts // 86400  # Convert epoch to day
        
        # In-degree maps (destination-centric)
        InUsrMap[dst][usr][day] += 1
        InSrcMap[dst][src][day] += 1
        InUsrSrcMap[dst][(usr, src)][day] += 1
        
        # Out-degree maps (source-centric)
        OutUsrMap[src][usr][day] += 1
        OutDstMap[src][dst][day] += 1
        OutUsrDstMap[src][(usr, dst)][day] += 1
        
        node_set.add(src)
        node_set.add(dst)
        total_events += 1
        
        if total_events % 5_000_000 == 0:
            print(f"  Processed {total_events} lines...")
            
    print("  Graph maps extracted.")
    return InUsrMap, InSrcMap, InUsrSrcMap, OutUsrMap, OutDstMap, OutUsrDstMap, node_set, total_events


def calculate_graph_features(events_stream, InUsrMap, InSrcMap, InUsrSrcMap,
                              OutUsrMap, OutDstMap, OutUsrDstMap):
    """
    Step 3: Calculate graph features using graph maps iteratively.
    Yields tuple of (event_data, features).
    """
    print("Calculating graph features...")
    for ts, src, usr, dst, label in events_stream:
        # In-degree features
        in_unique_usr = len(InUsrMap[dst])
        in_unique_src = len(InSrcMap[dst])
        in_unique_usr_src = len(InUsrSrcMap[dst])
        
        # Out-degree features
        out_unique_usr = len(OutUsrMap[src])
        out_unique_dst = len(OutDstMap[src])
        out_unique_usr_dst = len(OutUsrDstMap[src])
        
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
        
        yield (ts, src, usr, dst, label), features


def build_temporal_graph(features_iter, node_set, total_events, output_dir):
    """
    Step 4: Stream representations directly into TGN-compatible outputs.
    Avoids using pandas on huge datasets to prevent memory overload.
    """
    print("Building temporal graph representation and formatting...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1-based indexing applied directly for TGN formatting
    node_to_id = {node: i + 1 for i, node in enumerate(sorted(node_set))}
    num_nodes = len(node_to_id)
    print(f"  Total unique nodes: {num_nodes}")
    
    with open(os.path.join(output_dir, 'node_mapping.json'), 'w') as f:
        json.dump(node_to_id, f)
        
    csv_path = os.path.join(output_dir, 'ml_lanl.csv')
    
    # Pre-allocate numpy array to avoid Python list overhead
    edge_feats = np.zeros((total_events, 9), dtype=np.float32)
    malicious_count = 0
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['u', 'i', 'ts', 'label'])
        
        for idx, ((ts, src, usr, dst, label), feats) in enumerate(features_iter):
            u_id = node_to_id[src]
            i_id = node_to_id[dst]
            
            writer.writerow([u_id, i_id, ts, label])
            edge_feats[idx] = feats
            
            if label == 1:
                malicious_count += 1
            
            if (idx + 1) % 5_000_000 == 0:
                print(f"  Processed features & saved {idx+1} edges...")

    # Normalize features on the pre-allocated array (replacing Pandas normalize)
    print("Normalizing features...")
    mean = edge_feats.mean(axis=0, keepdims=True)
    std = edge_feats.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    edge_feats = (edge_feats - mean) / std
    
    np.save(os.path.join(output_dir, 'ml_lanl.npy'), edge_feats)
    
    node_feats = np.zeros((num_nodes + 1, 9), dtype=np.float32)
    np.save(os.path.join(output_dir, 'ml_lanl_node.npy'), node_feats)
    
    print(f"  Saved ml_lanl.csv with {total_events} edges")
    print(f"  Saved ml_lanl.npy with shape {edge_feats.shape}")
    print(f"  Saved ml_lanl_node.npy with shape {node_feats.shape}")
    print(f"  Malicious edges: {malicious_count}")
    
    return num_nodes


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
    
    # Step 1: Initialize iterator (avoids list materialization)
    events_stream = setup_event_stream(auth_file, redteam_file)
    
    # Step 2: Extract graph maps (Pass 1)
    *maps, node_set, total_events = extract_graph_maps(events_stream)
    
    # Step 3: Calculate graph features (Pass 2 setup)
    features_iter = calculate_graph_features(events_stream, *maps)
    
    # Step 4: Build temporal graph representation (Pass 2 execution)
    build_temporal_graph(features_iter, node_set, total_events, args.output_dir)
    
    print("\nPreprocessing complete!")


if __name__ == '__main__':
    main()