"""
Evaluation utilities for Jbeil.

Implements the evaluation metrics described in Section 4.3:
  - AUC score
  - Average Precision (AP) score
  - Precision and Recall with optimal G-mean threshold
  - ROC-based optimal threshold for imbalanced data
"""

import math
import torch
import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score,
    roc_curve
)


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors,
                          batch_size=200):
    """
    Evaluate model on edge prediction task.
    
    Returns:
        ap: Average Precision score
        auc: Area Under the ROC Curve score
    """
    # Ensure evaluation mode
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()
    
    val_ap, val_auc = [], []
    
    with torch.no_grad():
        model = model.eval()
        
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]
            
            size = len(sources_batch)
            _, negatives_batch = negative_edge_sampler.sample(size)
            
            pos_prob, neg_prob = model.compute_edge_probabilities(
                sources_batch, destinations_batch, negatives_batch,
                timestamps_batch, edge_idxs_batch, n_neighbors
            )
            
            pred_score = np.concatenate([
                pos_prob.cpu().numpy(), neg_prob.cpu().numpy()
            ])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
    
    return np.mean(val_ap), np.mean(val_auc)


def eval_lm_detection(model, negative_edge_sampler, data, n_neighbors,
                       batch_size=200):
    """
    Comprehensive LM detection evaluation with all metrics from Section 4.3.
    
    Uses the optimal G-mean threshold from the ROC curve to handle
    the imbalanced nature of the LANL dataset.
    
    Returns:
        dict with: auc, ap, precision, recall, threshold
    """
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()
    
    all_pred_scores = []
    all_true_labels = []
    
    with torch.no_grad():
        model = model.eval()
        
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]
            
            size = len(sources_batch)
            _, negatives_batch = negative_edge_sampler.sample(size)
            
            pos_prob, neg_prob = model.compute_edge_probabilities(
                sources_batch, destinations_batch, negatives_batch,
                timestamps_batch, edge_idxs_batch, n_neighbors
            )
            
            pred_score = np.concatenate([
                pos_prob.cpu().numpy(), neg_prob.cpu().numpy()
            ])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
            all_pred_scores.extend(pred_score)
            all_true_labels.extend(true_label)
    
    all_pred_scores = np.array(all_pred_scores)
    all_true_labels = np.array(all_true_labels)
    
    # AUC
    auc = roc_auc_score(all_true_labels, all_pred_scores)
    
    # AP
    ap = average_precision_score(all_true_labels, all_pred_scores)
    
    # Optimal threshold using G-mean (geometric mean of sensitivity and specificity)
    fpr, tpr, thresholds = roc_curve(all_true_labels, all_pred_scores)
    gmean = np.sqrt(tpr * (1 - fpr))
    optimal_idx = np.argmax(gmean)
    optimal_threshold = thresholds[optimal_idx]
    
    # Precision and Recall at optimal threshold
    predictions = (all_pred_scores >= optimal_threshold).astype(int)
    precision = precision_score(all_true_labels, predictions, zero_division=0)
    recall = recall_score(all_true_labels, predictions, zero_division=0)
    
    return {
        'auc': auc,
        'ap': ap,
        'precision': precision,
        'recall': recall,
        'threshold': optimal_threshold,
        'gmean': gmean[optimal_idx]
    }
