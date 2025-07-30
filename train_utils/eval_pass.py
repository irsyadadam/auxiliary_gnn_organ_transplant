"""
eval_pass.py - Updated evaluation functions with proper train/val/test splits
"""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# FIXED IMPORT PATH
from utils.data_splitting import create_data_splits, get_split_data

def setup_eval_logging(args):
    """Setup logging for evaluation"""
    os.makedirs(args.results_dir, exist_ok=True)
    
    log_file = os.path.join(args.results_dir, 'evaluation.log')
    
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if args.verbose else logging.NullHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger

def evaluate_link_prediction_split(model, graph_data, pos_edges, neg_edges, args):
    """
    Evaluate link prediction performance on a specific split
    
    Args:
        model: Trained model
        graph_data: Full graph data
        pos_edges: Positive edges for this split
        neg_edges: Negative edges for this split
        args: Evaluation arguments
    
    Returns:
        dict: Link prediction metrics
    """
    model.eval()
    device = args.device
    
    if pos_edges.shape[1] == 0:
        return {'auc': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    with torch.no_grad():
        # Move edges to device
        pos_edges = pos_edges.to(device)
        neg_edges = neg_edges.to(device)
        
        # Get embeddings
        node_embeddings = model.gnn_encoder(graph_data.x, graph_data.edge_index, graph_data.edge_type)
        
        # FIXED: Handle decoder interface consistently
        edge_type_zeros_pos = torch.zeros(pos_edges.shape[1], device=device, dtype=torch.long)
        edge_type_zeros_neg = torch.zeros(neg_edges.shape[1], device=device, dtype=torch.long)
        
        # Predict links for positive edges
        decoder_output_pos = model.decoder(node_embeddings, pos_edges, edge_type_zeros_pos)
        if isinstance(decoder_output_pos, tuple):
            pos_scores = decoder_output_pos[0]
        else:
            pos_scores = decoder_output_pos
            
        # Predict links for negative edges
        decoder_output_neg = model.decoder(node_embeddings, neg_edges, edge_type_zeros_neg)
        if isinstance(decoder_output_neg, tuple):
            neg_scores = decoder_output_neg[0]
        else:
            neg_scores = decoder_output_neg
        
        # Combine scores and labels
        all_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        all_labels = torch.cat([
            torch.ones(pos_scores.shape[0]), 
            torch.zeros(neg_scores.shape[0])
        ]).cpu().numpy()
        
        # Compute metrics
        probabilities = torch.sigmoid(torch.tensor(all_scores)).numpy()
        predictions = (probabilities > 0.5).astype(int)
        
        # Handle edge case where all predictions are the same
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, predictions, average='binary', zero_division=0
        )
        
        link_metrics = {
            'auc': roc_auc_score(all_labels, probabilities) if len(np.unique(all_labels)) > 1 else 0.0,
            'accuracy': accuracy_score(all_labels, predictions),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_positive': int(np.sum(all_labels)),
            'num_negative': int(len(all_labels) - np.sum(all_labels)),
            'predictions': probabilities,
            'labels': all_labels
        }
    
    return link_metrics

def evaluate_outcome_prediction_split(model, graph_data, outcome_pairs, args):
    """
    Evaluate outcome prediction performance on a specific split
    
    Args:
        model: Trained model
        graph_data: Full graph data
        outcome_pairs: Dict of (donor_node, recipient_node) -> label for this split
        args: Evaluation arguments
    
    Returns:
        dict: Outcome prediction metrics
    """
    model.eval()
    device = args.device
    
    if not outcome_pairs:
        return {'auc': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'total_cases': 0}
    
    with torch.no_grad():
        # Get node embeddings
        node_embeddings = model.gnn_encoder(graph_data.x, graph_data.edge_index, graph_data.edge_type)
        
        # Prepare data
        donor_nodes = []
        recipient_nodes = []
        labels = []
        
        for (donor_id, recipient_id), label in outcome_pairs.items():
            donor_nodes.append(donor_id)
            recipient_nodes.append(recipient_id)
            labels.append(label)
        
        donor_nodes = torch.tensor(donor_nodes, device=device)
        recipient_nodes = torch.tensor(recipient_nodes, device=device)
        labels = torch.tensor(labels, dtype=torch.float32, device=device)
        
        # Get embeddings and predict - FIXED: Use correct classifier interface
        donor_embeds = node_embeddings[donor_nodes]
        recipient_embeds = node_embeddings[recipient_nodes]
        predictions = model.classifier(donor_embeds, recipient_embeds)
        
        # Convert to numpy
        probabilities = torch.sigmoid(predictions.squeeze()).cpu().numpy()
        labels_np = labels.cpu().numpy()
        predictions_binary = (probabilities > 0.5).astype(int)
        
        # Handle edge case where all predictions are the same
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, predictions_binary, average='binary', zero_division=0
        )
        
        # Compute metrics
        outcome_metrics = {
            'auc': roc_auc_score(labels_np, probabilities) if len(np.unique(labels_np)) > 1 else 0.0,
            'accuracy': accuracy_score(labels_np, predictions_binary),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_positive': int(np.sum(labels_np)),
            'num_negative': int(len(labels_np) - np.sum(labels_np)),
            'total_cases': len(labels_np),
            'predictions': probabilities,
            'labels': labels_np
        }
        
        # Confusion matrix
        cm = confusion_matrix(labels_np, predictions_binary)
        outcome_metrics['confusion_matrix'] = cm.tolist()