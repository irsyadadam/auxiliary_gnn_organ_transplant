"""
eval_pass.py - Complete evaluation functions with proper train/val/test splits
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
    
    return outcome_metrics

def evaluate_model(model, graph_data, transplant_edges, pair_to_label, donor_mapping, recipient_mapping, args):
    """
    Comprehensive model evaluation on all splits
    
    Args:
        model: Trained model
        graph_data: Full graph data
        transplant_edges: All transplant edges
        pair_to_label: All outcome labels
        donor_mapping: Donor node mapping
        recipient_mapping: Recipient node mapping
        args: Evaluation arguments
    
    Returns:
        dict: Comprehensive evaluation results
    """
    
    logger = setup_eval_logging(args)
    logger.info("Starting comprehensive model evaluation")
    
    # Ensure model is on correct device
    model = model.to(args.device)
    graph_data = graph_data.to(args.device)
    
    # Create data splits
    splits = create_data_splits(pair_to_label, transplant_edges, graph_data.x.shape[0], args)
    
    results = {}
    
    for split_name in ['train', 'val', 'test']:
        logger.info(f"Evaluating {split_name} split...")
        
        # Get split data
        outcome_pairs, pos_edges, neg_edges = get_split_data(splits, split_name)
        
        # Evaluate outcome prediction
        outcome_metrics = evaluate_outcome_prediction_split(
            model, graph_data, outcome_pairs, args
        )
        
        # For ranking-based approach, use ranking evaluation if we have positive edges
        link_metrics = {}
        if args.use_link_prediction and pos_edges.shape[1] > 0:
            # Import ranking evaluation function
            from train_utils.train_pass import evaluate_ranking_link_prediction
            
            link_metrics = evaluate_ranking_link_prediction(
                model, graph_data, pos_edges, transplant_edges, 
                graph_data.x.shape[0], args
            )
        
        # Store results
        results[split_name] = {
            'outcome_prediction': outcome_metrics,
            'link_prediction': link_metrics if link_metrics else None
        }
        
        # Log key metrics
        logger.info(f"{split_name.upper()} Results:")
        logger.info(f"  Outcome AUC: {outcome_metrics['auc']:.4f}")
        logger.info(f"  Outcome Accuracy: {outcome_metrics['accuracy']:.4f}")
        logger.info(f"  Outcome F1: {outcome_metrics['f1']:.4f}")
        
        if link_metrics:
            logger.info(f"  Link MRR: {link_metrics.get('mrr', 0):.4f}")
            logger.info(f"  Link Hits@10: {link_metrics.get('hits@10', 0):.4f}")
    
    # Save results to file
    results_file = os.path.join(args.results_dir, 'evaluation_results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for split_name, split_results in results.items():
        json_results[split_name] = {}
        for task_name, task_results in split_results.items():
            if task_results is not None:
                json_results[split_name][task_name] = {}
                for metric_name, metric_value in task_results.items():
                    if isinstance(metric_value, np.ndarray):
                        json_results[split_name][task_name][metric_name] = metric_value.tolist()
                    else:
                        json_results[split_name][task_name][metric_name] = metric_value
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_file}")
    
    # Create evaluation plots
    create_evaluation_plots(results, args)
    
    return results

def create_evaluation_plots(results, args):
    """Create evaluation plots and save them"""
    
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Evaluation Results', fontsize=16)
    
    # Plot 1: Outcome AUC across splits
    splits = ['train', 'val', 'test']
    outcome_aucs = [results[split]['outcome_prediction']['auc'] for split in splits]
    
    axes[0, 0].bar(splits, outcome_aucs, color=['blue', 'orange', 'green'])
    axes[0, 0].set_title('Outcome Prediction AUC')
    axes[0, 0].set_ylabel('AUC Score')
    axes[0, 0].set_ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(outcome_aucs):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Plot 2: Outcome metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    test_metrics = [results['test']['outcome_prediction'][metric] for metric in metrics]
    
    axes[0, 1].bar(metrics, test_metrics, color='green', alpha=0.7)
    axes[0, 1].set_title('Test Set Outcome Metrics')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(test_metrics):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Plot 3: Link prediction metrics (if available)
    if results['test']['link_prediction'] is not None:
        link_metrics = results['test']['link_prediction']
        link_metric_names = ['mrr', 'hits@1', 'hits@10', 'hits@100']
        link_values = [link_metrics.get(metric, 0) for metric in link_metric_names]
        
        axes[1, 0].bar(link_metric_names, link_values, color='red', alpha=0.7)
        axes[1, 0].set_title('Test Set Link Prediction Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(link_values):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    else:
        axes[1, 0].text(0.5, 0.5, 'Link Prediction\nNot Enabled', 
                       ha='center', va='center', transform=axes[1, 0].transAxes,
                       fontsize=12)
        axes[1, 0].set_title('Link Prediction Metrics')
    
    # Plot 4: Confusion matrix for test set
    if 'confusion_matrix' in results['test']['outcome_prediction']:
        cm = np.array(results['test']['outcome_prediction']['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Test Set Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('True')
    else:
        axes[1, 1].text(0.5, 0.5, 'Confusion Matrix\nNot Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12)
        axes[1, 1].set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(args.results_dir, 'evaluation_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation plots saved to {plot_file}")

def print_evaluation_summary(results, args):
    """Print a comprehensive evaluation summary"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*80)
    
    for split_name in ['train', 'val', 'test']:
        split_results = results[split_name]
        outcome_results = split_results['outcome_prediction']
        link_results = split_results['link_prediction']
        
        print(f"\n{split_name.upper()} SET RESULTS:")
        print("-" * 40)
        
        # Outcome prediction results
        print(f"Outcome Prediction:")
        print(f"  • AUC:          {outcome_results['auc']:.4f}")
        print(f"  • Accuracy:     {outcome_results['accuracy']:.4f}")
        print(f"  • Precision:    {outcome_results['precision']:.4f}")
        print(f"  • Recall:       {outcome_results['recall']:.4f}")
        print(f"  • F1-Score:     {outcome_results['f1']:.4f}")
        print(f"  • Total Cases:  {outcome_results['total_cases']}")
        print(f"  • Positive:     {outcome_results['num_positive']}")
        print(f"  • Negative:     {outcome_results['num_negative']}")
        
        # Link prediction results
        if link_results is not None and args.use_link_prediction:
            print(f"\nLink Prediction (Ranking):")
            print(f"  • MRR:          {link_results.get('mrr', 0):.4f}")
            print(f"  • Hits@1:       {link_results.get('hits@1', 0):.4f}")
            print(f"  • Hits@10:      {link_results.get('hits@10', 0):.4f}")
            print(f"  • Hits@100:     {link_results.get('hits@100', 0):.4f}")
            print(f"  • Mean Rank:    {link_results.get('mean_rank', 0):.2f}")
            print(f"  • Test Edges:   {link_results.get('num_test_edges', 0)}")
        else:
            print(f"\nLink Prediction: Not enabled or no data")
    
    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Multi-task Learning:    {args.use_link_prediction}")
    print(f"Alpha (outcome weight): {args.alpha}")
    print(f"GNN Type:              {args.gnn_type}")
    print(f"Decoder Type:          {args.decoder_type}")
    print(f"Hidden Dimension:      {args.hidden_dim}")
    print(f"Device:                {args.device}")
    
    if args.use_link_prediction:
        print(f"Link Prediction Type:  Ranking-based (MRR, Hits@K)")
    else:
        print(f"Link Prediction Type:  Disabled")
    
    print("="*80)

def create_classification_report(results, args):
    """Create detailed classification report"""
    
    report_data = {}
    
    for split_name in ['train', 'val', 'test']:
        outcome_results = results[split_name]['outcome_prediction']
        
        if 'labels' in outcome_results and 'predictions' in outcome_results:
            labels = np.array(outcome_results['labels'])
            predictions = (np.array(outcome_results['predictions']) > 0.5).astype(int)
            
            # Generate classification report
            report = classification_report(
                labels, predictions, 
                target_names=['No Complication', 'Complication'],
                output_dict=True
            )
            
            report_data[split_name] = report
    
    # Save classification report
    report_file = os.path.join(args.results_dir, 'classification_report.json')
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    return report_data

def evaluate_feature_importance(model, graph_data, args):
    """
    Evaluate feature importance using gradient-based methods
    This is a placeholder for future implementation
    """
    print("Feature importance evaluation not implemented yet")
    return None

def evaluate_model_robustness(model, graph_data, args):
    """
    Evaluate model robustness with different perturbations
    This is a placeholder for future implementation
    """
    print("Model robustness evaluation not implemented yet")
    return None