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

# Import our data splitting module
from data_splitting import create_data_splits, get_split_data

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
        node_embeddings = model.encoder(graph_data.x, graph_data.edge_index, graph_data.edge_type)
        
        # Predict links
        pos_scores = model.decoder(
            node_embeddings, 
            pos_edges, 
            edge_type=torch.zeros(pos_edges.shape[1], device=device)
        )
        neg_scores = model.decoder(
            node_embeddings, 
            neg_edges, 
            edge_type=torch.zeros(neg_edges.shape[1], device=device)
        )
        
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
        node_embeddings = model.encoder(graph_data.x, graph_data.edge_index, graph_data.edge_type)
        
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
        
        # Get embeddings and predict
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

def evaluate_all_splits(model, graph_data, transplant_edges, pair_to_label, args):
    """
    Evaluate model on all splits (train/val/test)
    
    Args:
        model: Trained model
        graph_data: Full graph data
        transplant_edges: All transplant edges
        pair_to_label: All outcome pairs
        args: Evaluation arguments
    
    Returns:
        dict: Comprehensive evaluation results for all splits
    """
    # Create data splits
    splits = create_data_splits(pair_to_label, transplant_edges, graph_data.x.shape[0], args)
    
    # Move graph data to device
    graph_data = graph_data.to(args.device)
    
    results = {}
    
    for split_name in ['train', 'val', 'test']:
        print(f"\nEvaluating {split_name} split...")
        
        # Get split-specific data
        outcome_pairs, pos_edges, neg_edges = get_split_data(splits, split_name)
        
        # Evaluate outcome prediction
        outcome_metrics = evaluate_outcome_prediction_split(
            model, graph_data, outcome_pairs, args
        )
        
        # Evaluate link prediction if enabled
        link_metrics = {}
        if args.use_link_prediction:
            link_metrics = evaluate_link_prediction_split(
                model, graph_data, pos_edges, neg_edges, args
            )
        
        results[split_name] = {
            'outcome_prediction': outcome_metrics,
            'link_prediction': link_metrics if args.use_link_prediction else None
        }
        
        # Print summary for this split
        print(f"{split_name.capitalize()} Results:")
        print(f"  Outcome - AUC: {outcome_metrics['auc']:.4f}, Accuracy: {outcome_metrics['accuracy']:.4f}, F1: {outcome_metrics['f1']:.4f}")
        if args.use_link_prediction and link_metrics:
            print(f"  Link - AUC: {link_metrics['auc']:.4f}, Accuracy: {link_metrics['accuracy']:.4f}, F1: {link_metrics['f1']:.4f}")
    
    return results

def plot_evaluation_results_all_splits(results, args):
    """
    Create comprehensive evaluation plots for all splits
    
    Args:
        results: Results from evaluate_all_splits()
        args: Evaluation arguments
    
    Returns:
        str: Path to saved plot
    """
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    splits = ['train', 'val', 'test']
    colors = ['blue', 'orange', 'green']
    
    # 1. ROC Curves for Outcome Prediction
    for i, (split, color) in enumerate(zip(splits, colors)):
        outcome_metrics = results[split]['outcome_prediction']
        if 'predictions' in outcome_metrics and 'labels' in outcome_metrics:
            fpr, tpr, _ = roc_curve(outcome_metrics['labels'], outcome_metrics['predictions'])
            axes[0, 0].plot(fpr, tpr, label=f"{split.capitalize()} (AUC = {outcome_metrics['auc']:.3f})", 
                           color=color, alpha=0.8)
    
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curves - Outcome Prediction')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ROC Curves for Link Prediction
    if args.use_link_prediction:
        for i, (split, color) in enumerate(zip(splits, colors)):
            link_metrics = results[split]['link_prediction']
            if link_metrics and 'predictions' in link_metrics and 'labels' in link_metrics:
                fpr, tpr, _ = roc_curve(link_metrics['labels'], link_metrics['predictions'])
                axes[0, 1].plot(fpr, tpr, label=f"{split.capitalize()} (AUC = {link_metrics['auc']:.3f})", 
                               color=color, alpha=0.8)
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves - Link Prediction')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Link Prediction\nDisabled', ha='center', va='center', 
                       transform=axes[0, 1].transAxes, fontsize=14)
        axes[0, 1].set_title('Link Prediction (Disabled)')
    
    # 3. Precision-Recall Curves for Outcome Prediction
    for i, (split, color) in enumerate(zip(splits, colors)):
        outcome_metrics = results[split]['outcome_prediction']
        if 'predictions' in outcome_metrics and 'labels' in outcome_metrics:
            precision, recall, _ = precision_recall_curve(outcome_metrics['labels'], outcome_metrics['predictions'])
            axes[0, 2].plot(recall, precision, label=f"{split.capitalize()}", color=color, alpha=0.8)
    
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_title('Precision-Recall Curves - Outcome Prediction')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Test Set Confusion Matrix
    test_outcome = results['test']['outcome_prediction']
    if 'confusion_matrix' in test_outcome:
        cm = np.array(test_outcome['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Complication', 'Complication'],
                   yticklabels=['No Complication', 'Complication'],
                   ax=axes[1, 0])
        axes[1, 0].set_title('Test Set Confusion Matrix')
        axes[1, 0].set_ylabel('True Label')
        axes[1, 0].set_xlabel('Predicted Label')
    
    # 5. Performance Metrics Comparison - Outcome Prediction
    metrics_to_plot = ['auc', 'accuracy', 'f1', 'precision', 'recall']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for i, (split, color) in enumerate(zip(splits, colors)):
        outcome_metrics = results[split]['outcome_prediction']
        values = [outcome_metrics.get(metric, 0) for metric in metrics_to_plot]
        axes[1, 1].bar(x + i*width, values, width, label=split.capitalize(), 
                      color=color, alpha=0.8)
    
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Metrics - Outcome Prediction')
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels([m.upper() for m in metrics_to_plot], rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    # 6. Performance Metrics Comparison - Link Prediction
    if args.use_link_prediction:
        for i, (split, color) in enumerate(zip(splits, colors)):
            link_metrics = results[split]['link_prediction']
            if link_metrics:
                values = [link_metrics.get(metric, 0) for metric in metrics_to_plot]
                axes[1, 2].bar(x + i*width, values, width, label=split.capitalize(), 
                              color=color, alpha=0.8)
        
        axes[1, 2].set_xlabel('Metrics')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Performance Metrics - Link Prediction')
        axes[1, 2].set_xticks(x + width)
        axes[1, 2].set_xticklabels([m.upper() for m in metrics_to_plot], rotation=45)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim(0, 1)
    else:
        axes[1, 2].text(0.5, 0.5, 'Link Prediction\nDisabled', ha='center', va='center', 
                       transform=axes[1, 2].transAxes, fontsize=14)
        axes[1, 2].set_title('Link Prediction (Disabled)')
    
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'comprehensive_evaluation_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def save_comprehensive_results(results, args):
    """
    Save comprehensive evaluation results to files
    
    Args:
        results: Results from evaluate_all_splits()
        args: Evaluation arguments
    
    Returns:
        str: Path to main results file
    """
    # Prepare results for JSON serialization (remove numpy arrays)
    json_results = {}
    for split_name, split_results in results.items():
        json_results[split_name] = {}
        
        # Outcome prediction results
        outcome_results = split_results['outcome_prediction'].copy()
        outcome_results.pop('predictions', None)
        outcome_results.pop('labels', None)
        json_results[split_name]['outcome_prediction'] = outcome_results
        
        # Link prediction results
        if split_results['link_prediction']:
            link_results = split_results['link_prediction'].copy()
            link_results.pop('predictions', None)
            link_results.pop('labels', None)
            json_results[split_name]['link_prediction'] = link_results
        else:
            json_results[split_name]['link_prediction'] = None
    
    # Add metadata
    json_results['evaluation_args'] = vars(args)
    json_results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save main results
    results_path = os.path.join(args.results_dir, 'comprehensive_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save detailed classification reports for outcome prediction
    for split_name in ['train', 'val', 'test']:
        outcome_metrics = results[split_name]['outcome_prediction']
        if 'predictions' in outcome_metrics and 'labels' in outcome_metrics:
            predictions = (outcome_metrics['predictions'] > 0.5).astype(int)
            report = classification_report(
                outcome_metrics['labels'], 
                predictions, 
                target_names=['No Complication', 'Complication'],
                output_dict=True
            )
            
            report_path = os.path.join(args.results_dir, f'{split_name}_classification_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
    
    return results_path

def evaluate_model(model, graph_data, transplant_edges, pair_to_label, donor_mapping, recipient_mapping, args):
    """
    Main evaluation function with comprehensive train/val/test evaluation
    
    Args:
        model: Trained model
        graph_data: Full graph data
        transplant_edges: All transplant edges
        pair_to_label: All outcome pairs
        donor_mapping: Donor ID mappings
        recipient_mapping: Recipient ID mappings
        args: Evaluation arguments
    
    Returns:
        dict: Comprehensive evaluation results
    """
    
    # Setup logging
    logger = setup_eval_logging(args)
    
    logger.info("Starting comprehensive model evaluation")
    logger.info(f"Multi-task evaluation: {args.use_link_prediction}")
    
    start_time = time.time()
    
    # Evaluate on all splits
    results = evaluate_all_splits(model, graph_data, transplant_edges, pair_to_label, args)
    
    # Create plots
    logger.info("Creating comprehensive evaluation plots...")
    plot_path = plot_evaluation_results_all_splits(results, args)
    logger.info(f"Plots saved to: {plot_path}")
    
    # Save results
    logger.info("Saving comprehensive evaluation results...")
    results_path = save_comprehensive_results(results, args)
    logger.info(f"Results saved to: {results_path}")
    
    total_time = time.time() - start_time
    logger.info(f"Evaluation completed in {total_time:.2f} seconds")
    
    return results

def print_evaluation_summary(results, args):
    """
    Print a concise evaluation summary for all splits
    
    Args:
        results: Results from evaluate_all_splits()
        args: Evaluation arguments
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*80)
    
    # Table header
    print(f"{'Split':<10} {'Task':<15} {'AUC':<8} {'Accuracy':<10} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Cases':<8}")
    print("-" * 80)
    
    for split in ['train', 'val', 'test']:
        # Outcome prediction results
        outcome = results[split]['outcome_prediction']
        print(f"{split.capitalize():<10} {'Outcome':<15} {outcome['auc']:<8.4f} "
              f"{outcome['accuracy']:<10.4f} {outcome['f1']:<8.4f} "
              f"{outcome['precision']:<10.4f} {outcome['recall']:<8.4f} "
              f"{outcome['total_cases']:<8}")
        
        # Link prediction results
        if args.use_link_prediction and results[split]['link_prediction']:
            link = results[split]['link_prediction']
            total_links = link['num_positive'] + link['num_negative']
            print(f"{'':10} {'Link':<15} {link['auc']:<8.4f} "
                  f"{link['accuracy']:<10.4f} {link['f1']:<8.4f} "
                  f"{link['precision']:<10.4f} {link['recall']:<8.4f} "
                  f"{total_links:<8}")
    
    print("="*80)
    
    # Key insights
    test_outcome = results['test']['outcome_prediction']
    print(f"🎯 KEY RESULTS:")
    print(f"   Test Outcome AUC: {test_outcome['auc']:.4f}")
    print(f"   Test Outcome F1:  {test_outcome['f1']:.4f}")
    
    if args.use_link_prediction and results['test']['link_prediction']:
        test_link = results['test']['link_prediction']
        print(f"   Test Link AUC:    {test_link['auc']:.4f}")
    
    # Check for overfitting
    train_auc = results['train']['outcome_prediction']['auc']
    test_auc = results['test']['outcome_prediction']['auc']
    overfitting_gap = train_auc - test_auc
    
    if overfitting_gap > 0.1:
        print(f"⚠️  OVERFITTING DETECTED: Train AUC ({train_auc:.4f}) >> Test AUC ({test_auc:.4f})")
    elif overfitting_gap < 0.02:
        print(f"✅ GOOD GENERALIZATION: Train-Test gap = {overfitting_gap:.4f}")
    
    print("="*80)