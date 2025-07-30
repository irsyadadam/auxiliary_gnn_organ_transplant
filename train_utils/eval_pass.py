"""
eval_pass.py - Modular evaluation functions with multi-task learning support
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

def evaluate_link_prediction(model, graph_data, transplant_edges, args):
    """Evaluate link prediction performance"""
    model.eval()
    device = args.device
    
    with torch.no_grad():
        # Get positive edges
        pos_edges = transplant_edges.to(device)
        
        # Create negative samples for evaluation
        from train_pass import create_negative_samples
        neg_edges = create_negative_samples(
            pos_edges, 
            graph_data.x.shape[0], 
            args.negative_sampling_ratio, 
            device
        )
        
        # Get embeddings
        node_embeddings = model.encoder(graph_data.x, graph_data.edge_index, graph_data.edge_type)
        
        # Predict links
        pos_scores = model.decoder(node_embeddings, pos_edges, edge_type=torch.zeros(pos_edges.shape[1], device=device))
        neg_scores = model.decoder(node_embeddings, neg_edges, edge_type=torch.zeros(neg_edges.shape[1], device=device))
        
        # Combine scores and labels
        all_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        all_labels = torch.cat([
            torch.ones(pos_scores.shape[0]), 
            torch.zeros(neg_scores.shape[0])
        ]).cpu().numpy()
        
        # Compute metrics
        probabilities = torch.sigmoid(torch.tensor(all_scores)).numpy()
        predictions = (probabilities > 0.5).astype(int)
        
        link_metrics = {
            'auc': roc_auc_score(all_labels, probabilities),
            'accuracy': accuracy_score(all_labels, predictions),
            'precision': precision_recall_fscore_support(all_labels, predictions, average='binary')[0],
            'recall': precision_recall_fscore_support(all_labels, predictions, average='binary')[1],
            'f1': precision_recall_fscore_support(all_labels, predictions, average='binary')[2],
            'num_positive': int(np.sum(all_labels)),
            'num_negative': int(len(all_labels) - np.sum(all_labels))
        }
        
        # Store predictions for plotting
        link_metrics['predictions'] = probabilities
        link_metrics['labels'] = all_labels
    
    return link_metrics

def evaluate_outcome_prediction(model, graph_data, pair_to_label, donor_mapping, recipient_mapping, args):
    """Evaluate outcome prediction performance"""
    model.eval()
    device = args.device
    
    with torch.no_grad():
        # Get node embeddings
        node_embeddings = model.encoder(graph_data.x, graph_data.edge_index, graph_data.edge_type)
        
        # Prepare data
        donor_nodes = []
        recipient_nodes = []
        labels = []
        
        for (donor_id, recipient_id), label in pair_to_label.items():
            donor_nodes.append(donor_id)
            recipient_nodes.append(recipient_id)
            labels.append(label)
        
        if len(donor_nodes) == 0:
            return {'error': 'No outcome data available'}
        
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
        
        # Compute metrics
        outcome_metrics = {
            'auc': roc_auc_score(labels_np, probabilities) if len(np.unique(labels_np)) > 1 else 0.0,
            'accuracy': accuracy_score(labels_np, predictions_binary),
            'precision': precision_recall_fscore_support(labels_np, predictions_binary, average='binary')[0],
            'recall': precision_recall_fscore_support(labels_np, predictions_binary, average='binary')[1],
            'f1': precision_recall_fscore_support(labels_np, predictions_binary, average='binary')[2],
            'num_positive': int(np.sum(labels_np)),
            'num_negative': int(len(labels_np) - np.sum(labels_np)),
            'total_cases': len(labels_np)
        }
        
        # Confusion matrix
        cm = confusion_matrix(labels_np, predictions_binary)
        outcome_metrics['confusion_matrix'] = cm.tolist()
        
        # Store predictions for plotting
        outcome_metrics['predictions'] = probabilities
        outcome_metrics['labels'] = labels_np
    
    return outcome_metrics

def plot_evaluation_results(link_metrics, outcome_metrics, args):
    """Create evaluation plots"""
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ROC Curves
    if 'predictions' in outcome_metrics and 'labels' in outcome_metrics:
        fpr, tpr, _ = roc_curve(outcome_metrics['labels'], outcome_metrics['predictions'])
        axes[0, 0].plot(fpr, tpr, label=f"Outcome Prediction (AUC = {outcome_metrics['auc']:.3f})")
    
    if args.use_link_prediction and 'predictions' in link_metrics and 'labels' in link_metrics:
        fpr_link, tpr_link, _ = roc_curve(link_metrics['labels'], link_metrics['predictions'])
        axes[0, 0].plot(fpr_link, tpr_link, label=f"Link Prediction (AUC = {link_metrics['auc']:.3f})")
    
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curves
    if 'predictions' in outcome_metrics and 'labels' in outcome_metrics:
        precision, recall, _ = precision_recall_curve(outcome_metrics['labels'], outcome_metrics['predictions'])
        axes[0, 1].plot(recall, precision, label=f"Outcome Prediction")
    
    if args.use_link_prediction and 'predictions' in link_metrics and 'labels' in link_metrics:
        precision_link, recall_link, _ = precision_recall_curve(link_metrics['labels'], link_metrics['predictions'])
        axes[0, 1].plot(recall_link, precision_link, label=f"Link Prediction")
    
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix for Outcome Prediction
    if 'confusion_matrix' in outcome_metrics:
        cm = np.array(outcome_metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Complication', 'Complication'],
                   yticklabels=['No Complication', 'Complication'],
                   ax=axes[1, 0])
        axes[1, 0].set_title('Outcome Prediction Confusion Matrix')
        axes[1, 0].set_ylabel('True Label')
        axes[1, 0].set_xlabel('Predicted Label')
    
    # 4. Metrics Comparison Bar Plot
    metrics_data = {}
    if outcome_metrics.get('auc', 0) > 0:
        metrics_data['Outcome'] = {
            'AUC': outcome_metrics['auc'],
            'Accuracy': outcome_metrics['accuracy'],
            'F1': outcome_metrics['f1'],
            'Precision': outcome_metrics['precision'],
            'Recall': outcome_metrics['recall']
        }
    
    if args.use_link_prediction and link_metrics.get('auc', 0) > 0:
        metrics_data['Link'] = {
            'AUC': link_metrics['auc'],
            'Accuracy': link_metrics['accuracy'],
            'F1': link_metrics['f1'],
            'Precision': link_metrics['precision'],
            'Recall': link_metrics['recall']
        }
    
    if metrics_data:
        x = np.arange(len(list(metrics_data.values())[0]))
        width = 0.35
        
        for i, (task, metrics) in enumerate(metrics_data.items()):
            values = list(metrics.values())
            axes[1, 1].bar(x + i*width, values, width, label=task, alpha=0.8)
        
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Performance Metrics Comparison')
        axes[1, 1].set_xticks(x + width/2)
        axes[1, 1].set_xticklabels(list(list(metrics_data.values())[0].keys()), rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, 'evaluation_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def save_evaluation_results(link_metrics, outcome_metrics, args):
    """Save evaluation results to files"""
    results = {
        'outcome_prediction': {k: v for k, v in outcome_metrics.items() 
                              if k not in ['predictions', 'labels']},
        'evaluation_args': vars(args),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if args.use_link_prediction:
        results['link_prediction'] = {k: v for k, v in link_metrics.items() 
                                     if k not in ['predictions', 'labels']}
    
    # Save main results
    results_path = os.path.join(args.results_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed classification report for outcome prediction
    if 'predictions' in outcome_metrics and 'labels' in outcome_metrics:
        predictions = (outcome_metrics['predictions'] > 0.5).astype(int)
        report = classification_report(
            outcome_metrics['labels'], 
            predictions, 
            target_names=['No Complication', 'Complication'],
            output_dict=True
        )
        
        report_path = os.path.join(args.results_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    return results_path

def evaluate_model(model, graph_data, transplant_edges, pair_to_label, donor_mapping, recipient_mapping, args):
    """Main evaluation function with multi-task support"""
    
    # Setup logging
    logger = setup_eval_logging(args)
    
    logger.info("Starting model evaluation")
    logger.info(f"Multi-task evaluation: {args.use_link_prediction}")
    
    start_time = time.time()
    
    # Move data to device
    graph_data = graph_data.to(args.device)
    
    # Evaluate outcome prediction (primary task)
    logger.info("Evaluating outcome prediction...")
    outcome_metrics = evaluate_outcome_prediction(
        model, graph_data, pair_to_label, donor_mapping, recipient_mapping, args
    )
    
    # Evaluate link prediction (auxiliary task) if enabled
    link_metrics = {}
    if args.use_link_prediction:
        logger.info("Evaluating link prediction...")
        link_metrics = evaluate_link_prediction(model, graph_data, transplant_edges, args)
    
    # Log results
    logger.info("=== EVALUATION RESULTS ===")
    logger.info(f"Outcome Prediction - AUC: {outcome_metrics.get('auc', 0):.4f}, "
                f"Accuracy: {outcome_metrics.get('accuracy', 0):.4f}, "
                f"F1: {outcome_metrics.get('f1', 0):.4f}")
    
    if args.use_link_prediction and link_metrics:
        logger.info(f"Link Prediction - AUC: {link_metrics.get('auc', 0):.4f}, "
                    f"Accuracy: {link_metrics.get('accuracy', 0):.4f}, "
                    f"F1: {link_metrics.get('f1', 0):.4f}")
    
    # Create plots
    logger.info("Creating evaluation plots...")
    plot_path = plot_evaluation_results(link_metrics, outcome_metrics, args)
    logger.info(f"Plots saved to: {plot_path}")
    
    # Save results
    logger.info("Saving evaluation results...")
    results_path = save_evaluation_results(link_metrics, outcome_metrics, args)
    logger.info(f"Results saved to: {results_path}")
    
    total_time = time.time() - start_time
    logger.info(f"Evaluation completed in {total_time:.2f} seconds")
    
    return outcome_metrics, link_metrics

def print_evaluation_summary(outcome_metrics, link_metrics, args):
    """Print a concise evaluation summary"""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # Outcome prediction results
    print(f"📊 OUTCOME PREDICTION ({args.outcome_variable})")
    print(f"   AUC:       {outcome_metrics.get('auc', 0):.4f}")
    print(f"   Accuracy:  {outcome_metrics.get('accuracy', 0):.4f}")
    print(f"   F1 Score:  {outcome_metrics.get('f1', 0):.4f}")
    print(f"   Precision: {outcome_metrics.get('precision', 0):.4f}")
    print(f"   Recall:    {outcome_metrics.get('recall', 0):.4f}")
    print(f"   Cases:     {outcome_metrics.get('total_cases', 0)} "
          f"({outcome_metrics.get('num_positive', 0)} positive, "
          f"{outcome_metrics.get('num_negative',