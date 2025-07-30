"""
train_pass.py - Updated training functions with proper train/val/test splits
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import json
import logging
from collections import defaultdict

# FIXED IMPORT PATH
from utils.data_splitting import create_data_splits, get_split_data

def setup_logging(args):
    """Setup logging configuration"""
    os.makedirs(args.results_dir, exist_ok=True)
    
    log_file = os.path.join(args.results_dir, 'training.log')
    
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if args.verbose else logging.NullHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Training started with arguments: {vars(args)}")
    return logger

def setup_optimizer_and_scheduler(model, args):
    """Setup optimizer and learning rate scheduler"""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.epochs//3, gamma=0.1)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    return optimizer, scheduler

def compute_link_prediction_loss(model, graph_data, pos_edges, neg_edges, args):
    """
    Compute link prediction loss using proper train/val/test splits
    
    Args:
        model: Joint model with encoder/decoder
        graph_data: Full graph data (used for node embeddings)
        pos_edges: Positive edges for this split
        neg_edges: Negative edges for this split
        args: Training arguments
    
    Returns:
        tuple: (loss, auc_score)
    """
    device = args.device
    
    # Move edges to device
    pos_edges = pos_edges.to(device)
    neg_edges = neg_edges.to(device)
    
    # Get embeddings from full graph (but only use split-specific edges for loss)
    node_embeddings = model.gnn_encoder(graph_data.x, graph_data.edge_index, graph_data.edge_type)
    
    # FIXED: Use consistent decoder interface
    edge_type_zeros = torch.zeros(pos_edges.shape[1], device=device, dtype=torch.long)
    
    # Get positive scores - handle decoder return format
    decoder_output = model.decoder(node_embeddings, pos_edges, edge_type_zeros)
    if isinstance(decoder_output, tuple):
        pos_scores = decoder_output[0]  # Take first element if tuple
    else:
        pos_scores = decoder_output
    
    # Get negative scores
    edge_type_zeros_neg = torch.zeros(neg_edges.shape[1], device=device, dtype=torch.long)
    decoder_output_neg = model.decoder(node_embeddings, neg_edges, edge_type_zeros_neg)
    if isinstance(decoder_output_neg, tuple):
        neg_scores = decoder_output_neg[0]  # Take first element if tuple
    else:
        neg_scores = decoder_output_neg
    
    # Binary classification loss
    pos_labels = torch.ones(pos_scores.shape[0], device=device)
    neg_labels = torch.zeros(neg_scores.shape[0], device=device)
    
    link_loss = F.binary_cross_entropy_with_logits(
        torch.cat([pos_scores, neg_scores]), 
        torch.cat([pos_labels, neg_labels])
    )
    
    # Compute metrics
    with torch.no_grad():
        all_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        all_labels = torch.cat([pos_labels, neg_labels]).cpu().numpy()
        link_auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    
    return link_loss, link_auc

def compute_outcome_prediction_loss(model, graph_data, outcome_pairs, args):
    """
    Compute outcome prediction loss using proper train/val/test splits
    
    Args:
        model: Joint model with encoder/classifier
        graph_data: Full graph data
        outcome_pairs: Dict of (donor_node, recipient_node) -> label for this split
        args: Training arguments
    
    Returns:
        tuple: (loss, auc_score)
    """
    device = args.device
    
    if not outcome_pairs:
        return torch.tensor(0.0, device=device), 0.0
    
    # Get node embeddings from full graph
    node_embeddings = model.gnn_encoder(graph_data.x, graph_data.edge_index, graph_data.edge_type)
    
    # Prepare data for this split
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
    
    # Get donor and recipient embeddings
    donor_embeds = node_embeddings[donor_nodes]
    recipient_embeds = node_embeddings[recipient_nodes]
    
    # Predict outcomes - FIXED: Use correct classifier interface
    predictions = model.classifier(donor_embeds, recipient_embeds)
    
    # Compute loss
    outcome_loss = F.binary_cross_entropy_with_logits(predictions.squeeze(), labels)
    
    # Compute metrics
    with torch.no_grad():
        probs = torch.sigmoid(predictions.squeeze()).cpu().numpy()
        labels_np = labels.cpu().numpy()
        outcome_auc = roc_auc_score(labels_np, probs) if len(np.unique(labels_np)) > 1 else 0.0
    
    return outcome_loss, outcome_auc

def evaluate_split(model, graph_data, splits, split_name, args):
    """
    Evaluate model on a specific data split (train/val/test)
    
    Args:
        model: Trained model
        graph_data: Full graph data
        splits: Data splits from create_data_splits()
        split_name: 'train', 'val', or 'test'
        args: Training arguments
    
    Returns:
        dict: Evaluation metrics for this split
    """
    model.eval()
    
    with torch.no_grad():
        # Get split-specific data
        outcome_pairs, pos_edges, neg_edges = get_split_data(splits, split_name)
        
        # Evaluate outcome prediction
        outcome_loss, outcome_auc = compute_outcome_prediction_loss(
            model, graph_data, outcome_pairs, args
        )
        
        # Evaluate link prediction if enabled
        link_loss, link_auc = 0.0, 0.0
        if args.use_link_prediction and pos_edges.shape[1] > 0:
            link_loss, link_auc = compute_link_prediction_loss(
                model, graph_data, pos_edges, neg_edges, args
            )
        
        metrics = {
            'outcome_loss': outcome_loss.item(),
            'outcome_auc': outcome_auc,
            'link_loss': link_loss.item() if isinstance(link_loss, torch.Tensor) else link_loss,
            'link_auc': link_auc,
            'total_loss': args.alpha * outcome_loss.item() + (1 - args.alpha) * (link_loss.item() if isinstance(link_loss, torch.Tensor) else link_loss)
        }
    
    return metrics

def train_epoch(model, graph_data, splits, optimizer, scheduler, args, epoch, logger):
    """
    Train for one epoch using proper train split
    
    Args:
        model: Model to train
        graph_data: Full graph data
        splits: Data splits from create_data_splits()
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        args: Training arguments
        epoch: Current epoch
        logger: Logger instance
    
    Returns:
        dict: Training metrics for this epoch
    """
    model.train()
    device = args.device
    
    # Move data to device
    graph_data = graph_data.to(device)
    
    # Get training data
    train_outcome_pairs, train_pos_edges, train_neg_edges = get_split_data(splits, 'train')
    
    optimizer.zero_grad()
    
    # Compute outcome prediction loss (primary task)
    outcome_loss, outcome_auc = compute_outcome_prediction_loss(
        model, graph_data, train_outcome_pairs, args
    )
    
    total_loss = args.alpha * outcome_loss
    
    # Compute link prediction loss (auxiliary task) if enabled
    link_loss, link_auc = 0.0, 0.0
    if args.use_link_prediction and train_pos_edges.shape[1] > 0:
        link_loss, link_auc = compute_link_prediction_loss(
            model, graph_data, train_pos_edges, train_neg_edges, args
        )
        total_loss += (1 - args.alpha) * link_loss
    
    # Backward pass
    total_loss.backward()
    
    # Gradient clipping
    if args.gradient_clipping > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
    
    optimizer.step()
    
    if scheduler is not None and args.scheduler != 'plateau':
        scheduler.step()
    
    # Logging
    if epoch % args.log_interval == 0:
        log_msg = f"Epoch {epoch:04d} | Total Loss: {total_loss.item():.4f} | Outcome Loss: {outcome_loss.item():.4f} | Outcome AUC: {outcome_auc:.4f}"
        if args.use_link_prediction:
            link_loss_val = link_loss.item() if isinstance(link_loss, torch.Tensor) else link_loss
            log_msg += f" | Link Loss: {link_loss_val:.4f} | Link AUC: {link_auc:.4f}"
        logger.info(log_msg)
        
        if args.verbose:
            print(log_msg)
    
    metrics = {
        'total_loss': total_loss.item(),
        'outcome_loss': outcome_loss.item(),
        'outcome_auc': outcome_auc,
        'link_loss': link_loss.item() if isinstance(link_loss, torch.Tensor) else link_loss,
        'link_auc': link_auc
    }
    
    return metrics

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, args, is_best=False):
    """Save model checkpoint"""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'args': vars(args)
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        
    # Save latest model
    latest_path = os.path.join(args.checkpoint_dir, 'latest_model.pt')
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path

def save_training_history(training_history, args):
    """Save training history to JSON file"""
    history_path = os.path.join(args.results_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

def train_model(model, graph_data, transplant_edges, pair_to_label, donor_mapping, recipient_mapping, args):
    """
    Main training function with proper train/val/test splits
    
    Args:
        model: Model to train
        graph_data: Full graph data
        transplant_edges: All transplant edges
        pair_to_label: All outcome pairs
        donor_mapping: Donor ID mappings
        recipient_mapping: Recipient ID mappings
        args: Training arguments
    
    Returns:
        tuple: (trained_model, training_history)
    """
    
    # Setup logging
    logger = setup_logging(args)
    
    # Create proper data splits
    logger.info("Creating train/validation/test splits...")
    splits = create_data_splits(pair_to_label, transplant_edges, graph_data.x.shape[0], args)
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, args)
    
    # Training history
    training_history = defaultdict(list)
    best_val_metric = 0.0
    patience_counter = 0
    
    logger.info(f"Starting training for {args.epochs} epochs")
    logger.info(f"Multi-task learning: {args.use_link_prediction}")
    logger.info(f"Alpha (outcome weight): {args.alpha}")
    
    start_time = time.time()
    
    try:
        for epoch in range(1, args.epochs + 1):
            # Train epoch
            train_metrics = train_epoch(
                model, graph_data, splits, optimizer, scheduler, 
                args, epoch, logger
            )
            
            # Store training metrics
            for key, value in train_metrics.items():
                training_history[f'train_{key}'].append(value)
            
            # Evaluate on validation set
            val_metrics = evaluate_split(model, graph_data, splits, 'val', args)
            for key, value in val_metrics.items():
                training_history[f'val_{key}'].append(value)
            
            # Check for best model (using validation outcome AUC)
            current_val_metric = val_metrics['outcome_auc']
            is_best = current_val_metric > best_val_metric
            
            if is_best:
                best_val_metric = current_val_metric
                patience_counter = 0
                logger.info(f"New best model at epoch {epoch} with val outcome AUC: {best_val_metric:.4f}")
            else:
                patience_counter += 1
            
            # Log validation metrics periodically
            if epoch % args.log_interval == 0:
                logger.info(f"Validation - Outcome AUC: {val_metrics['outcome_auc']:.4f}, "
                           f"Outcome Loss: {val_metrics['outcome_loss']:.4f}")
                if args.use_link_prediction:
                    logger.info(f"Validation - Link AUC: {val_metrics['link_auc']:.4f}, "
                               f"Link Loss: {val_metrics['link_loss']:.4f}")
            
            # Save checkpoint
            if epoch % args.save_interval == 0 or is_best:
                save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, args, is_best)
            
            # Early stopping
            if args.early_stopping > 0 and patience_counter >= args.early_stopping:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Scheduler step for plateau scheduler
            if scheduler is not None and args.scheduler == 'plateau':
                scheduler.step(val_metrics['total_loss'])
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    
    finally:
        # Save final checkpoint and training history
        save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, args, False)
        save_training_history(dict(training_history), args)
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Best validation outcome AUC: {best_val_metric:.4f}")
    
    # Evaluate on test set with best model
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_split(model, graph_data, splits, 'test', args)
    logger.info(f"Test Results - Outcome AUC: {test_metrics['outcome_auc']:.4f}")
    if args.use_link_prediction:
        logger.info(f"Test Results - Link AUC: {test_metrics['link_auc']:.4f}")
    
    # Add test metrics to history
    for key, value in test_metrics.items():
        training_history[f'test_{key}'] = value
    
    return model, dict(training_history)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']

# Cross-validation utility (optional)
def evaluate_with_cross_validation(model_class, graph_data, transplant_edges, pair_to_label, 
                                 donor_mapping, recipient_mapping, args, n_folds=5):
    """
    Perform k-fold cross-validation evaluation
    
    Args:
        model_class: Model class to instantiate
        graph_data: Full graph data
        transplant_edges: All transplant edges
        pair_to_label: All outcome pairs
        donor_mapping: Donor ID mappings
        recipient_mapping: Recipient ID mappings
        args: Training arguments
        n_folds: Number of CV folds
    
    Returns:
        dict: Cross-validation results
    """
    from sklearn.model_selection import StratifiedKFold
    
    # Prepare data for CV
    pairs = list(pair_to_label.keys())
    labels = list(pair_to_label.values())
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    
    cv_results = {
        'outcome_aucs': [],
        'link_aucs': [],
        'fold_histories': []
    }
    
    print(f"Starting {n_folds}-fold cross-validation...")
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(pairs, labels)):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        # Create fold-specific pair_to_label
        train_val_pairs = [pairs[i] for i in train_val_idx]
        test_pairs = [pairs[i] for i in test_idx]
        train_val_labels = [labels[i] for i in train_val_idx]
        test_labels = [labels[i] for i in test_idx]
        
        fold_pair_to_label = {pair: label for pair, label in zip(train_val_pairs, train_val_labels)}
        
        # Create new model for this fold
        model = model_class(
            input_dim=graph_data.x.shape[1],
            num_relations=graph_data.num_relations
        ).to(args.device)
        
        # Temporarily adjust args for this fold
        original_epochs = args.epochs
        args.epochs = min(50, args.epochs)  # Reduce epochs for CV
        
        # Train on this fold
        trained_model, fold_history = train_model(
            model, graph_data, transplant_edges, fold_pair_to_label,
            donor_mapping, recipient_mapping, args
        )
        
        # Restore original epochs
        args.epochs = original_epochs
        
        # Extract test performance
        test_outcome_auc = fold_history.get('test_outcome_auc', 0.0)
        test_link_auc = fold_history.get('test_link_auc', 0.0)
        
        cv_results['outcome_aucs'].append(test_outcome_auc)
        cv_results['link_aucs'].append(test_link_auc)
        cv_results['fold_histories'].append(fold_history)
        
        print(f"Fold {fold + 1} - Outcome AUC: {test_outcome_auc:.4f}")
        if args.use_link_prediction:
            print(f"Fold {fold + 1} - Link AUC: {test_link_auc:.4f}")
    
    # Calculate summary statistics
    outcome_aucs = np.array(cv_results['outcome_aucs'])
    link_aucs = np.array(cv_results['link_aucs'])
    
    cv_results['outcome_aucs'] = {
        'mean': np.mean(outcome_aucs),
        'std': np.std(outcome_aucs),
        'values': outcome_aucs.tolist()
    }
    
    cv_results['link_aucs'] = {
        'mean': np.mean(link_aucs),
        'std': np.std(link_aucs),
        'values': link_aucs.tolist()
    }
    
    print(f"\nCross-validation completed!")
    print(f"Outcome AUC: {cv_results['outcome_aucs']['mean']:.4f} ± {cv_results['outcome_aucs']['std']:.4f}")
    if args.use_link_prediction:
        print(f"Link AUC: {cv_results['link_aucs']['mean']:.4f} ± {cv_results['link_aucs']['std']:.4f}")
    
    return cv_results