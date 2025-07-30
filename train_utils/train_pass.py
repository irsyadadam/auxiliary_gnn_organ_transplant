"""
train_pass.py - Modular training functions with multi-task learning support
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

def setup_logging(args):
    """Setup logging configuration"""
    os.makedirs(args.results_dir, exist_ok=True)
    
    log_file = os.path.join(args.results_dir, 'training.log')
    
    # Configure logging
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

def create_negative_samples(positive_edges, num_nodes, negative_ratio=1.0, device='cpu'):
    """Create negative samples for link prediction"""
    num_pos = positive_edges.shape[1]
    num_neg = int(num_pos * negative_ratio)
    
    # Random negative sampling
    neg_sources = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_targets = torch.randint(0, num_nodes, (num_neg,), device=device)
    
    negative_edges = torch.stack([neg_sources, neg_targets], dim=0)
    return negative_edges

def compute_link_prediction_loss(model, graph_data, transplant_edges, args):
    """Compute link prediction loss for auxiliary task"""
    device = args.device
    
    # Get positive edges (actual transplants)
    pos_edges = transplant_edges.to(device)
    
    # Create negative samples
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
        link_auc = roc_auc_score(all_labels, all_scores)
    
    return link_loss, link_auc

def compute_outcome_prediction_loss(model, graph_data, pair_to_label, donor_mapping, recipient_mapping, args):
    """Compute outcome prediction loss for primary task"""
    device = args.device
    
    # Get node embeddings
    node_embeddings = model.encoder(graph_data.x, graph_data.edge_index, graph_data.edge_type)
    
    # Prepare data for outcome prediction
    donor_nodes = []
    recipient_nodes = []
    labels = []
    
    for (donor_id, recipient_id), label in pair_to_label.items():
        donor_nodes.append(donor_id)
        recipient_nodes.append(recipient_id)
        labels.append(label)
    
    if len(donor_nodes) == 0:
        return torch.tensor(0.0, device=device), 0.0
    
    donor_nodes = torch.tensor(donor_nodes, device=device)
    recipient_nodes = torch.tensor(recipient_nodes, device=device)
    labels = torch.tensor(labels, dtype=torch.float32, device=device)
    
    # Get donor and recipient embeddings
    donor_embeds = node_embeddings[donor_nodes]
    recipient_embeds = node_embeddings[recipient_nodes]
    
    # Predict outcomes
    predictions = model.classifier(donor_embeds, recipient_embeds)
    
    # Compute loss
    outcome_loss = F.binary_cross_entropy_with_logits(predictions.squeeze(), labels)
    
    # Compute metrics
    with torch.no_grad():
        probs = torch.sigmoid(predictions.squeeze()).cpu().numpy()
        labels_np = labels.cpu().numpy()
        outcome_auc = roc_auc_score(labels_np, probs) if len(np.unique(labels_np)) > 1 else 0.0
    
    return outcome_loss, outcome_auc

def train_epoch(model, graph_data, transplant_edges, pair_to_label, donor_mapping, recipient_mapping, 
                optimizer, scheduler, args, epoch, logger):
    """Train for one epoch with optional multi-task learning"""
    model.train()
    device = args.device
    
    total_loss = 0.0
    link_loss_total = 0.0
    outcome_loss_total = 0.0
    link_auc_total = 0.0
    outcome_auc_total = 0.0
    
    # Move data to device
    graph_data = graph_data.to(device)
    
    optimizer.zero_grad()
    
    # Compute outcome prediction loss (primary task)
    outcome_loss, outcome_auc = compute_outcome_prediction_loss(
        model, graph_data, pair_to_label, donor_mapping, recipient_mapping, args
    )
    
    total_loss = args.alpha * outcome_loss
    outcome_loss_total = outcome_loss.item()
    outcome_auc_total = outcome_auc
    
    # Compute link prediction loss (auxiliary task) if enabled
    if args.use_link_prediction:
        link_loss, link_auc = compute_link_prediction_loss(model, graph_data, transplant_edges, args)
        total_loss += (1 - args.alpha) * link_loss
        link_loss_total = link_loss.item()
        link_auc_total = link_auc
    
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
        log_msg = f"Epoch {epoch:04d} | Total Loss: {total_loss.item():.4f} | Outcome Loss: {outcome_loss_total:.4f} | Outcome AUC: {outcome_auc_total:.4f}"
        if args.use_link_prediction:
            log_msg += f" | Link Loss: {link_loss_total:.4f} | Link AUC: {link_auc_total:.4f}"
        logger.info(log_msg)
        
        if args.verbose:
            print(log_msg)
    
    metrics = {
        'total_loss': total_loss.item(),
        'outcome_loss': outcome_loss_total,
        'outcome_auc': outcome_auc_total,
        'link_loss': link_loss_total,
        'link_auc': link_auc_total
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
    """Main training function with multi-task learning support"""
    
    # Setup logging
    logger = setup_logging(args)
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, args)
    
    # Training history
    training_history = defaultdict(list)
    best_metric = 0.0
    patience_counter = 0
    
    logger.info(f"Starting training for {args.epochs} epochs")
    logger.info(f"Multi-task learning: {args.use_link_prediction}")
    logger.info(f"Alpha (outcome weight): {args.alpha}")
    
    start_time = time.time()
    
    try:
        for epoch in range(1, args.epochs + 1):
            # Train epoch
            train_metrics = train_epoch(
                model, graph_data, transplant_edges, pair_to_label, 
                donor_mapping, recipient_mapping, optimizer, scheduler, 
                args, epoch, logger
            )
            
            # Store metrics
            for key, value in train_metrics.items():
                training_history[f'train_{key}'].append(value)
            
            # Check for best model (using outcome AUC as primary metric)
            current_metric = train_metrics['outcome_auc']
            is_best = current_metric > best_metric
            
            if is_best:
                best_metric = current_metric
                patience_counter = 0
                logger.info(f"New best model at epoch {epoch} with outcome AUC: {best_metric:.4f}")
            else:
                patience_counter += 1
            
            # Save checkpoint
            if epoch % args.save_interval == 0 or is_best:
                save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, args, is_best)
            
            # Early stopping
            if args.early_stopping > 0 and patience_counter >= args.early_stopping:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Scheduler step for plateau scheduler
            if scheduler is not None and args.scheduler == 'plateau':
                scheduler.step(train_metrics['total_loss'])
    
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
        logger.info(f"Best outcome AUC: {best_metric:.4f}")
    
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