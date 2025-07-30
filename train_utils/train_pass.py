"""
train_pass.py - Fixed training functions with proper device handling and gradient flow
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

def compute_ranking_link_prediction_loss(model, graph_data, pos_edges, num_nodes, args, 
                                       num_negatives_per_positive=50):
    """
    Compute ranking-based link prediction loss using margin ranking loss
    
    Args:
        model: Joint model
        graph_data: Full graph data
        pos_edges: Positive edges for training [2, num_pos]
        num_nodes: Total number of nodes in graph
        args: Training arguments
        num_negatives_per_positive: Number of negative samples per positive edge
    
    Returns:
        tuple: (loss, mrr_score)
    """
    device = args.device
    
    if pos_edges.shape[1] == 0:
        return torch.tensor(0.0, device=device), 0.0
    
    # Ensure edges are on correct device
    pos_edges = pos_edges.to(device)
    
    # Get embeddings
    embeddings = model.get_embeddings(graph_data.x, graph_data.edge_index, graph_data.edge_type)
    
    # Create edge types for link prediction (assume transplant relation = 0)
    pos_edge_types = torch.zeros(pos_edges.shape[1], device=device, dtype=torch.long)
    
    # Get positive scores
    pos_scores = model.predict_links(embeddings, pos_edges, pos_edge_types)
    
    # Generate negative samples for each positive edge
    total_loss = 0.0
    num_batches = 0
    all_ranks = []
    
    # Create positive edge set for filtering
    pos_edge_set = set()
    for i in range(pos_edges.shape[1]):
        pos_edge_set.add((pos_edges[0, i].item(), pos_edges[1, i].item()))
    
    for i in range(pos_edges.shape[1]):
        head, tail = pos_edges[0, i], pos_edges[1, i]
        pos_score = pos_scores[i]
        
        # Generate negative samples by corrupting tail
        neg_tails = torch.randint(0, num_nodes, (num_negatives_per_positive,), device=device)
        # Filter out positive edges
        valid_negs = []
        for neg_tail in neg_tails:
            if (head.item(), neg_tail.item()) not in pos_edge_set:
                valid_negs.append(neg_tail.item())
        
        if len(valid_negs) == 0:
            continue
            
        # Create negative edges
        neg_edges = torch.stack([
            head.repeat(len(valid_negs)),
            torch.tensor(valid_negs, device=device)
        ])
        neg_edge_types = torch.zeros(len(valid_negs), device=device, dtype=torch.long)
        
        # Get negative scores
        neg_scores = model.predict_links(embeddings, neg_edges, neg_edge_types)
        
        # Margin ranking loss: positive score should be higher than negative scores
        pos_score_expanded = pos_score.repeat(len(valid_negs))
        margin_loss = F.margin_ranking_loss(
            pos_score_expanded, neg_scores, 
            torch.ones(len(valid_negs), device=device),
            margin=1.0, reduction='mean'
        )
        
        total_loss += margin_loss
        num_batches += 1
        
        # Calculate rank for MRR (for monitoring during training)
        with torch.no_grad():
            all_scores = torch.cat([pos_score.unsqueeze(0), neg_scores])
            sorted_scores, _ = torch.sort(all_scores, descending=True)
            rank = (sorted_scores >= pos_score).sum().item()
            all_ranks.append(rank)
    
    if num_batches == 0:
        return torch.tensor(0.0, device=device), 0.0
    
    # Average loss across all positive edges
    avg_loss = total_loss / num_batches
    
    # Calculate MRR for monitoring
    mrr = np.mean([1.0 / rank for rank in all_ranks]) if all_ranks else 0.0
    
    return avg_loss, mrr

def evaluate_ranking_link_prediction(model, graph_data, test_edges, all_edges, num_nodes, args):
    """
    Evaluate ranking-based link prediction using MRR and Hits@K
    
    Args:
        model: Trained model
        graph_data: Full graph data
        test_edges: Test edges to evaluate [2, num_test]
        all_edges: All known edges for filtering [2, num_all]
        num_nodes: Total number of nodes
        args: Evaluation arguments
    
    Returns:
        dict: Ranking metrics (MRR, Hits@1, Hits@3, Hits@10)
    """
    model.eval()
    device = args.device
    
    if test_edges.shape[1] == 0:
        return {'mrr': 0.0, 'hits@1': 0.0, 'hits@3': 0.0, 'hits@10': 0.0}
    
    # Ensure data is on device
    test_edges = test_edges.to(device)
    all_edges = all_edges.to(device)
    
    # Create set of all known edges for filtering
    all_edge_set = set()
    for i in range(all_edges.shape[1]):
        all_edge_set.add((all_edges[0, i].item(), all_edges[1, i].item()))
    
    # Get embeddings
    with torch.no_grad():
        embeddings = model.get_embeddings(graph_data.x, graph_data.edge_index, graph_data.edge_type)
    
    ranks = []
    
    print(f"Evaluating {test_edges.shape[1]} test edges...")
    
    for i in tqdm(range(test_edges.shape[1]), desc="Ranking evaluation"):
        head, tail = test_edges[0, i], test_edges[1, i]
        
        # Create all possible tail candidates
        all_tails = torch.arange(num_nodes, device=device)
        all_test_edges = torch.stack([head.repeat(num_nodes), all_tails])
        
        # Get scores for all candidates
        edge_types = torch.zeros(num_nodes, device=device, dtype=torch.long)
        with torch.no_grad():
            all_scores = model.predict_links(embeddings, all_test_edges, edge_types)
        
        # Filter out known edges (except the target)
        filtered_scores = all_scores.clone()
        for j in range(num_nodes):
            if j != tail.item() and (head.item(), j) in all_edge_set:
                filtered_scores[j] = float('-inf')
        
        # Get rank of the true tail
        target_score = filtered_scores[tail.item()]
        rank = (filtered_scores >= target_score).sum().item()
        ranks.append(rank)
    
    # Calculate metrics
    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)
    hits_at_1 = np.mean(ranks == 1)
    hits_at_10 = np.mean(ranks <= 10)
    hits_at_100 = np.mean(ranks <= 100)
    
    metrics = {
        'mrr': mrr,
        'hits@1': hits_at_1,
        'hits@10': hits_at_10,
        'hits@100': hits_at_100,
        'mean_rank': np.mean(ranks),
        'num_test_edges': len(ranks)
    }
    
    return metrics

def compute_outcome_prediction_loss(model, graph_data, outcome_pairs, args):
    """
    Compute outcome prediction loss with proper device handling
    """
    device = args.device
    
    if not outcome_pairs:
        return torch.tensor(0.0, device=device), 0.0
    
    # Get embeddings
    embeddings = model.get_embeddings(graph_data.x, graph_data.edge_index, graph_data.edge_type)
    
    # Prepare data for this split
    donor_nodes = []
    recipient_nodes = []
    labels = []
    
    for (donor_id, recipient_id), label in outcome_pairs.items():
        donor_nodes.append(donor_id)
        recipient_nodes.append(recipient_id)
        labels.append(label)
    
    # Convert to tensors and move to device
    donor_nodes = torch.tensor(donor_nodes, device=device, dtype=torch.long)
    recipient_nodes = torch.tensor(recipient_nodes, device=device, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32, device=device)
    
    # Predict outcomes
    predictions = model.predict_outcomes(embeddings, donor_nodes, recipient_nodes)
    
    # Compute loss
    outcome_loss = F.binary_cross_entropy_with_logits(predictions.squeeze(), labels)
    
    # Compute metrics
    with torch.no_grad():
        probs = torch.sigmoid(predictions.squeeze()).cpu().numpy()
        labels_np = labels.cpu().numpy()
        outcome_auc = roc_auc_score(labels_np, probs) if len(np.unique(labels_np)) > 1 else 0.0
    
    return outcome_loss, outcome_auc

def evaluate_split(model, graph_data, splits, split_name, args, all_transplant_edges=None):
    """
    Evaluate model on a specific data split with ranking-based link prediction
    """
    model.eval()
    device = args.device
    
    # Ensure graph data is on device
    graph_data = graph_data.to(device)
    
    with torch.no_grad():
        # Get split-specific data
        outcome_pairs, pos_edges, _ = get_split_data(splits, split_name)
        
        # Evaluate outcome prediction
        outcome_loss, outcome_auc = compute_outcome_prediction_loss(
            model, graph_data, outcome_pairs, args
        )
        
        # Evaluate ranking-based link prediction if enabled
        link_loss, link_mrr = 0.0, 0.0
        link_metrics = {}
        
        if args.use_link_prediction and pos_edges.shape[1] > 0 and all_transplant_edges is not None:
            if split_name == 'train':
                # For training, use margin ranking loss
                link_loss, link_mrr = compute_ranking_link_prediction_loss(
                    model, graph_data, pos_edges, graph_data.x.shape[0], args
                )
            else:
                # For val/test, use full ranking evaluation
                link_metrics = evaluate_ranking_link_prediction(
                    model, graph_data, pos_edges, all_transplant_edges, 
                    graph_data.x.shape[0], args
                )
                link_mrr = link_metrics.get('mrr', 0.0)
                link_loss = 0.0  # No loss computation for evaluation
        
        metrics = {
            'outcome_loss': outcome_loss.item(),
            'outcome_auc': outcome_auc,
            'link_loss': link_loss.item() if isinstance(link_loss, torch.Tensor) else link_loss,
            'link_mrr': link_mrr,
            'total_loss': args.alpha * outcome_loss.item() + (1 - args.alpha) * (link_loss.item() if isinstance(link_loss, torch.Tensor) else link_loss)
        }
        
        if link_metrics:
            metrics.update({
                'link_hits@1': link_metrics.get('hits@1', 0.0),
                'link_hits@10': link_metrics.get('hits@10', 0.0),
                'link_hits@100': link_metrics.get('hits@100', 0.0),
                'link_mean_rank': link_metrics.get('mean_rank', 0.0)
            })
    
    return metrics

def train_epoch(model, graph_data, splits, optimizer, scheduler, args, epoch, logger, all_transplant_edges):
    """
    Train for one epoch with ranking-based link prediction
    """
    model.train()
    device = args.device
    
    # Ensure graph data is on device
    graph_data = graph_data.to(device)
    
    # Get training data
    train_outcome_pairs, train_pos_edges, _ = get_split_data(splits, 'train')
    
    optimizer.zero_grad()
    
    # Compute outcome prediction loss (primary task)
    outcome_loss, outcome_auc = compute_outcome_prediction_loss(
        model, graph_data, train_outcome_pairs, args
    )
    
    total_loss = args.alpha * outcome_loss
    
    # Compute ranking-based link prediction loss (auxiliary task) if enabled
    link_loss, link_mrr = 0.0, 0.0
    if args.use_link_prediction and train_pos_edges.shape[1] > 0:
        link_loss, link_mrr = compute_ranking_link_prediction_loss(
            model, graph_data, train_pos_edges, graph_data.x.shape[0], args
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
            log_msg += f" | Link Loss: {link_loss_val:.4f} | Link MRR: {link_mrr:.4f}"
        logger.info(log_msg)
        
        if args.verbose:
            print(log_msg)
    
    metrics = {
        'total_loss': total_loss.item(),
        'outcome_loss': outcome_loss.item(),
        'outcome_auc': outcome_auc,
        'link_loss': link_loss.item() if isinstance(link_loss, torch.Tensor) else link_loss,
        'link_mrr': link_mrr
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
    Main training function with ranking-based link prediction
    """
    
    # Setup logging
    logger = setup_logging(args)
    
    # Ensure model is on correct device
    model = model.to(args.device)
    
    # Create proper data splits (modified for ranking-based LP)
    logger.info("Creating train/validation/test splits...")
    splits = create_ranking_data_splits(pair_to_label, transplant_edges, graph_data.x.shape[0], args)
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, args)
    
    # Training history
    training_history = defaultdict(list)
    best_val_metric = 0.0
    patience_counter = 0
    
    logger.info(f"Starting training for {args.epochs} epochs")
    logger.info(f"Multi-task learning: {args.use_link_prediction}")
    logger.info(f"Link prediction: Ranking-based (MRR, Hits@K)")
    logger.info(f"Alpha (outcome weight): {args.alpha}")
    logger.info(f"Device: {args.device}")
    
    start_time = time.time()
    
    try:
        for epoch in range(1, args.epochs + 1):
            # Train epoch
            train_metrics = train_epoch(
                model, graph_data, splits, optimizer, scheduler, 
                args, epoch, logger, transplant_edges
            )
            
            # Store training metrics
            for key, value in train_metrics.items():
                training_history[f'train_{key}'].append(value)
            
            # Evaluate on validation set
            val_metrics = evaluate_split(model, graph_data, splits, 'val', args, transplant_edges)
            for key, value in val_metrics.items():
                training_history[f'val_{key}'].append(value)
            
            # Check for best model (using validation outcome AUC + link MRR)
            current_val_metric = val_metrics['outcome_auc']
            if args.use_link_prediction:
                # Combine outcome and link metrics for best model selection
                current_val_metric = args.alpha * val_metrics['outcome_auc'] + (1 - args.alpha) * val_metrics['link_mrr']
            
            is_best = current_val_metric > best_val_metric
            
            if is_best:
                best_val_metric = current_val_metric
                patience_counter = 0
                logger.info(f"New best model at epoch {epoch} with combined metric: {best_val_metric:.4f}")
            else:
                patience_counter += 1
            
            # Log validation metrics periodically
            if epoch % args.log_interval == 0:
                logger.info(f"Validation - Outcome AUC: {val_metrics['outcome_auc']:.4f}, "
                           f"Outcome Loss: {val_metrics['outcome_loss']:.4f}")
                if args.use_link_prediction:
                    logger.info(f"Validation - Link MRR: {val_metrics['link_mrr']:.4f}")
                    if 'link_hits@10' in val_metrics:
                        logger.info(f"Validation - Hits@10: {val_metrics['link_hits@10']:.4f}")
            
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
        logger.info(f"Best validation combined metric: {best_val_metric:.4f}")
    
    # Evaluate on test set with best model
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_split(model, graph_data, splits, 'test', args, transplant_edges)
    logger.info(f"Test Results - Outcome AUC: {test_metrics['outcome_auc']:.4f}")
    if args.use_link_prediction:
        logger.info(f"Test Results - Link MRR: {test_metrics['link_mrr']:.4f}")
        logger.info(f"Test Results - Hits@1: {test_metrics.get('link_hits@1', 0):.4f}")
        logger.info(f"Test Results - Hits@10: {test_metrics.get('link_hits@10', 0):.4f}")
    
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