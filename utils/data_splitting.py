"""
data_splitting.py - Data splitting utilities for PyTorch Geometric multi-task learning

This module handles proper train/validation/test splits for both:
1. Outcome prediction (donor-recipient pairs)
2. Ranking-based link prediction (transplant edges)
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

def stratified_split_pairs(pair_to_label, test_ratio=0.2, val_ratio=0.1, random_state=42):
    """
    Stratified split of donor-recipient pairs for outcome prediction
    
    Args:
        pair_to_label: Dict mapping (donor_node, recipient_node) -> label
        test_ratio: Proportion for test set
        val_ratio: Proportion for validation set (from remaining after test)
        random_state: Random seed
    
    Returns:
        tuple: (train_pairs, val_pairs, test_pairs) where each is a dict
    """
    
    if not pair_to_label:
        return {}, {}, {}
    
    pairs = list(pair_to_label.keys())
    labels = list(pair_to_label.values())
    
    print(f"Splitting {len(pairs)} donor-recipient pairs")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # First split: separate test set
    train_val_pairs, test_pairs, train_val_labels, test_labels = train_test_split(
        pairs, labels, 
        test_size=test_ratio, 
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: separate validation from training
    if val_ratio > 0 and len(train_val_pairs) > 2:
        # Adjust val_ratio to be relative to remaining data
        adjusted_val_ratio = val_ratio / (1 - test_ratio)
        
        train_pairs, val_pairs, train_labels, val_labels = train_test_split(
            train_val_pairs, train_val_labels,
            test_size=adjusted_val_ratio,
            stratify=train_val_labels,
            random_state=random_state + 1
        )
    else:
        train_pairs, val_pairs = train_val_pairs, []
        train_labels, val_labels = train_val_labels, []
    
    # Convert back to dictionaries
    train_dict = {pair: label for pair, label in zip(train_pairs, train_labels)}
    val_dict = {pair: label for pair, label in zip(val_pairs, val_labels)}
    test_dict = {pair: label for pair, label in zip(test_pairs, test_labels)}
    
    print(f"Split results:")
    print(f"  Train: {len(train_dict)} pairs ({len(train_dict)/(len(pair_to_label)):.1%})")
    print(f"  Val:   {len(val_dict)} pairs ({len(val_dict)/(len(pair_to_label)):.1%})")
    print(f"  Test:  {len(test_dict)} pairs ({len(test_dict)/(len(pair_to_label)):.1%})")
    
    return train_dict, val_dict, test_dict

def split_transplant_edges_ranking(transplant_edges, test_ratio=0.2, val_ratio=0.1, random_state=42):
    """
    Split transplant edges for ranking-based link prediction evaluation
    
    Args:
        transplant_edges: torch.Tensor of shape [2, num_edges]
        test_ratio: Proportion for test set
        val_ratio: Proportion for validation set
        random_state: Random seed
    
    Returns:
        tuple: (train_edges, val_edges, test_edges)
    """
    
    num_edges = transplant_edges.shape[1]
    indices = np.arange(num_edges)
    
    print(f"Splitting {num_edges} transplant edges for ranking-based link prediction")
    
    # Random shuffle
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    # Calculate split points
    test_size = int(num_edges * test_ratio)
    val_size = int(num_edges * val_ratio)
    train_size = num_edges - test_size - val_size
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Split edges
    train_edges = transplant_edges[:, train_indices]
    val_edges = transplant_edges[:, val_indices]
    test_edges = transplant_edges[:, test_indices]
    
    print(f"Ranking-based link prediction edge splits:")
    print(f"  Train: {train_edges.shape[1]} edges ({train_edges.shape[1]/num_edges:.1%})")
    print(f"  Val:   {val_edges.shape[1]} edges ({val_edges.shape[1]/num_edges:.1%})")
    print(f"  Test:  {test_edges.shape[1]} edges ({test_edges.shape[1]/num_edges:.1%})")
    
    return train_edges, val_edges, test_edges

def create_data_splits(pair_to_label, transplant_edges, num_nodes, args):
    """
    Create comprehensive data splits for multi-task learning with ranking-based LP
    
    Args:
        pair_to_label: Dict mapping (donor_node, recipient_node) -> outcome
        transplant_edges: torch.Tensor of transplant edges [2, num_edges]
        num_nodes: Total number of nodes
        args: Arguments containing split ratios and random seed
    
    Returns:
        dict: Comprehensive data splits containing:
            - outcome_splits: {train, val, test} dicts for outcome prediction
            - link_splits: {train, val, test} edge tensors for ranking-based link prediction
            - negative_edges: Empty tensors (not used in ranking approach)
    """
    
    print("\n" + "="*60)
    print("CREATING DATA SPLITS FOR RANKING-BASED LINK PREDICTION")
    print("="*60)
    
    # Split outcome prediction pairs
    train_pairs, val_pairs, test_pairs = stratified_split_pairs(
        pair_to_label, 
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        random_state=args.seed
    )
    
    # Split transplant edges for ranking-based link prediction
    train_edges, val_edges, test_edges = split_transplant_edges_ranking(
        transplant_edges,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        random_state=args.seed
    )
    
    # For ranking-based approach, we don't pre-generate negative samples
    # They are created on-the-fly during evaluation
    empty_neg_edges = torch.empty((2, 0), dtype=torch.long)
    
    splits = {
        'outcome_splits': {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        },
        'link_splits': {
            'train': train_edges,
            'val': val_edges,
            'test': test_edges
        },
        'negative_edges': {  # Empty for ranking-based approach
            'train': empty_neg_edges,
            'val': empty_neg_edges,
            'test': empty_neg_edges
        }
    }
    
    print(f"\nRanking-based data splitting completed!")
    print(f"Ready for MRR and Hits@K evaluation")
    
    return splits

def get_split_data(splits, split_name='train'):
    """
    Get data for a specific split (train, val, or test)
    
    Args:
        splits: Data splits from create_data_splits()
        split_name: 'train', 'val', or 'test'
    
    Returns:
        tuple: (outcome_pairs, pos_edges, neg_edges)
        Note: neg_edges will be empty for ranking-based approach
    """
    
    if split_name not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid split_name: {split_name}. Must be 'train', 'val', or 'test'")
    
    outcome_pairs = splits['outcome_splits'][split_name]
    pos_edges = splits['link_splits'][split_name]
    neg_edges = splits['negative_edges'][split_name]  # Will be empty for ranking approach
    
    return outcome_pairs, pos_edges, neg_edges

# Legacy compatibility functions (for binary classification approach)
def create_negative_samples_split(positive_edges, num_nodes, negative_ratio=1.0, random_state=42):
    """
    Create negative samples for binary classification link prediction
    
    This function is kept for backward compatibility but not used in ranking approach.
    """
    
    num_pos = positive_edges.shape[1]
    num_neg = int(num_pos * negative_ratio)
    
    # Convert positive edges to set for fast lookup
    positive_set = set()
    for i in range(num_pos):
        edge = (positive_edges[0, i].item(), positive_edges[1, i].item())
        positive_set.add(edge)
        positive_set.add((edge[1], edge[0]))  # Add reverse edge
    
    # Generate negative samples
    torch.manual_seed(random_state)
    negative_edges = []
    
    max_attempts = num_neg * 10  # Prevent infinite loop
    attempts = 0
    
    while len(negative_edges) < num_neg and attempts < max_attempts:
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()
        
        # Skip self-loops and existing positive edges
        if src != dst and (src, dst) not in positive_set:
            negative_edges.append([src, dst])
        
        attempts += 1
    
    if len(negative_edges) < num_neg:
        print(f"Warning: Could only generate {len(negative_edges)} negative samples instead of {num_neg}")
    
    negative_edges_tensor = torch.tensor(negative_edges, dtype=torch.long).t()
    return negative_edges_tensor

def create_split_masks(pair_to_label, splits):
    """
    Create boolean masks for existing code that expects full datasets
    
    Args:
        pair_to_label: Original full pair_to_label dict
        splits: Data splits from create_data_splits()
    
    Returns:
        dict: Boolean masks for each split
    """
    
    all_pairs = list(pair_to_label.keys())
    train_pairs = set(splits['outcome_splits']['train'].keys())
    val_pairs = set(splits['outcome_splits']['val'].keys())
    test_pairs = set(splits['outcome_splits']['test'].keys())
    
    masks = {
        'train': [pair in train_pairs for pair in all_pairs],
        'val': [pair in val_pairs for pair in all_pairs],
        'test': [pair in test_pairs for pair in all_pairs]
    }
    
    return masks

# Example usage and testing
if __name__ == "__main__":
    # Example data
    pair_to_label = {
        (0, 100): 1, (1, 101): 0, (2, 102): 1, (3, 103): 0,
        (4, 104): 1, (5, 105): 0, (6, 106): 1, (7, 107): 0,
        (8, 108): 1, (9, 109): 0
    }
    
    transplant_edges = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    ])
    
    # Mock args
    class Args:
        test_ratio = 0.2
        val_ratio = 0.1
        negative_sampling_ratio = 1.0
        seed = 42
    
    args = Args()
    
    # Test ranking-based splitting
    splits = create_data_splits(pair_to_label, transplant_edges, 200, args)
    
    # Test getting split data
    train_outcome, train_pos, train_neg = get_split_data(splits, 'train')
    print(f"\nTrain split contains:")
    print(f"  Outcome pairs: {len(train_outcome)}")
    print(f"  Positive edges: {train_pos.shape[1]}")
    print(f"  Negative edges: {train_neg.shape[1]} (empty for ranking approach)")