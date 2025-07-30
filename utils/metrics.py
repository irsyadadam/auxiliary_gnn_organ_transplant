import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def get_link_prediction_metrics(link_logits, edge_label):
    """Compute link prediction metrics: MRR and Hits@10"""
    pos_mask = (edge_label == 1)
    pos_logits = link_logits[pos_mask]
    neg_logits = link_logits[~pos_mask]
    
    ranks = []
    for pos in pos_logits:
        # Combine with negative scores
        all_scores = torch.cat([pos.unsqueeze(0), neg_logits])
        # Get rank of positive sample
        rank = (all_scores >= pos).sum().item()
        ranks.append(rank)
    
    mrr = np.mean(1.0 / np.array(ranks))
    hits_at_100 = np.mean(np.array(ranks) <= 100)
    hits_at_10 = np.mean(np.array(ranks) <= 10)
    hits_at_1 = np.mean(np.array(ranks) <= 1)
    
    return {'mrr': mrr, 'hits@10': hits_at_10}

def get_classification_metrics(probs, labels):
    """Compute classification metrics: AUC and Accuracy"""
    if len(np.unique(labels)) < 2:
        return {'auc': 0.0, 'accuracy': 0.0}
    
    auc = roc_auc_score(labels, probs)
    preds = (probs > 0.5).astype(int)
    accuracy = accuracy_score(labels, preds)
    return {'auc': auc, 'accuracy': accuracy}