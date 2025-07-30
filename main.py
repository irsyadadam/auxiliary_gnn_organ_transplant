import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric import transforms as T
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
import random

warnings.filterwarnings("ignore")

# Import modularized components
from models.gnn_encoder import RGCNEncoder
from models.decoder import DistMultDecoder, TransEDecoder
from models.classifier import MLPClassifier
from models.joint_model import JointModel

# Import our custom modules - FIXED PATHS
from utils.data_processing import preprocess_transplant_data
from utils.create_pyg_graph import create_graph_full_pipeline
from train_utils.train_pass import train_model, load_checkpoint, setup_logging, evaluate_with_cross_validation
from train_utils.eval_pass import evaluate_model, print_evaluation_summary
from utils.data_splitting import create_data_splits

def parse_args():
    parser = argparse.ArgumentParser(description='Kidney Transplant Graph Neural Network Training')
    
    # Data and I/O
    parser.add_argument('--data_path', type=str, default='data/anycomp.csv', help='Path to input data CSV')
    parser.add_argument('--outcome_variable', type=str, default='po_anycomp', help='Target outcome variable to predict (e.g., po_anycomp, readm30daydc)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory to save model checkpoints')
    parser.add_argument('--results_dir', type=str, default='results/', help='Directory to save results and logs')
    
    # Graph construction
    parser.add_argument('--k_neighbors', type=int, default=10, help='Number of nearest neighbors for similarity networks')
    parser.add_argument('--similarity_metric', type=str, default='euclidean', choices=['euclidean', 'cosine', 'manhattan'], help='Distance metric for similarity networks')
    parser.add_argument('--use_homogeneous', action='store_true', default=True, help='Use homogeneous graph (default: True)')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size for GNN layers')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--gnn_type', type=str, default='rgcn', choices=['rgcn', 'rgat'], help='Type of GNN encoder to use')
    parser.add_argument('--decoder_type', type=str, default='distmult', choices=['distmult', 'transe'], help='Type of decoder for link prediction')
    parser.add_argument('--relation_dim', type=int, default=128, help='Dimension for relation embeddings in decoder')
    
    # Training
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'test'], help='Training mode')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'plateau', 'none'], help='Learning rate scheduler')
    
    # Multi-task learning
    parser.add_argument('--alpha', type=float, default=0.7, help='Weight for outcome prediction loss (1-alpha for link prediction)')
    parser.add_argument('--use_link_prediction', action='store_true', default=True, help='Use link prediction as auxiliary task')
    parser.add_argument('--negative_sampling_ratio', type=float, default=1.0, help='Ratio of negative samples for link prediction')
    
    # Data splitting and evaluation
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--stratify', action='store_true', default=True, help='Use stratified splitting based on outcomes')
    parser.add_argument('--cross_validation', type=int, default=0, help='Number of CV folds (0 for no CV)')
    
    # Optimization and regularization
    parser.add_argument('--early_stopping', type=int, default=20, help='Early stopping patience (0 to disable)')
    parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Gradient clipping threshold (0 to disable)')
    
    # Hardware and performance
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Computation device')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Logging and debugging
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval for training progress')
    parser.add_argument('--save_interval', type=int, default=20, help='Model checkpoint saving interval')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
    
    # Resume and evaluation
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true', default=False, help='Only run evaluation (requires --resume)')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_model(args, input_dim, num_relations):
    """Create model based on arguments"""
    
    # Create encoder
    if args.gnn_type == 'rgcn':
        encoder = RGCNEncoder(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_relations=num_relations,
            dropout=args.dropout
        )
    elif args.gnn_type == 'rgat':
        from models.gnn_encoder import RGATEncoder
        encoder = RGATEncoder(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_relations=num_relations,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unsupported GNN type: {args.gnn_type}")
    
    # Create decoder - FIXED INTERFACE
    if args.decoder_type == 'distmult':
        decoder = DistMultDecoder(
            num_relations=num_relations,
            hidden_dim=args.hidden_dim
        )
    elif args.decoder_type == 'transe':
        decoder = TransEDecoder(
            num_relations=num_relations,
            hidden_channels=args.hidden_dim
        )
    else:
        raise ValueError(f"Unsupported decoder type: {args.decoder_type}")
    
    # Create classifier - FIXED INTERFACE
    classifier = MLPClassifier(
        input_dim=args.hidden_dim * 2,  # Concatenated donor + recipient embeddings
        hidden_dim=args.hidden_dim,
        output_dim=1,  # Binary classification
        dropout=args.dropout
    )
    
    # Create joint model
    model = JointModel(
        gnn_encoder=encoder,
        decoder=decoder,
        classifier=classifier
    )
    
    return model

def print_model_config(args):
    """Print model configuration for verification"""
    print("="*60)
    print("MODEL CONFIGURATION")
    print("="*60)
    print(f"Data Path: {args.data_path}")
    print(f"Outcome Variable: {args.outcome_variable}")
    print(f"GNN Encoder: {args.gnn_type.upper()}")
    print(f"Decoder: {args.decoder_type.upper()}")
    print(f"Hidden Dimension: {args.hidden_dim}")
    print(f"Relation Dimension: {args.relation_dim}")
    print(f"Number of Layers: {args.num_layers}")
    print(f"Dropout: {args.dropout}")
    print(f"Multi-task Learning: {args.use_link_prediction}")
    print(f"Multi-task Alpha: {args.alpha}")
    print(f"K-Neighbors: {args.k_neighbors}")
    print(f"Similarity Metric: {args.similarity_metric}")
    print(f"Device: {args.device}")
    print("="*60)

def extract_transplant_edges(homo_data, relation_mapping):
    """Extract transplant edges from homogeneous graph"""
    # Find transplant relation edges
    transplant_relation_id = relation_mapping['transplant']
    transplant_mask = homo_data.edge_type == transplant_relation_id
    transplant_edges = homo_data.edge_index[:, transplant_mask]
    return transplant_edges

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Print configuration
    print_model_config(args)
    
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("LOADING AND PROCESSING DATA")
    print("="*60)
    
    # Load and process data using your existing pipeline - FIXED FUNCTION NAME
    homo_data, pair_to_label, relation_mapping, edge_stats = create_graph_full_pipeline(
        csv_path=args.data_path,
        outcome_variable=args.outcome_variable,
        k_neighbors=args.k_neighbors,
        similarity_metric=args.similarity_metric,
        return_homogeneous=True
    )
    
    # Extract mappings from preprocessing 
    preprocessed_data = preprocess_transplant_data(args.data_path, args.outcome_variable)
    donor_mapping = preprocessed_data['donor_node_mapping']
    recipient_mapping = preprocessed_data['recipient_node_mapping']
    
    # Extract transplant edges for link prediction
    transplant_edges = extract_transplant_edges(homo_data, relation_mapping)
    
    # Get graph dimensions
    input_dim = homo_data.x.shape[1]
    num_relations = homo_data.num_relations
    
    print(f"\nGraph loaded successfully:")
    print(f"  Nodes: {homo_data.x.shape[0]}")
    print(f"  Features: {input_dim}")
    print(f"  Total edges: {homo_data.edge_index.shape[1]}")
    print(f"  Transplant edges: {transplant_edges.shape[1]}")
    print(f"  Relations: {num_relations}")
    print(f"  Outcome pairs: {len(pair_to_label)}")
    
    # Create model
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    
    model = create_model(args, input_dim, num_relations)
    model = model.to(args.device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    if args.mode == 'train':
        print("\n" + "="*60)
        print("STARTING TRAINING WITH PROPER DATA SPLITS")
        print("="*60)
        
        # Load checkpoint if resuming
        if args.resume:
            print(f"Resuming training from: {args.resume}")
            epoch, metrics = load_checkpoint(args.resume, model)
            print(f"Resumed from epoch {epoch}")
        
        # Train with automatic data splitting
        trained_model, training_history = train_model(
            model=model,
            graph_data=homo_data,
            transplant_edges=transplant_edges,
            pair_to_label=pair_to_label,
            donor_mapping=donor_mapping,
            recipient_mapping=recipient_mapping,
            args=args
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED - EVALUATING ON ALL SPLITS")
        print("="*60)
        
        # Comprehensive evaluation on train/val/test
        results = evaluate_model(
            model=trained_model,
            graph_data=homo_data,
            transplant_edges=transplant_edges,
            pair_to_label=pair_to_label,
            donor_mapping=donor_mapping,
            recipient_mapping=recipient_mapping,
            args=args
        )
        
        # Print comprehensive summary
        print_evaluation_summary(results, args)
        
    elif args.mode == 'eval' or args.eval_only:
        # Evaluation mode
        if not args.resume:
            raise ValueError("Evaluation mode requires --resume with path to checkpoint")
        
        print(f"\nLoading model from: {args.resume}")
        epoch, metrics = load_checkpoint(args.resume, model)
        print(f"Loaded model from epoch {epoch}")
        
        print("\n" + "="*60)
        print("STARTING COMPREHENSIVE EVALUATION")
        print("="*60)
        
        # Evaluate on all splits (train/val/test)
        results = evaluate_model(
            model=model,
            graph_data=homo_data,
            transplant_edges=transplant_edges,
            pair_to_label=pair_to_label,
            donor_mapping=donor_mapping,
            recipient_mapping=recipient_mapping,
            args=args
        )
        
        # Print comprehensive summary
        print_evaluation_summary(results, args)
        
        # Cross-validation if requested
        if args.cross_validation > 1:
            print("\n" + "="*60)
            print("RUNNING CROSS-VALIDATION")
            print("="*60)
            
            def create_model_for_cv(input_dim, num_relations):
                encoder = RGCNEncoder(input_dim, args.hidden_dim, num_relations)
                decoder = DistMultDecoder(num_relations, args.hidden_dim)
                classifier = MLPClassifier(args.hidden_dim * 2, args.hidden_dim, 1, args.dropout)
                return JointModel(encoder, decoder, classifier)
            
            cv_results = evaluate_with_cross_validation(
                model_class=create_model_for_cv,
                graph_data=homo_data,
                transplant_edges=transplant_edges,
                pair_to_label=pair_to_label,
                donor_mapping=donor_mapping,
                recipient_mapping=recipient_mapping,
                args=args
            )
            
            if cv_results:
                print(f"Cross-validation completed:")
                print(f"  Outcome AUC: {cv_results['outcome_aucs']['mean']:.4f} ± {cv_results['outcome_aucs']['std']:.4f}")
                if args.use_link_prediction:
                    print(f"  Link AUC: {cv_results['link_aucs']['mean']:.4f} ± {cv_results['link_aucs']['std']:.4f}")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)
    print(f"Results saved to: {args.results_dir}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
    except Exception as e:
        print(f"\n\nExperiment failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()