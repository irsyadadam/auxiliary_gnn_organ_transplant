#create_pyg_graph.py

import torch
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def create_transplant_edges(original_data, donor_node_mapping, recipient_node_mapping):
    """Create donor-recipient transplant edges based on shared CASEID"""
    
    source_nodes = []
    target_nodes = []
    
    for _, row in original_data.iterrows():
        caseid = row['CASEID']
        if caseid in donor_node_mapping and caseid in recipient_node_mapping:
            donor_node = donor_node_mapping[caseid]
            recipient_node = recipient_node_mapping[caseid]
            source_nodes.append(donor_node)
            target_nodes.append(recipient_node)
    
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    print(f"Created {edge_index.shape[1]} transplant edges")
    return edge_index

def create_similarity_edges(feature_matrix, offset, k=10, metric='euclidean'):
    """Create similarity edges using k-NN within a population"""
    
    n_nodes = feature_matrix.shape[0]
    
    if n_nodes == 0:
        return torch.tensor([[], []], dtype=torch.long)
    
    if k >= n_nodes:
        k = n_nodes - 1
        print(f"Warning: k ({k+1}) >= number of nodes ({n_nodes}), setting k to {k}")
    
    # Fit k-NN
    nn_model = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1)
    nn_model.fit(feature_matrix)
    _, neighbors = nn_model.kneighbors(feature_matrix)
    
    # Build edges (excluding self-connections)
    source_nodes = []
    target_nodes = []
    
    for i in range(n_nodes):
        for j in neighbors[i, 1:]:  # Skip first neighbor (self)
            source_nodes.append(i + offset)
            target_nodes.append(j + offset)
    
    # Make edges bidirectional
    all_sources = source_nodes + target_nodes
    all_targets = target_nodes + source_nodes
    
    edge_index = torch.tensor([all_sources, all_targets], dtype=torch.long)
    
    print(f"Created {edge_index.shape[1]} similarity edges for {n_nodes} nodes (k={k}, offset={offset})")
    return edge_index

def build_heterogeneous_graph(preprocessed_data, k_neighbors=10, similarity_metric='euclidean'):
    """
    Build a heterogeneous graph from preprocessed transplant data.
    
    Args:
        preprocessed_data: Dictionary from data_preprocess.preprocess_transplant_data()
        k_neighbors: Number of nearest neighbors for similarity networks
        similarity_metric: Distance metric for similarity ('euclidean', 'cosine', etc.)
        
    Returns:
        HeteroData: PyTorch Geometric heterogeneous graph
        dict: Edge statistics and metadata
    """
    
    print("="*60)
    print("BUILDING HETEROGENEOUS GRAPH")
    print("="*60)
    
    # Extract data components
    donor_features = preprocessed_data['donor_features']
    recipient_features = preprocessed_data['recipient_features']
    unified_feature_matrix = preprocessed_data['unified_feature_matrix']
    donor_node_mapping = preprocessed_data['donor_node_mapping']
    recipient_node_mapping = preprocessed_data['recipient_node_mapping']
    original_data = preprocessed_data['original_data']
    
    n_donors = len(donor_features)
    n_recipients = len(recipient_features)
    
    print(f"Graph will have {n_donors + n_recipients} total nodes")
    print(f"Using k={k_neighbors} neighbors for similarity networks")
    
    # Create all edge types
    edges = {}
    
    # 1. Donor-Recipient transplant edges
    print("\n1. Creating transplant edges...")
    edges['transplant'] = create_transplant_edges(original_data, donor_node_mapping, recipient_node_mapping)
    
    # 2. Donor-Donor similarity edges
    print("\n2. Creating donor similarity edges...")
    donor_feat_matrix = donor_features.drop(['CASEID'], axis=1).values
    edges['donor_similarity'] = create_similarity_edges(
        donor_feat_matrix, 
        offset=0,  # donor nodes start at 0
        k=k_neighbors,
        metric=similarity_metric
    )
    
    # 3. Recipient-Recipient similarity edges  
    print("\n3. Creating recipient similarity edges...")
    recipient_feat_matrix = recipient_features.drop(['CASEID'], axis=1).values
    edges['recipient_similarity'] = create_similarity_edges(
        recipient_feat_matrix, 
        offset=n_donors,  # recipient nodes start at n_donors
        k=k_neighbors,
        metric=similarity_metric
    )
    
    # Build heterogeneous graph
    print("\n4. Building heterogeneous graph structure...")
    hetero_data = HeteroData()
    node_type = "patient"
    
    # Add node features
    hetero_data[node_type].x = torch.tensor(unified_feature_matrix, dtype=torch.float32)
    
    # Add edge types with descriptive relation names
    hetero_data[node_type, 'transplant', node_type].edge_index = edges['transplant']
    hetero_data[node_type, 'donor_similarity', node_type].edge_index = edges['donor_similarity']  
    hetero_data[node_type, 'recipient_similarity', node_type].edge_index = edges['recipient_similarity']
    
    # Create edge statistics
    edge_stats = {
        'n_donors': n_donors,
        'n_recipients': n_recipients,
        'total_nodes': n_donors + n_recipients,
        'transplant_edges': edges['transplant'].shape[1],
        'donor_similarity_edges': edges['donor_similarity'].shape[1],
        'recipient_similarity_edges': edges['recipient_similarity'].shape[1],
        'total_edges': sum(edge.shape[1] for edge in edges.values()),
        'k_neighbors': k_neighbors,
        'similarity_metric': similarity_metric
    }
    
    print(f"\nHeterogeneous graph created successfully!")
    print(f"  Nodes: {edge_stats['total_nodes']}")
    print(f"  Total edges: {edge_stats['total_edges']}")
    print(f"  Features per node: {unified_feature_matrix.shape[1]}")
    
    return hetero_data, edge_stats

def convert_to_homogeneous(hetero_data, edge_stats):
    """Convert heterogeneous graph to homogeneous format"""
    
    print("\n" + "="*60)
    print("CONVERTING TO HOMOGENEOUS GRAPH")
    print("="*60)
    
    # Convert to homogeneous
    homo_data = hetero_data.to_homogeneous()
    
    # Set number of relations and create mapping
    homo_data.num_relations = 3
    
    relation_mapping = {
        'transplant': 0,
        'donor_similarity': 1, 
        'recipient_similarity': 2
    }
    
    print(f"Homogeneous graph created:")
    print(f"  Nodes: {homo_data.x.shape[0]}")
    print(f"  Edges: {homo_data.edge_index.shape[1]}")
    print(f"  Features: {homo_data.x.shape[1]}")
    print(f"  Relations: {homo_data.num_relations}")
    print(f"  Edge types: {list(relation_mapping.keys())}")
    
    return homo_data, relation_mapping

def create_transplant_graph(preprocessed_data, 
                          k_neighbors=10, 
                          similarity_metric='euclidean',
                          return_homogeneous=True):
    """
    Main function to create transplant knowledge graph.
    
    Args:
        preprocessed_data: Dictionary from data_preprocess.preprocess_transplant_data()
        k_neighbors: Number of nearest neighbors for similarity networks
        similarity_metric: Distance metric for similarity ('euclidean', 'cosine', etc.)
        return_homogeneous: If True, returns homogeneous graph; if False, returns heterogeneous
        
    Returns:
        If return_homogeneous=True:
            - homo_data: Homogeneous PyTorch Geometric graph
            - relation_mapping: Dict mapping relation names to indices
            - edge_stats: Dictionary with graph statistics
            
        If return_homogeneous=False:
            - hetero_data: Heterogeneous PyTorch Geometric graph  
            - edge_stats: Dictionary with graph statistics
    """
    
    # Build heterogeneous graph
    hetero_data, edge_stats = build_heterogeneous_graph(
        preprocessed_data, 
        k_neighbors=k_neighbors,
        similarity_metric=similarity_metric
    )
    
    if return_homogeneous:
        # Convert to homogeneous and return
        homo_data, relation_mapping = convert_to_homogeneous(hetero_data, edge_stats)
        
        print("\n" + "="*60)
        print("GRAPH CREATION COMPLETE")
        print("="*60)
        
        return homo_data, relation_mapping, edge_stats
    else:
        # Return heterogeneous graph
        print("\n" + "="*60)
        print("GRAPH CREATION COMPLETE")
        print("="*60)
        
        return hetero_data, edge_stats

def create_graph_full_pipeline(csv_path, 
                        outcome_variable='po_anycomp',
                        k_neighbors=10,
                        similarity_metric='euclidean', 
                        return_homogeneous=True):
    """
    Complete pipeline from CSV to graph (combines preprocessing + graph creation).
    
    Args:
        csv_path: Path to CSV file
        outcome_variable: Name of outcome variable to predict
        k_neighbors: Number of nearest neighbors for similarity networks
        similarity_metric: Distance metric for similarity  
        return_homogeneous: Whether to return homogeneous or heterogeneous graph
        
    Returns:
        Tuple containing graph, mappings, and statistics
    """
    
    # Import here to avoid circular imports
    from data_preprocess import preprocess_transplant_data
    
    # Preprocess data
    preprocessed_data = preprocess_transplant_data(csv_path, outcome_variable)
    
    # Create graph
    graph_result = create_transplant_graph(
        preprocessed_data,
        k_neighbors=k_neighbors,
        similarity_metric=similarity_metric,
        return_homogeneous=return_homogeneous
    )
    
    # Add pair_to_label to results
    pair_to_label = preprocessed_data['pair_to_label']
    
    if return_homogeneous:
        homo_data, relation_mapping, edge_stats = graph_result
        return homo_data, pair_to_label, relation_mapping, edge_stats
    else:
        hetero_data, edge_stats = graph_result
        return hetero_data, pair_to_label, edge_stats

# Example usage:
if __name__ == "__main__":
    from data_preprocess import preprocess_transplant_data
    
    # Example 1: Step by step
    print("Example 1: Step-by-step approach")
    preprocessed_data = preprocess_transplant_data('your_data.csv', 'po_anycomp')
    
    # Get heterogeneous graph
    hetero_data, edge_stats = create_transplant_graph(
        preprocessed_data, 
        k_neighbors=10,
        return_homogeneous=False
    )
    
    # Get homogeneous graph  
    homo_data, relation_mapping, edge_stats = create_transplant_graph(
        preprocessed_data,
        k_neighbors=10, 
        return_homogeneous=True
    )
    
    print("\nExample 2: Full pipeline")
    # Example 2: Full pipeline in one call
    homo_data, pair_to_label, relation_mapping, edge_stats = create_graph_full_pipeline(
        csv_path='your_data.csv',
        outcome_variable='po_anycomp',
        k_neighbors=10,
        return_homogeneous=True
    )