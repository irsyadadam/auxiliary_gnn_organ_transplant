#data_preprocess.py

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def load_data(csv_path):
    """Load data from CSV file with basic validation"""
    try:
        data = pd.read_csv(csv_path)
        print(f"Loaded data with {len(data)} rows and {len(data.columns)} columns")
        return data
    except Exception as e:
        raise ValueError(f"Error loading data from {csv_path}: {str(e)}")

def validate_outcome_variable(data, outcome_variable):
    """Validate that the outcome variable exists in the data"""
    if outcome_variable not in data.columns:
        available_outcomes = [col for col in data.columns if any(keyword in col.lower() 
                             for keyword in ['outcome', 'comp', 'readm', 'mort', 'reject'])]
        raise ValueError(f"Outcome variable '{outcome_variable}' not found in data.\n"
                        f"Available potential outcome columns: {available_outcomes}")
    
    print(f"Using outcome variable: '{outcome_variable}'")
    print(f"Outcome distribution:\n{data[outcome_variable].value_counts()}")

def extract_features(data, outcome_variable):
    """Extract and clean donor and recipient features, excluding ID and outcome columns"""
    
    # Validate outcome variable exists
    validate_outcome_variable(data, outcome_variable)
    
    # Columns to exclude from features
    excluded_cols = {'CASEID', outcome_variable}
    
    # Separate donor and recipient columns
    donor_cols = [col for col in data.columns if col.startswith('d_') and col not in excluded_cols]
    recipient_cols = [col for col in data.columns if col.startswith('r_') and col not in excluded_cols]
    
    print(f"Found {len(donor_cols)} donor features and {len(recipient_cols)} recipient features")
    
    # Extract features
    donor_features = data[['CASEID'] + donor_cols].copy()
    recipient_features = data[['CASEID'] + recipient_cols].copy()
    
    # Clean column names (remove prefixes)
    donor_features.columns = ['CASEID'] + [col[2:] for col in donor_cols]  # Remove 'd_'
    recipient_features.columns = ['CASEID'] + [col[2:] for col in recipient_cols]  # Remove 'r_'
    
    # Convert to numeric and handle missing values
    for df_name, df in [('donor', donor_features), ('recipient', recipient_features)]:
        numeric_conversions = 0
        for col in df.columns:
            if col != 'CASEID' and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_conversions += 1
        
        # Fill missing values with 0
        missing_before = df.isnull().sum().sum()
        df.fillna(0, inplace=True)
        
        print(f"{df_name.capitalize()} features: {numeric_conversions} columns converted to numeric, "
              f"{missing_before} missing values filled with 0")
    
    return donor_features, recipient_features

def create_node_mappings(donor_features, recipient_features):
    """Create node ID mappings for donors and recipients"""
    
    n_donors = len(donor_features)
    n_recipients = len(recipient_features)
    
    # Donors get indices 0 to n_donors-1
    # Recipients get indices n_donors to n_donors+n_recipients-1
    donor_node_mapping = {caseid: i for i, caseid in enumerate(donor_features['CASEID'])}
    recipient_node_mapping = {caseid: i + n_donors for i, caseid in enumerate(recipient_features['CASEID'])}
    
    print(f"Created node mappings: {n_donors} donors (indices 0-{n_donors-1}), "
          f"{n_recipients} recipients (indices {n_donors}-{n_donors+n_recipients-1})")
    
    return donor_node_mapping, recipient_node_mapping

def create_unified_features(donor_features, recipient_features):
    """Create unified feature matrix with consistent columns across donors and recipients"""
    
    # Get all unique feature columns (excluding CASEID)
    donor_feat_cols = set(donor_features.columns) - {'CASEID'}
    recipient_feat_cols = set(recipient_features.columns) - {'CASEID'}
    all_feat_cols = sorted(donor_feat_cols | recipient_feat_cols)
    
    print(f"Unified feature matrix will have {len(all_feat_cols)} features")
    
    # Show feature overlap statistics  
    shared_features = donor_feat_cols & recipient_feat_cols
    donor_only = donor_feat_cols - recipient_feat_cols
    recipient_only = recipient_feat_cols - donor_feat_cols
    
    print(f"Feature overlap: {len(shared_features)} shared, {len(donor_only)} donor-only, {len(recipient_only)} recipient-only")
    
    # Align features for both donor and recipient
    donor_aligned = donor_features.reindex(columns=['CASEID'] + all_feat_cols, fill_value=0)
    recipient_aligned = recipient_features.reindex(columns=['CASEID'] + all_feat_cols, fill_value=0)
    
    # Combine feature matrices (donors first, then recipients)
    donor_matrix = donor_aligned.drop(['CASEID'], axis=1).values.astype(np.float32)
    recipient_matrix = recipient_aligned.drop(['CASEID'], axis=1).values.astype(np.float32)
    
    unified_matrix = np.vstack([donor_matrix, recipient_matrix])
    
    print(f"Created unified feature matrix: {unified_matrix.shape}")
    
    return unified_matrix, all_feat_cols

def create_outcome_mapping(data, donor_node_mapping, recipient_node_mapping, outcome_variable):
    """Create mapping from (donor_node, recipient_node) pairs to outcomes"""
    
    pair_to_label = {}
    missing_mappings = 0
    
    for _, row in data.iterrows():
        caseid = row['CASEID']
        if caseid in donor_node_mapping and caseid in recipient_node_mapping:
            donor_node = donor_node_mapping[caseid]
            recipient_node = recipient_node_mapping[caseid]
            outcome = row[outcome_variable]
            pair_to_label[(donor_node, recipient_node)] = outcome
        else:
            missing_mappings += 1
    
    print(f"Created outcome mapping: {len(pair_to_label)} valid pairs, {missing_mappings} cases with missing node mappings")
    
    return pair_to_label

def preprocess_transplant_data(csv_path, outcome_variable='po_anycomp'):
    """
    Main preprocessing function that handles all data preparation steps.
    
    Args:
        csv_path: Path to the CSV file containing transplant data
        outcome_variable: Name of the outcome variable column to predict
        
    Returns:
        dict: Dictionary containing all preprocessed data components
            - 'donor_features': DataFrame with donor features
            - 'recipient_features': DataFrame with recipient features  
            - 'unified_feature_matrix': Numpy array with combined features
            - 'feature_names': List of feature column names
            - 'donor_node_mapping': Dict mapping CASEID to donor node indices
            - 'recipient_node_mapping': Dict mapping CASEID to recipient node indices
            - 'pair_to_label': Dict mapping (donor_node, recipient_node) to outcomes
            - 'original_data': Original DataFrame for reference
    """
    
    print("="*60)
    print("TRANSPLANT DATA PREPROCESSING")
    print("="*60)
    
    # Load data
    data = load_data(csv_path)
    
    # Extract and clean features
    donor_features, recipient_features = extract_features(data, outcome_variable)
    
    # Create node mappings
    donor_node_mapping, recipient_node_mapping = create_node_mappings(donor_features, recipient_features)
    
    # Create unified feature matrix
    unified_feature_matrix, feature_names = create_unified_features(donor_features, recipient_features)
    
    # Create outcome mapping
    pair_to_label = create_outcome_mapping(data, donor_node_mapping, recipient_node_mapping, outcome_variable)
    
    print("="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    
    return {
        'donor_features': donor_features,
        'recipient_features': recipient_features,
        'unified_feature_matrix': unified_feature_matrix,
        'feature_names': feature_names,
        'donor_node_mapping': donor_node_mapping,
        'recipient_node_mapping': recipient_node_mapping,
        'pair_to_label': pair_to_label,
        'original_data': data
    }

