#!/usr/bin/env bash
# run.sh - Main training script with comprehensive argument support

set -e  # Exit on any error

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pyg_CUDA

# Set GPU visibility (modify as needed - single GPU for graph training)
export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Generate unique experiment name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="kidney_transplant_${TIMESTAMP}"

# Directories
BASE_DIR="experiments"
CHECKPOINT_DIR="${BASE_DIR}/${EXPERIMENT_NAME}/checkpoints"
RESULTS_DIR="${BASE_DIR}/${EXPERIMENT_NAME}/results"

# Create directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$RESULTS_DIR"

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

# Data Configuration
DATA_PATH="data/po_anycomp.csv"
OUTCOME_VARIABLE="po_anycomp"  # Options: po_anycomp, readm30daydc

# Graph Construction
K_NEIGHBORS=10
SIMILARITY_METRIC="euclidean"  # Options: euclidean, cosine, manhattan

# Model Architecture
GNN_TYPE="rgcn"          # Options: rgcn, rgat
DECODER_TYPE="distmult"  # Options: distmult, transe
HIDDEN_DIM=128
RELATION_DIM=128
NUM_LAYERS=2
DROPOUT=0.2

# Training Configuration
EPOCHS=100
BATCH_SIZE=256
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-4
SCHEDULER="step"  # Options: cosine, step, plateau, none

# Multi-task Learning
ALPHA=0.7  # Weight for outcome prediction (1-alpha for link prediction)
NEGATIVE_SAMPLING_RATIO=1.0

# Data Splitting
TEST_RATIO=0.2
VAL_RATIO=0.1

# Optimization
EARLY_STOPPING=20
GRADIENT_CLIPPING=1.0

# Logging
LOG_INTERVAL=1
SAVE_INTERVAL=20
SEED=42

# =============================================================================
# LOGGING SETUP
# =============================================================================

echo "=============================================================="
echo "KIDNEY TRANSPLANT GNN EXPERIMENT"
echo "=============================================================="
echo "Experiment Name: $EXPERIMENT_NAME"
echo "Started at: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Results: $RESULTS_DIR"
echo ""
echo "Configuration:"
echo "  Data: $DATA_PATH"
echo "  Outcome: $OUTCOME_VARIABLE"
echo "  Model: $GNN_TYPE + $DECODER_TYPE"
echo "  Hidden Dim: $HIDDEN_DIM"
echo "  Epochs: $EPOCHS"
echo "  Multi-task Alpha: $ALPHA"
echo "=============================================================="

# Log configuration to file
LOG_FILE="${RESULTS_DIR}/experiment_config.txt"
{
    echo "Experiment: $EXPERIMENT_NAME"
    echo "Started: $(date)"
    echo "Command: $0 $*"
    echo ""
    echo "Configuration:"
    echo "  DATA_PATH=$DATA_PATH"
    echo "  OUTCOME_VARIABLE=$OUTCOME_VARIABLE"
    echo "  GNN_TYPE=$GNN_TYPE"
    echo "  DECODER_TYPE=$DECODER_TYPE"
    echo "  HIDDEN_DIM=$HIDDEN_DIM"
    echo "  EPOCHS=$EPOCHS"
    echo "  ALPHA=$ALPHA"
    echo "  K_NEIGHBORS=$K_NEIGHBORS"
    echo "  LEARNING_RATE=$LEARNING_RATE"
} > "$LOG_FILE"

# =============================================================================
# MAIN TRAINING COMMAND
# =============================================================================

echo ""
echo "Starting training..."
echo ""

python main.py \
    --mode train \
    \
    --data_path "$DATA_PATH" \
    --outcome_variable "$OUTCOME_VARIABLE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --results_dir "$RESULTS_DIR" \
    \
    --k_neighbors $K_NEIGHBORS \
    --similarity_metric "$SIMILARITY_METRIC" \
    --use_homogeneous \
    \
    --gnn_type "$GNN_TYPE" \
    --decoder_type "$DECODER_TYPE" \
    --hidden_dim $HIDDEN_DIM \
    --relation_dim $RELATION_DIM \
    --num_layers $NUM_LAYERS \
    --dropout $DROPOUT \
    \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --scheduler "$SCHEDULER" \
    \
    --alpha $ALPHA \
    --use_link_prediction \
    --negative_sampling_ratio $NEGATIVE_SAMPLING_RATIO \
    \
    --test_ratio $TEST_RATIO \
    --val_ratio $VAL_RATIO \
    --stratify \
    \
    --early_stopping $EARLY_STOPPING \
    --gradient_clipping $GRADIENT_CLIPPING \
    \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --seed $SEED \
    --verbose

# =============================================================================
# POST-TRAINING SUMMARY
# =============================================================================

TRAINING_EXIT_CODE=$?

echo ""
echo "=============================================================="
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ TRAINING COMPLETED SUCCESSFULLY"
else
    echo "❌ TRAINING FAILED (Exit code: $TRAINING_EXIT_CODE)"
fi
echo "=============================================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Completed at: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo ""

# Show key result files if they exist
if [ -f "${RESULTS_DIR}/evaluation_results.json" ]; then
    echo "📊 Evaluation results available:"
    echo "   ${RESULTS_DIR}/evaluation_results.json"
fi

if [ -f "${RESULTS_DIR}/evaluation_plots.png" ]; then
    echo "📈 Plots saved to:"
    echo "   ${RESULTS_DIR}/evaluation_plots.png"
fi

if [ -f "${CHECKPOINT_DIR}/best_model.pt" ]; then
    echo "💾 Best model saved:"
    echo "   ${CHECKPOINT_DIR}/best_model.pt"
fi

echo ""
echo "To evaluate this model later, run:"
echo "   ./run_eval.sh $EXPERIMENT_NAME"
echo ""

# =============================================================================
# CLEANUP AND SUGGESTIONS
# =============================================================================

# Clean up GPU memory
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true

# Log completion
echo "Experiment completed: $EXPERIMENT_NAME" >> "${BASE_DIR}/experiment_log.txt"

# =============================================================================
# ALTERNATIVE CONFIGURATIONS (COMMENTED EXAMPLES)
# =============================================================================

# Uncomment sections below to try different configurations:

: <<'ALTERNATIVE_CONFIGS'

# ============== CONFIGURATION 1: RGCN + TransE ==============
# GNN_TYPE="rgcn"
# DECODER_TYPE="transe"
# ALPHA=0.8

# ============== CONFIGURATION 2: RGAT + DistMult ==============
# GNN_TYPE="rgat"
# DECODER_TYPE="distmult"
# HIDDEN_DIM=256
# NUM_LAYERS=3

# ============== CONFIGURATION 3: High Multi-task Weight ==============
# ALPHA=0.5  # Equal weight to both tasks
# NEGATIVE_SAMPLING_RATIO=2.0
# USE_LINK_PREDICTION="--use_link_prediction"

# ============== CONFIGURATION 4: Different Outcome ==============
# OUTCOME_VARIABLE="readm30daydc"
# DATA_PATH="data/readmission.csv"

# ============== CONFIGURATION 5: More Neighbors ==============
# K_NEIGHBORS=20
# SIMILARITY_METRIC="cosine"

# ============== CONFIGURATION 6: Longer Training ==============
# EPOCHS=200
# LEARNING_RATE=0.0005
# SCHEDULER="plateau"
# EARLY_STOPPING=30

ALTERNATIVE_CONFIGS

exit $TRAINING_EXIT_CODE