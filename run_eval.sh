#!/usr/bin/env bash
# run_eval.sh - Evaluation script for trained models

set -e  # Exit on any error

# =============================================================================
# USAGE CHECK
# =============================================================================

if [ $# -ne 1 ]; then
    echo "Usage: $0 <experiment_name>"
    echo ""
    echo "Example: $0 kidney_transplant_20250729_143022"
    echo ""
    echo "Available experiments:"
    ls -1 experiments/ 2>/dev/null | grep kidney_transplant || echo "  No experiments found"
    exit 1
fi

EXPERIMENT_NAME="$1"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pyg_CUDA

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR="experiments"
EXPERIMENT_DIR="${BASE_DIR}/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"
ORIGINAL_RESULTS_DIR="${EXPERIMENT_DIR}/results"
EVAL_RESULTS_DIR="${EXPERIMENT_DIR}/evaluation_$(date +%Y%m%d_%H%M%S)"

# Check if experiment exists
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "❌ Error: Experiment '$EXPERIMENT_NAME' not found"
    echo "Available experiments:"
    ls -1 "$BASE_DIR" 2>/dev/null | grep kidney_transplant || echo "  No experiments found"
    exit 1
fi

# Check for model checkpoint
BEST_MODEL="${CHECKPOINT_DIR}/best_model.pt"
LATEST_MODEL="${CHECKPOINT_DIR}/latest_model.pt"

if [ -f "$BEST_MODEL" ]; then
    MODEL_PATH="$BEST_MODEL"
    MODEL_TYPE="best"
elif [ -f "$LATEST_MODEL" ]; then
    MODEL_PATH="$LATEST_MODEL"
    MODEL_TYPE="latest"
else
    echo "❌ Error: No model checkpoint found in $CHECKPOINT_DIR"
    echo "Looking for: best_model.pt or latest_model.pt"
    exit 1
fi

# Create evaluation results directory
mkdir -p "$EVAL_RESULTS_DIR"

# =============================================================================
# LOAD ORIGINAL CONFIGURATION
# =============================================================================

CONFIG_FILE="${ORIGINAL_RESULTS_DIR}/experiment_config.txt"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "⚠️  Warning: Original configuration not found. Using defaults."
    # Set defaults
    DATA_PATH="data/anycomp.csv"
    OUTCOME_VARIABLE="po_anycomp"
    GNN_TYPE="rgcn"
    DECODER_TYPE="distmult"
    HIDDEN_DIM=128
    RELATION_DIM=128
    NUM_LAYERS=2
    DROPOUT=0.2
    K_NEIGHBORS=10
    SIMILARITY_METRIC="euclidean"
    ALPHA=0.7
    TEST_RATIO=0.2
    VAL_RATIO=0.1
    SEED=42
else
    echo "📋 Loading configuration from: $CONFIG_FILE"
    # Extract configuration from original experiment
    DATA_PATH=$(grep "DATA_PATH=" "$CONFIG_FILE" | cut -d'=' -f2)
    OUTCOME_VARIABLE=$(grep "OUTCOME_VARIABLE=" "$CONFIG_FILE" | cut -d'=' -f2)
    GNN_TYPE=$(grep "GNN_TYPE=" "$CONFIG_FILE" | cut -d'=' -f2)
    DECODER_TYPE=$(grep "DECODER_TYPE=" "$CONFIG_FILE" | cut -d'=' -f2)
    HIDDEN_DIM=$(grep "HIDDEN_DIM=" "$CONFIG_FILE" | cut -d'=' -f2)
    K_NEIGHBORS=$(grep "K_NEIGHBORS=" "$CONFIG_FILE" | cut -d'=' -f2)
    ALPHA=$(grep "ALPHA=" "$CONFIG_FILE" | cut -d'=' -f2)
    
    # Set defaults for missing values
    RELATION_DIM=${RELATION_DIM:-$HIDDEN_DIM}
    NUM_LAYERS=${NUM_LAYERS:-2}
    DROPOUT=${DROPOUT:-0.2}
    SIMILARITY_METRIC=${SIMILARITY_METRIC:-euclidean}
    TEST_RATIO=${TEST_RATIO:-0.2}
    VAL_RATIO=${VAL_RATIO:-0.1}
    SEED=${SEED:-42}
fi

# =============================================================================
# EVALUATION SETUP
# =============================================================================

echo "=============================================================="
echo "KIDNEY TRANSPLANT GNN EVALUATION"
echo "=============================================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Model: $MODEL_PATH ($MODEL_TYPE)"
echo "Started at: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Results will be saved to: $EVAL_RESULTS_DIR"
echo ""
echo "Configuration:"
echo "  Data: $DATA_PATH"
echo "  Outcome: $OUTCOME_VARIABLE"
echo "  Model: $GNN_TYPE + $DECODER_TYPE"
echo "  Hidden Dim: $HIDDEN_DIM"
echo "  Alpha: $ALPHA"
echo "=============================================================="

# Log evaluation configuration
EVAL_LOG_FILE="${EVAL_RESULTS_DIR}/evaluation_config.txt"
{
    echo "Evaluation of: $EXPERIMENT_NAME"
    echo "Started: $(date)"
    echo "Model: $MODEL_PATH"
    echo "Command: $0 $*"
    echo ""
    echo "Configuration:"
    echo "  DATA_PATH=$DATA_PATH"
    echo "  OUTCOME_VARIABLE=$OUTCOME_VARIABLE"
    echo "  GNN_TYPE=$GNN_TYPE"
    echo "  DECODER_TYPE=$DECODER_TYPE"
    echo "  HIDDEN_DIM=$HIDDEN_DIM"
    echo "  ALPHA=$ALPHA"
} > "$EVAL_LOG_FILE"

# =============================================================================
# EVALUATION COMMANDS
# =============================================================================

echo ""
echo "🔍 Starting evaluation..."
echo ""

# Basic evaluation
echo "Running standard evaluation..."
python main.py \
    --mode eval \
    --eval_only \
    --resume "$MODEL_PATH" \
    \
    --data_path "$DATA_PATH" \
    --outcome_variable "$OUTCOME_VARIABLE" \
    --results_dir "$EVAL_RESULTS_DIR" \
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
    --alpha $ALPHA \
    --use_link_prediction \
    \
    --test_ratio $TEST_RATIO \
    --val_ratio $VAL_RATIO \
    --stratify \
    \
    --seed $SEED \
    --verbose

EVAL_EXIT_CODE=$?

# Optional: Cross-validation evaluation
echo ""
echo "❓ Run cross-validation? (y/N): "
read -r CROSS_VAL_RESPONSE

if [[ "$CROSS_VAL_RESPONSE" =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔄 Running 5-fold cross-validation..."
    python main.py \
        --mode eval \
        --eval_only \
        --resume "$MODEL_PATH" \
        \
        --data_path "$DATA_PATH" \
        --outcome_variable "$OUTCOME_VARIABLE" \
        --results_dir "${EVAL_RESULTS_DIR}/cross_validation" \
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
        --alpha $ALPHA \
        --use_link_prediction \
        \
        --cross_validation 5 \
        --test_ratio $TEST_RATIO \
        --val_ratio $VAL_RATIO \
        --stratify \
        \
        --seed $SEED \
        --verbose
    
    CV_EXIT_CODE=$?
else
    CV_EXIT_CODE=0
fi

# =============================================================================
# POST-EVALUATION SUMMARY
# =============================================================================

echo ""
echo "=============================================================="
if [ $EVAL_EXIT_CODE -eq 0 ] && [ $CV_EXIT_CODE -eq 0 ]; then
    echo "✅ EVALUATION COMPLETED SUCCESSFULLY"
else
    echo "❌ EVALUATION FAILED"
    echo "   Standard eval exit code: $EVAL_EXIT_CODE"
    echo "   Cross-validation exit code: $CV_EXIT_CODE"
fi
echo "=============================================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Model: $MODEL_TYPE"
echo "Completed at: $(date)"
echo "Results saved to: $EVAL_RESULTS_DIR"
echo ""

# Show generated files
echo "📁 Generated files:"
if [ -f "${EVAL_RESULTS_DIR}/evaluation_results.json" ]; then
    echo "   📊 Evaluation metrics: evaluation_results.json"
fi

if [ -f "${EVAL_RESULTS_DIR}/evaluation_plots.png" ]; then
    echo "   📈 Performance plots: evaluation_plots.png"
fi

if [ -f "${EVAL_RESULTS_DIR}/classification_report.json" ]; then
    echo "   📋 Classification report: classification_report.json"
fi

if [ -f "${EVAL_RESULTS_DIR}/cross_validation/cross_validation_results.json" ]; then
    echo "   🔄 Cross-validation: cross_validation/cross_validation_results.json"
fi

echo ""

# Quick results preview
if [ -f "${EVAL_RESULTS_DIR}/evaluation_results.json" ]; then
    echo "🎯 Quick Results Preview:"
    echo "=============================================================="
    python -c "
import json
try:
    with open('${EVAL_RESULTS_DIR}/evaluation_results.json', 'r') as f:
        results = json.load(f)
    outcome = results.get('outcome_prediction', {})
    print(f\"   Outcome AUC: {outcome.get('auc', 0):.4f}\")
    print(f\"   Outcome Accuracy: {outcome.get('accuracy', 0):.4f}\")
    print(f\"   Outcome F1: {outcome.get('f1', 0):.4f}\")
    
    if 'link_prediction' in results:
        link = results['link_prediction']
        print(f\"   Link AUC: {link.get('auc', 0):.4f}\")
        print(f\"   Link Accuracy: {link.get('accuracy', 0):.4f}\")
except Exception as e:
    print(f\"   Could not parse results: {e}\")
" 2>/dev/null || echo "   Results file not readable"
    echo "=============================================================="
fi

# Clean up GPU memory
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true

exit $((EVAL_EXIT_CODE + CV_EXIT_CODE))