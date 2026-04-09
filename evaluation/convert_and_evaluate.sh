#!/bin/bash

# RFTVRP Model Conversion and Evaluation Script
#
# Usage:
#   ./convert_and_evaluate.sh [OPTIONS]
#
# Options:
#   --experiment, -e        experiment name (required, e.g. "2025xxxx_xxxxxx")
#   --step, -s              checkpoint step (required, e.g. "global_step_700")
#   --base-model-path, -b   base model path (required, e.g. "/path/to/Qwen3-14B")
#   --experiment-path, -c   checkpoint root dir (default: ./checkpoints/$EXPERIMENT)
#   --problem-mode, -p      problem type (default: mtsp)
#     - tsp:  TSP problem
#     - mtsp: CVRP problem (multi-TSP)
#   --module-to-modify, -m  module to optimize (default: subpopulation)
#     - subpopulation: subpopulation operator
#     - crossover:     crossover operator
#   --num-samples, -n       number of operator samples (default: 64)
#   --iters, -i             HGS iteration budget (default: 800)
#   --num-procs, -np        parallel worker processes (default: 16)
#   --temperature, -t       sampling temperature (default: 1)
#   --top-p, -tp            top-p sampling (default: 0.95)
#   --top-k, -tk            top-k sampling (default: 200)
#   --rollout-rounds, -r    number of refinement rounds (default: 1)
#   --cuda-devices, -d      CUDA device id(s) (default: 0)
#
# Output:
#   - converted model: Ckpt_to_be_evaluated/<experiment>/<step>/
#   - log file:        eval_multiturn_<experiment>_<step>_iters<N>.log
#   - result file:     results_multi_round_<experiment>_<step>_iters<N>.json
#
# Example:
# ./convert_and_evaluate.sh \
#     --experiment <your_experiment_id> \
#     --step global_step_700 \
#     --base-model-path /path/to/Qwen3-14B \
#     --problem-mode mtsp \
#     --module-to-modify crossover \
#     --num-samples 64 \
#     --iters 800 \
#     --num-procs 16 \
#     --temperature 1 \
#     --top-p 0.95 \
#     --top-k 200 \
#     --rollout-rounds 1 \
#     --cuda-devices 0

# Check for help option
if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ "$1" = "help" ]; then
    echo "RFTVRP Model Conversion and Evaluation Script"
    echo ""
    echo "Usage:"
    echo "  ./convert_and_evaluate.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --experiment, -e        experiment name (required)"
    echo "  --step, -s              checkpoint step (required, e.g. global_step_700)"
    echo "  --base-model-path, -b   base model path (required)"
    echo "  --experiment-path, -c   checkpoint root (default: ./checkpoints/\$EXPERIMENT)"
    echo "  --problem-mode, -p      problem type (default: mtsp)"
    echo "  --module-to-modify, -m  module to optimize (default: subpopulation)"
    echo "  --num-samples, -n       operator samples (default: 64)"
    echo "  --iters, -i             HGS iteration budget (default: 800)"
    echo "  --num-procs, -np        parallel processes (default: 16)"
    echo "  --temperature, -t       sampling temperature (default: 1)"
    echo "  --top-p, -tp            top-p sampling (default: 0.95)"
    echo "  --top-k, -tk            top-k sampling (default: 200)"
    echo "  --rollout-rounds, -r    refinement rounds (default: 1)"
    echo "  --cuda-devices, -d      CUDA device id(s) (default: 0)"
    echo ""
    echo "Examples:"
    echo "  # Minimal example"
    echo "  ./convert_and_evaluate.sh \\"
    echo "      --experiment <exp_id> \\"
    echo "      --step global_step_700 \\"
    echo "      --base-model-path /path/to/Qwen3-14B"
    echo ""
    echo "  # Full example"
    echo "  ./convert_and_evaluate.sh \\"
    echo "      --experiment <exp_id> \\"
    echo "      --step global_step_700 \\"
    echo "      --base-model-path /path/to/Qwen3-14B \\"
    echo "      --problem-mode mtsp \\"
    echo "      --module-to-modify crossover \\"
    echo "      --num-samples 64 \\"
    echo "      --iters 800 \\"
    echo "      --num-procs 16 \\"
    echo "      --temperature 1 \\"
    echo "      --top-p 0.95 \\"
    echo "      --top-k 200 \\"
    echo "      --rollout-rounds 1 \\"
    echo "      --cuda-devices 0"
    echo ""
    echo "Output:"
    echo "  - converted model: Ckpt_to_be_evaluated/<experiment>/<step>/"
    echo "  - log file:        eval_multiturn_<experiment>_<step>_iters<N>.log"
    echo "  - result file:     results_multi_round_<experiment>_<step>_iters<N>.json"
    echo ""
    echo "Help:"
    echo "  -h, --help, help: show this message"
    exit 0
fi

# Default values
EXPERIMENT=""
STEP=""
BASE_MODEL_PATH=""
EXPERIMENT_PATH=""
PROBLEM_MODE="mtsp"
MODULE_TO_MODIFY="subpopulation"
NUM_SAMPLES=64
ITERS=800
NUM_PROCS=16
TEMPERATURE=1
TOP_P=0.95
TOP_K=200
ROLLOUT_ROUNDS=1
CUDA_DEVICES=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment|-e)
            EXPERIMENT="$2"
            shift 2
            ;;
        --step|-s)
            STEP="$2"
            shift 2
            ;;
        --base-model-path|-b)
            BASE_MODEL_PATH="$2"
            shift 2
            ;;
        --experiment-path|-c)
            EXPERIMENT_PATH="$2"
            shift 2
            ;;
        --problem-mode|-p)
            PROBLEM_MODE="$2"
            shift 2
            ;;
        --module-to-modify|-m)
            MODULE_TO_MODIFY="$2"
            shift 2
            ;;
        --num-samples|-n)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --iters|-i)
            ITERS="$2"
            shift 2
            ;;
        --num-procs|-np)
            NUM_PROCS="$2"
            shift 2
            ;;
        --temperature|-t)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p|-tp)
            TOP_P="$2"
            shift 2
            ;;
        --top-k|-tk)
            TOP_K="$2"
            shift 2
            ;;
        --rollout-rounds|-r)
            ROLLOUT_ROUNDS="$2"
            shift 2
            ;;
        --cuda-devices|-d)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Set EXPERIMENT_PATH default (if not provided)
if [ -z "$EXPERIMENT_PATH" ]; then
    if [ -z "$EXPERIMENT" ]; then
        echo "ERROR: --experiment is required (or provide --experiment-path directly)"
        exit 1
    fi
    EXPERIMENT_PATH="./checkpoints/$EXPERIMENT"
fi

# Set target directory
TARGET_DIR="Ckpt_to_be_evaluated/$EXPERIMENT/$STEP"

echo "=========================================="
echo "RFTVRP Model Conversion and Evaluation"
echo "=========================================="
echo "Experiment:       $EXPERIMENT"
echo "Step:             $STEP"
echo "Base model path:  $BASE_MODEL_PATH"
echo "Experiment path:  $EXPERIMENT_PATH"
echo "Target dir:       $TARGET_DIR"
echo "Problem mode:     $PROBLEM_MODE"
echo "Module:           $MODULE_TO_MODIFY"
echo "Num samples:      $NUM_SAMPLES"
echo "Iters:            $ITERS"
echo "Num procs:        $NUM_PROCS"
echo "Temperature:      $TEMPERATURE"
echo "Top-p:            $TOP_P"
echo "Top-k:            $TOP_K"
echo "Rollout rounds:   $ROLLOUT_ROUNDS"
echo "CUDA devices:     $CUDA_DEVICES"
echo "=========================================="

# Function: copy tokenizer files from base model into checkpoint dir
copy_tokenizer_files() {
    local ckpt_path=$1
    local base_model_path=$2
    local files_to_copy=(
        "added_tokens.json"
        "config.json"
        "generation_config.json"
        "special_tokens_map.json"
        "tokenizer_config.json"
        "tokenizer.json"
        "vocab.json"
    )
    if [ -f "$base_model_path/merges.txt" ]; then
        files_to_copy+=("merges.txt")
    fi

    # Create target directory if it does not exist
    if [ ! -d "$ckpt_path" ]; then
        mkdir -p "$ckpt_path"
        echo "Created checkpoint directory: $ckpt_path"
    else
        echo "Checkpoint directory already exists: $ckpt_path"
    fi

    # Copy each file
    for filename in "${files_to_copy[@]}"; do
        src="$base_model_path/$filename"
        dst="$ckpt_path/$filename"
        if [ -e "$src" ]; then
            cp "$src" "$dst"
            echo "Copied $src -> $dst"
        else
            echo "Warning: $src does not exist, skipping."
        fi
    done
}

# Validate inputs
echo "Validating inputs..."
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "ERROR: base model path does not exist: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -d "$EXPERIMENT_PATH/$STEP/actor" ]; then
    echo "ERROR: checkpoint path does not exist: $EXPERIMENT_PATH/$STEP/actor"
    exit 1
fi

# Step 1: Build environment
echo "=========================================="
echo "Step 1: Build environment"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/pyvrp"

echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "BUILD_DIR:  $BUILD_DIR"

cd "$BUILD_DIR"
rm -rf build
meson setup build -Doptimization=3
meson compile -C build
rm -rf build/_crossover*.so
rm -rf "$SCRIPT_DIR/../benchmark/pyvrp*"
cd ..

# Update gcc
conda install -c conda-forge -y gcc=12.1.0

# make sure pyvrp is not installed
pip uninstall -y pyvrp

# Step 2: Model conversion
echo "=========================================="
echo "Step 2: Model conversion"
echo "=========================================="

echo "Checking whether model has already been converted..."
CONVERSION_SKIPPED=false

if [ -d "$TARGET_DIR" ]; then
    echo "Target directory already exists: $TARGET_DIR"

    REQUIRED_FILES=(
        "config.json"
        "generation_config.json"
        "tokenizer.json"
        "tokenizer_config.json"
    )

    SAFETENSORS_FILES=$(find "$TARGET_DIR" -name "*.safetensors" 2>/dev/null | wc -l)

    ALL_FILES_EXIST=true
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$TARGET_DIR/$file" ]; then
            echo "Missing required file: $TARGET_DIR/$file"
            ALL_FILES_EXIST=false
            break
        fi
    done

    if [ "$SAFETENSORS_FILES" -gt 0 ] && [ "$ALL_FILES_EXIST" = true ]; then
        echo "✓ Found existing converted model:"
        echo "  - $SAFETENSORS_FILES safetensors file(s)"
        echo "  - all required config files present"
        echo "✓ Skipping model conversion"
        CONVERSION_SKIPPED=true
    else
        echo "⚠ Target directory exists but is incomplete; reconverting"
        [ "$SAFETENSORS_FILES" -eq 0 ] && echo "  - no safetensors files found"
        [ "$ALL_FILES_EXIST" = false ]  && echo "  - missing required config files"
    fi
else
    echo "Target directory does not exist; conversion required."
fi

if [ "$CONVERSION_SKIPPED" = true ]; then
    echo "Model conversion skipped."
else
    pip install transformers==4.56.1

    CONVERT_SCRIPT="$SCRIPT_DIR/verl/scripts/convert_pt_to_safetensors.py"
    if [ ! -f "$CONVERT_SCRIPT" ]; then
        echo "ERROR: conversion script not found: $CONVERT_SCRIPT"
        exit 1
    fi

    echo "Starting model conversion..."
    python "$CONVERT_SCRIPT" \
        --backend fsdp \
        --hf_model_path "$BASE_MODEL_PATH" \
        --local_dir "$EXPERIMENT_PATH/$STEP/actor" \
        --target_dir "$TARGET_DIR"

    if [ -d "$EXPERIMENT_PATH/$STEP/actor/huggingface" ]; then
        echo "Copying HuggingFace config files..."
        cp "$EXPERIMENT_PATH/$STEP/actor/huggingface"/* "$TARGET_DIR/"
        echo "HuggingFace files copied."
    else
        echo "Warning: huggingface directory not found: $EXPERIMENT_PATH/$STEP/actor/huggingface"
        echo "Falling back to copying tokenizer files from base model..."
        copy_tokenizer_files "$TARGET_DIR" "$BASE_MODEL_PATH"
    fi

    echo "Model conversion complete: $TARGET_DIR"
fi

# Step 3: Evaluation
echo "=========================================="
echo "Step 3: Evaluation"
echo "=========================================="

if [ "$PROBLEM_MODE" = "tsp" ]; then
    PROBLEM_TYPES="['TSP']"
    CROSSOVER_TYPE="tsp"
elif [ "$PROBLEM_MODE" = "mtsp" ]; then
    PROBLEM_TYPES="['CVRP_all']"
    CROSSOVER_TYPE="mtsp"
else
    echo "ERROR: unknown problem mode '$PROBLEM_MODE'. Use 'tsp' or 'mtsp'."
    exit 1
fi

echo "PROBLEM_TYPES:  $PROBLEM_TYPES"
echo "CROSSOVER_TYPE: $CROSSOVER_TYPE"

echo "Starting evaluation..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python vllm_evaluate.py \
    "$TARGET_DIR" \
    --multi_model \
    -n $NUM_SAMPLES \
    --problem_types "$PROBLEM_TYPES" \
    --iters $ITERS \
    --num_procs $NUM_PROCS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --output_file "results_multi_round_${EXPERIMENT}_${STEP}_iters${ITERS}.json" \
    --rollout_rounds $ROLLOUT_ROUNDS \
    --module_to_modify $MODULE_TO_MODIFY \
    --crossover_type $CROSSOVER_TYPE \
    2>&1 | tee "eval_multiturn_${EXPERIMENT}_${STEP}_iters${ITERS}.log"

echo "=========================================="
echo "Done!"
echo "Converted model: $TARGET_DIR"
echo "Log file:        eval_multiturn_${EXPERIMENT}_${STEP}_iters${ITERS}.log"
echo "Result file:     results_multi_round_${EXPERIMENT}_${STEP}_iters${ITERS}.json"
echo "=========================================="
