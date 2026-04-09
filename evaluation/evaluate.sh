#!/bin/bash

# RFTVRP Model Evaluation Script
#
# Usage:
#   ./evaluate.sh [PROBLEM_MODE] \
#                 [MODULE_TO_MODIFY] \
#                 [MODEL_TO_EVALUATE] \
#                 [NUM_SAMPLES] \
#                 [ITERS] \
#                 [NUM_PROCS] \
#                 [TEMPERATURE] \
#                 [TOP_P] \
#                 [TOP_K] \
#                 [ROLLOUT_ROUNDS] \
#                 [CUDA_DEVICES]
#
# Arguments:
#   PROBLEM_MODE: problem type (default: mtsp)
#     - tsp: TSP problem
#     - mtsp: CVRP problem (multi-TSP)
#
#   MODULE_TO_MODIFY: module to optimize (default: subpopulation)
#     - subpopulation: subpopulation operator
#     - crossover: crossover operator
#
#   MODEL_TO_EVALUATE: path to the model checkpoint (required)
#     - format: /path/to/model/<experiment_id>/global_step_*
#     - the log filename is derived from the experiment_id part of the path
#
#   NUM_SAMPLES:    number of generated operator samples (default: 64)
#   ITERS:          HGS iteration budget (default: 800)
#   NUM_PROCS:      number of parallel worker processes (default: 16)
#   TEMPERATURE:    sampling temperature (default: 1)
#   TOP_P:          top-p sampling parameter (default: 0.95)
#   TOP_K:          top-k sampling parameter (default: 200)
#   ROLLOUT_ROUNDS: number of refinement rollout rounds (default: 1)
#   CUDA_DEVICES:   CUDA device id(s) (default: 0)
#
# Examples:
#   # Default parameters
#   ./evaluate.sh
#
#   # Specify problem type and module
#   ./evaluate.sh tsp crossover
#
#   # Specify model path
#   ./evaluate.sh mtsp subpopulation /path/to/model/<experiment_id>/global_step_700
#
#   # Specify all parameters
#   ./evaluate.sh mtsp subpopulation /path/to/model/<experiment_id>/global_step_700 32 400 8 0.8 0.9 100 2 1
#
#   # Multi-GPU
#   ./evaluate.sh mtsp subpopulation /path/to/model/<experiment_id>/global_step_700 64 800 16 1 0.95 200 1 0,1,2,3
#
# Output:
#   - log file:    eval_multiturn_<experiment_id>.log
#   - result file: results_multi_round_<experiment_id>.json

# Setup directories
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/pyvrp"

echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "BUILD_DIR: $BUILD_DIR"

# Meson setup
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

# single model
# python vllm_evaluate.py ~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e -n 10 --problem_types "['CVRP_all']" --iters 800 --temperature 1 --top_p 1 --top_k 100

# Check for help option
if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ "$1" = "help" ]; then
    echo "RFTVRP Model Evaluation Script"
    echo ""
    echo "Usage:"
    echo "  ./evaluate.sh [PROBLEM_MODE] \\"
    echo "                [MODULE_TO_MODIFY] \\"
    echo "                [MODEL_TO_EVALUATE] \\"
    echo "                [NUM_SAMPLES] \\"
    echo "                [ITERS] \\"
    echo "                [NUM_PROCS] \\"
    echo "                [TEMPERATURE] \\"
    echo "                [TOP_P] \\"
    echo "                [TOP_K] \\"
    echo "                [ROLLOUT_ROUNDS] \\"
    echo "                [CUDA_DEVICES]"
    echo ""
    echo "Arguments:"
    echo "  PROBLEM_MODE: problem type (default: mtsp)"
    echo "    - tsp:  TSP problem"
    echo "    - mtsp: CVRP problem (multi-TSP)"
    echo ""
    echo "  MODULE_TO_MODIFY: module to optimize (default: subpopulation)"
    echo "    - subpopulation: subpopulation operator"
    echo "    - crossover:     crossover operator"
    echo ""
    echo "  MODEL_TO_EVALUATE: path to model checkpoint (required)"
    echo "    - format: /path/to/model/<experiment_id>/global_step_*"
    echo "    - the log filename is derived from the experiment_id part"
    echo ""
    echo "  NUM_SAMPLES:    number of operator samples (default: 64)"
    echo "  ITERS:          HGS iteration budget (default: 800)"
    echo "  NUM_PROCS:      parallel worker processes (default: 16)"
    echo "  TEMPERATURE:    sampling temperature (default: 1)"
    echo "  TOP_P:          top-p sampling (default: 0.95)"
    echo "  TOP_K:          top-k sampling (default: 200)"
    echo "  ROLLOUT_ROUNDS: refinement rounds (default: 1)"
    echo "  CUDA_DEVICES:   CUDA device id(s) (default: 0)"
    echo ""
    echo "Examples:"
    echo "  # Default parameters"
    echo "  ./evaluate.sh"
    echo ""
    echo "  # Specify problem type and module"
    echo "  ./evaluate.sh tsp crossover"
    echo ""
    echo "  # Specify model path"
    echo "  ./evaluate.sh mtsp subpopulation /path/to/model/<experiment_id>/global_step_700"
    echo ""
    echo "  # Specify all parameters"
    echo "  ./evaluate.sh mtsp subpopulation /path/to/model/<experiment_id>/global_step_700 32 400 8 0.8 0.9 100 2 1"
    echo ""
    echo "  # Multi-GPU"
    echo "  ./evaluate.sh mtsp subpopulation /path/to/model/<experiment_id>/global_step_700 64 800 16 1 0.95 200 1 0,1,2,3"
    echo ""
    echo "Output:"
    echo "  - log file:    eval_multiturn_<experiment_id>.log"
    echo "  - result file: results_multi_round_<experiment_id>.json"
    echo ""
    echo "Help:"
    echo "  -h, --help, help: show this message"
    exit 0
fi

# provide parent folder of your models
PROBLEM_MODE=${1:-mtsp} # tsp or mtsp
MODULE_TO_MODIFY=${2:-subpopulation} # subpopulation or crossover
MODEL_TO_EVALUATE=${3:-/path/to/your/model/checkpoint}
NUM_SAMPLES=${4:-64}
ITERS=${5:-800}
NUM_PROCS=${6:-16}
TEMPERATURE=${7:-1}
TOP_P=${8:-0.95}
TOP_K=${9:-200}
ROLLOUT_ROUNDS=${10:-1}
CUDA_DEVICES=${11:-0}

# Extract model identifier from MODEL_TO_EVALUATE path
# Extract the directory name before global_step_* (e.g., <experiment_id> from the path)
MODEL_ID=$(basename $(dirname $MODEL_TO_EVALUATE))

echo "MODE: $PROBLEM_MODE"
echo "MODEL_TO_EVALUATE: $MODEL_TO_EVALUATE"
echo "MODEL_ID: $MODEL_ID"
echo "NUM_SAMPLES: $NUM_SAMPLES"
echo "ITERS: $ITERS"
echo "NUM_PROCS: $NUM_PROCS"
echo "TEMPERATURE: $TEMPERATURE"
echo "TOP_P: $TOP_P"
echo "TOP_K: $TOP_K"
echo "ROLLOUT_ROUNDS: $ROLLOUT_ROUNDS"
echo "CUDA_DEVICES: $CUDA_DEVICES"

if [ "$PROBLEM_MODE" = "tsp" ]; then
    PROBLEM_TYPES="['TSP']"
    CROSSOVER_TYPE="tsp"
elif [ "$PROBLEM_MODE" = "mtsp" ]; then
    PROBLEM_TYPES="['CVRP_all']" 
    CROSSOVER_TYPE="mtsp"
else
    echo "ERROR: Unknown problem type '$PROBLEM_MODE'. Please use 'tsp' or 'mtsp'"
    exit 1
fi

echo "PROBLEM_TYPES: $PROBLEM_TYPES"
echo "MODULE_TO_MODIFY: $MODULE_TO_MODIFY"
echo "CROSSOVER_TYPE: $CROSSOVER_TYPE"

# "problem types" is ignored temporarily. Test CVRP_all only.

 CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python vllm_evaluate.py \
    $MODEL_TO_EVALUATE \
    --multi_model \
    -n $NUM_SAMPLES \
    --problem_types "$PROBLEM_TYPES" \
    --iters $ITERS \
    --num_procs $NUM_PROCS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --output_file results_multi_round_${MODEL_ID}.json \
    --rollout_rounds $ROLLOUT_ROUNDS \
    --module_to_modify $MODULE_TO_MODIFY \
    --crossover_type $CROSSOVER_TYPE \
    2>&1 | tee eval_multiturn_${MODEL_ID}.log