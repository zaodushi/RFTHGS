#!/bin/bash
# exit if any command fails
set -e

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Note: Boolean parameters (marked as BOOL) accept 'True' or 'False' values"
    echo ""
    echo "Options:"
    echo "  Basic Options:"
    echo "  --model MODEL_PATH                          Model path (default: Qwen/Qwen3-14B)"
    echo "  --train TRAIN_PATH                          Training data path (default:$PROJECT_ROOT/data/batch_training_data.parquet)"
    echo "  --val VAL_PATH                              Validation data path (default:$PROJECT_ROOT/data/batch_test_data.parquet)"
    echo "  --batch-size SIZE                           Training batch size (default: 16)"
    echo "  --epochs EPOCHS                             Total epochs (default: 2000)"
    echo "  --lr LEARNING_RATE                          Learning rate (default: 1e-6)"
    echo "  --rollout-n N                               Rollout n value (default: 8)"
    echo "  --gpus-per-node GPUS                        GPUs per node (default: 8)"
    echo "  --nnodes NODES                              Number of nodes (default: 1)"
    echo "  --project PROJECT                           Wandb project name (default: RFTVRP)"
    echo "  --run-name NAME                             Wandb run name (default: auto-generated)"
    echo "  --pyvrp-code BOOL                           Use original pyvrp code as baseline (default: False)"
    echo "  --skip-data-gen BOOL                        Skip data generation step (default: False)"
    echo "  --num-training-prompts NUM                  Number of training prompts for data generation (default: 4)"
    echo "  --default-local-dir DIR                     Default local directory for model checkpoints (default: ./checkpoints)"
    echo "  --problem-type TYPE                         Problem type: CVRP or CVRPTW (default: CVRP)"
    echo ""
    echo "  Advanced Options:"
    echo "  --adv-estimator ESTIMATOR                   Algorithm advantage estimator (default: grpo)"
    echo "  --max-prompt-length LENGTH                  Max prompt length (default: 4096)"
    echo "  --max-response-length LENGTH                Max response length (default: 8192)"
    echo "  --ppo-mini-batch-size SIZE                  PPO mini batch size (default: same as batch-size)"
    echo "  --ppo-micro-batch-size-per-gpu SIZE         PPO micro batch size per GPU (default: 1)"
    echo "  --ppo-max-token-len-per-gpu LENGTH          PPO max token length per GPU (default: 12288)"
    echo "  --log-prob-micro-batch-size SIZE            Log prob micro batch size (default: 2)"
    echo "  --rollout-log-prob-micro-batch-size SIZE    Rollout log prob micro batch size per GPU (default: 2)"
    echo "  --use-kl-loss BOOL                          Enable KL loss (default: False)"
    echo "  --kl-loss-coef COEF                         KL loss coefficient (default: 0.0)"
    echo "  --kl-loss-type TYPE                         KL loss type (default: low_var_kl)"
    echo "  --entropy-coeff COEF                        Entropy coefficient (default: 0)"
    echo "  --gradient-checkpointing BOOL               Enable gradient checkpointing (default: True)"
    echo "  --param-offload BOOL                        Enable FSDP param offload (default: True)"
    echo "  --optimizer-offload BOOL                    Enable FSDP optimizer offload (default: True)"
    echo "  --tensor-model-parallel-size SIZE           Tensor model parallel size (default: 1)"
    echo "  --gpu-memory-utilization UTIL               GPU memory utilization (default: 0.7)"
    echo "  --chunked-prefill BOOL                      Enable chunked prefill (default: False)"
    echo "  --use-kl-in-reward BOOL                     Use KL in reward (default: False)"
    echo "  --critic-warmup STEPS                       Critic warmup steps (default: 0)"
    echo "  --trainer-logger LOGGERS                     Trainer loggers (default: console,wandb)"
    echo "  --save-freq FREQ                            Save frequency (default: -1)"
    echo "  --test-freq FREQ                            Test frequency (default: -1)"
    echo "  --val-before-train BOOL                     Validate before training (default: False)"
    echo "  --clip-ratio-low RATIO                      Low clip ratio (default: 0.2)"
    echo "  --clip-ratio-high RATIO                     High clip ratio (default: 0.28)"
    echo "  --clip-ratio-c RATIO                        Clip ratio C (default: 10.0)"
    echo "  --loss-agg-mode MODE                        Loss aggregation mode (default: token-mean)"
    echo "  --filter-groups BOOL                        Enable filter groups (default: False)"
    echo "  --truncation MODE                           Data truncation mode (default: error)"
    echo "  --filter-overlong-prompts BOOL              Filter overlong prompts (default: False)"
    echo "  --use-remove-padding BOOL                   Use remove padding (default: True)"
    echo "  --use-dynamic-bsz BOOL                      Use dynamic batch size (default: False)"
    echo "  --rebuild-dataloader BOOL                      Rebuild dataloader (default: True)"
    echo "  --loss-scale-factor FACTOR                  Loss scale factor (default: 1000)"
    echo "  --actor-lora-rank RANK                      Actor LoRA rank (default: 32)"
    echo "  --actor-lora-alpha ALPHA                    Actor LoRA alpha (default: 16)"
    echo "  --actor-target-modules MODULES              Actor target modules (default: [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj])"
    echo "  --actor-exclude-modules MODULES             Actor exclude modules (default: null)"
    echo ""
    echo "  Validation Sampling Options:"
    echo "  --val-top-k VALUE                           Top-k sampling parameter for validation (default: -1)"
    echo "  --val-top-p VALUE                           Top-p sampling parameter for validation (default: 1.0)"
    echo "  --val-temperature VALUE                     Sampling temperature for validation (default: 1)"
    echo "  --val-n VALUE                               Number of times to repeat validation (default: 16)"
    echo "  --val-do-sample BOOL                        Whether to sample during validation (default: True)"
    echo "  --help                                      Display this help message"
    exit 1
}

# Get absolute path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# Default values
model_path="Qwen/Qwen3-14B"
train_path="$PROJECT_ROOT/data/batch_training_data.parquet"
test_path="$PROJECT_ROOT/data/batch_test_data.parquet"
train_batch_size=16
total_epochs=2000
learning_rate="1e-6"
rollout_n=8
n_gpus_per_node=8
nnodes=1
project_name="RFTVRP"
run_name=""
use_pyvrp_code="True" # True: use original pyvrp code as baseline; False: use simplified crossover code as baseline.
skip_data_gen="False"
num_training_prompts=16
num_test_prompts=1
module_to_modify="crossover" # subpopulation, subpopulation_new_prompt, or crossover
use_smooth_reward="True"
use_ast_check="True"
penalty_compile_fail="-1"
penalty_runtime_error="-0.8"
score_relative_lowerbound="-0.7"
default_local_dir="./checkpoints"
trainer_logger="console,wandb"
problem_type="CVRP"

# Advanced parameters defaults
adv_estimator="grpo"
max_prompt_length=8192
max_response_length=8192
ppo_max_token_len_per_gpu=12288
ppo_mini_batch_size=""  # Will default to train_batch_size if not set
ppo_micro_batch_size_per_gpu=2
ref_log_prob_micro_batch_size_per_gpu=2
rollout_log_prob_micro_batch_size_per_gpu=2

# Validation sampling parameters
val_top_k=-1
val_top_p=1.0
val_temperature=1
val_n=16
val_do_sample="True"
use_kl_loss="False"
kl_loss_coef=0.0
kl_loss_type="low_var_kl"
entropy_coeff=0
actor_lora_rank=0 # 0 to block lora, > 1 to use lora
actor_lora_alpha=16
actor_target_modules="all-linear" # "[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"
actor_exclude_modules="null"
enable_gradient_checkpointing="True"
param_offload="True"
optimizer_offload="True"
tensor_model_parallel_size=1
gpu_memory_utilization=0.7
enable_chunked_prefill="False"
use_kl_in_reward="False"
critic_warmup=0
save_freq=-1
test_freq=20
val_before_train="False"
clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c=10.0
loss_agg_mode="seq-mean-token-sum"
filter_groups_enable="False"
truncation="error"
filter_overlong_prompts="False"
use_remove_padding="True"
use_dynamic_bsz="False"
rebuild_dataloader="True"
loss_scale_factor=500

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        # Basic options
        --model)
            model_path="$2"
            shift 2
            ;;
        --train)
            train_path="$2"
            shift 2
            ;;
        --val)
            test_path="$2"
            shift 2
            ;;
        --batch-size)
            train_batch_size="$2"
            shift 2
            ;;
        --epochs)
            total_epochs="$2"
            shift 2
            ;;
        --lr)
            learning_rate="$2"
            shift 2
            ;;
        --rollout-n)
            rollout_n="$2"
            shift 2
            ;;
        --gpus-per-node)
            n_gpus_per_node="$2"
            shift 2
            ;;
        --nnodes)
            nnodes="$2"
            shift 2
            ;;
        --project)
            project_name="$2"
            shift 2
            ;;
        --run-name)
            run_name="$2"
            shift 2
            ;;
        --pyvrp-code)
            use_pyvrp_code="$2"
            shift 2
            ;;
        --skip-data-gen)
            skip_data_gen="$2"
            shift 2
            ;;
        --num-training-prompts)
            num_training_prompts="$2"
            shift 2
            ;;
        --default-local-dir)
            default_local_dir="$2"
            shift 2
            ;;
        --module-to-modify)
            module_to_modify="$2"
            shift 2
            ;;
        --use-smooth-reward)
            use_smooth_reward="$2"
            shift 2
            ;;
        --penalty-compile-fail)
            penalty_compile_fail="$2"
            shift 2
            ;;
        --penalty-runtime-error)
            penalty_runtime_error="$2"
            shift 2
            ;;
        --score-relative-lowerbound)
            score_relative_lowerbound="$2"
            shift 2
            ;;
        --use-ast-check)
            use_ast_check="$2"
            shift 2
            ;;
        --problem-type)
            problem_type="$2"
            shift 2
            ;;
        --adv-estimator)
            adv_estimator="$2"
            shift 2
            ;;
        --max-prompt-length)
            max_prompt_length="$2"
            shift 2
            ;;
        --max-response-length)
            max_response_length="$2"
            shift 2
            ;;
        --ppo-mini-batch-size)
            ppo_mini_batch_size="$2"
            shift 2
            ;;
        --ppo-micro-batch-size-per-gpu)
            ppo_micro_batch_size_per_gpu="$2"
            shift 2
            ;;
        --ppo-max-token-len-per-gpu)
            ppo_max_token_len_per_gpu="$2"
            shift 2
            ;;
        --ref-log-prob-micro-batch-size-per-gpu)
            ref_log_prob_micro_batch_size_per_gpu="$2"
            shift 2
            ;;
        --rollout-log-prob-micro-batch-size)
            rollout_log_prob_micro_batch_size_per_gpu="$2"
            shift 2
            ;;
        --use-kl-loss)
            use_kl_loss="$2"
            shift 2
            ;;
        --kl-loss-coef)
            kl_loss_coef="$2"
            shift 2
            ;;
        --kl-loss-type)
            kl_loss_type="$2"
            shift 2
            ;;
        --entropy-coeff)
            entropy_coeff="$2"
            shift 2
            ;;
        --gradient-checkpointing)
            enable_gradient_checkpointing="$2"
            shift 2
            ;;
        --param-offload)
            param_offload="$2"
            shift 2
            ;;
        --optimizer-offload)
            optimizer_offload="$2"
            shift 2
            ;;
        --tensor-model-parallel-size)
            tensor_model_parallel_size="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            gpu_memory_utilization="$2"
            shift 2
            ;;
        --chunked-prefill)
            enable_chunked_prefill="$2"
            shift 2
            ;;
        --use-kl-in-reward)
            use_kl_in_reward="$2"
            shift 2
            ;;
        --critic-warmup)
            critic_warmup="$2"
            shift 2
            ;;
        --trainer-logger)
            trainer_logger="$2"
            shift 2
            ;;
        --save-freq)
            save_freq="$2"
            shift 2
            ;;
        --test-freq)
            test_freq="$2"
            shift 2
            ;;
        --val-before-train)
            val_before_train="$2"
            shift 2
            ;;
        --clip-ratio-low)
            clip_ratio_low="$2"
            shift 2
            ;;
        --clip-ratio-high)
            clip_ratio_high="$2"
            shift 2
            ;;
        --clip-ratio-c)
            clip_ratio_c="$2"
            shift 2
            ;;
        --loss-agg-mode)
            loss_agg_mode="$2"
            shift 2
            ;;
        --filter-groups)
            filter_groups_enable="$2"
            shift 2
            ;;
        --truncation)
            truncation="$2"
            shift 2
            ;;
        --filter-overlong-prompts)
            filter_overlong_prompts="$2"
            shift 2
            ;;
        --use-remove-padding)
            use_remove_padding="$2"
            shift 2
            ;;
        --use-dynamic-bsz)
            use_dynamic_bsz="$2"
            shift 2
            ;;
        --rebuild-dataloader)
            rebuild_dataloader="$2"
            shift 2
            ;;
        --loss-scale-factor)
            loss_scale_factor="$2"
            shift 2
            ;;
        --actor-lora-rank)
            actor_lora_rank="$2"
            shift 2
            ;;
        --actor-lora-alpha)
            actor_lora_alpha="$2"
            shift 2
            ;;
        --actor-target-modules)
            actor_target_modules="$2"
            shift 2
            ;;
        --actor-exclude-modules)
            actor_exclude_modules="$2"
            shift 2
            ;;
        --val-top-k)
            val_top_k="$2"
            shift 2
            ;;
        --val-top-p)
            val_top_p="$2"
            shift 2
            ;;
        --val-temperature)
            val_temperature="$2"
            shift 2
            ;;
        --val-n)
            val_n="$2"
            shift 2
            ;;
        --val-do-sample)
            val_do_sample="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Generate run name if not provided
if [[ -z "$run_name" ]]; then
    model_name=$(basename "$model_path")
    run_name="$model_name-$(date '+%m%d')"
fi

# Set ppo_mini_batch_size to train_batch_size if not specified
if [[ -z "$ppo_mini_batch_size" ]]; then
    ppo_mini_batch_size=$train_batch_size
fi

# Setup environment variables
export FLASH_ATTENTION_DETERMINISTIC=1
export use_pyvrp_code=$use_pyvrp_code
export module_to_modify=$module_to_modify
export USE_SMOOTH_REWARD=$use_smooth_reward
export penalty_compile_fail=$penalty_compile_fail
export penalty_runtime_error=$penalty_runtime_error
export score_relative_lowerbound=$score_relative_lowerbound
export use_ast_check=$use_ast_check
export PROBLEM_TYPE=$problem_type

# Print configuration
echo "=== Basic Configuration ==="
echo "Model Path: $model_path"
echo "Training Data: $train_path"
echo "Validation Data: $test_path"
echo "Batch Size: $train_batch_size"
echo "Total Epochs: $total_epochs"
echo "Learning Rate: $learning_rate"
echo "Rollout N: $rollout_n"
echo "GPUs per Node: $n_gpus_per_node"
echo "Number of Nodes: $nnodes"
echo "Project Name: $project_name"
echo "Run Name: $run_name"
echo "Use PyVRP Code: $use_pyvrp_code"
echo "Skip Data Generation: $skip_data_gen"
echo "Number of Training Prompts: $num_training_prompts"
echo "Default Local Directory: $default_local_dir"
echo "Module to Modify: $module_to_modify"
echo "Use Smooth Reward: $use_smooth_reward"
echo "Penalty Compile Fail: $penalty_compile_fail"
echo "Penalty Runtime Error: $penalty_runtime_error"
echo "Score Relative Lowerbound: $score_relative_lowerbound"
echo "Use AST Check: $use_ast_check"
echo "Problem Type: $problem_type"
echo ""
echo "=== Advanced Configuration ==="
echo "Advantage Estimator: $adv_estimator"
echo "Max Prompt Length: $max_prompt_length"
echo "Max Response Length: $max_response_length"
echo "PPO Mini Batch Size: $ppo_mini_batch_size"
echo "PPO Micro Batch Size per GPU: $ppo_micro_batch_size_per_gpu"
echo "PPO Max Token Length per GPU: $ppo_max_token_len_per_gpu"
echo "Ref Log Prob Micro Batch Size: $ref_log_prob_micro_batch_size_per_gpu"
echo "Rollout Log Prob Micro Batch Size: $rollout_log_prob_micro_batch_size_per_gpu"
echo "Use KL Loss: $use_kl_loss"
echo "KL Loss Coefficient: $kl_loss_coef"
echo "KL Loss Type: $kl_loss_type"
echo "Entropy Coefficient: $entropy_coeff"
echo "Gradient Checkpointing: $enable_gradient_checkpointing"
echo "FSDP Param Offload: $param_offload"
echo "FSDP Optimizer Offload: $optimizer_offload"
echo "Tensor Model Parallel Size: $tensor_model_parallel_size"
echo "GPU Memory Utilization: $gpu_memory_utilization"
echo "Chunked Prefill: $enable_chunked_prefill"
echo "Use KL in Reward: $use_kl_in_reward"
echo "Critic Warmup: $critic_warmup"
echo "Save Frequency: $save_freq"
echo "Test Frequency: $test_freq"
echo "Validate Before Train: $val_before_train"
echo "Clip Ratios: low=$clip_ratio_low, high=$clip_ratio_high, c=$clip_ratio_c"
echo "Loss Aggregation Mode: $loss_agg_mode"
echo "Filter Groups: $filter_groups_enable"
echo "Data Truncation: $truncation"
echo "Filter Overlong Prompts: $filter_overlong_prompts"
echo "Use Remove Padding: $use_remove_padding"
echo "Use Dynamic Batch Size: $use_dynamic_bsz"
echo "Rebuild dataloader: $rebuild_dataloader"
echo "Loss Scale Factor: $loss_scale_factor"
echo ""
echo "=== Validation Sampling Configuration ==="
echo "Validation Top-k: $val_top_k"
echo "Validation Top-p: $val_top_p"
echo "Validation Temperature: $val_temperature"
echo "Validation Repeat Count (n): $val_n"
echo "Validation Do Sample: $val_do_sample"
echo "===================="

# Generate data if not skipped
if [[ "$skip_data_gen" = "False" ]]; then
    echo "Generating training data..."
    python ../data/generate_batch_training_data.py --num_training_prompts $num_training_prompts --num_test_prompts $num_test_prompts
fi

# Prepare file paths for python
train_files="['$train_path']"
test_files="['$test_path']"

# Setup directories
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../pyvrp"

echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "BUILD_DIR: $BUILD_DIR"

# Meson setup
cd "$BUILD_DIR"
rm -rf build
meson setup build -Doptimization=3
meson compile -C build
rm -rf build/_crossover*.so
rm -rf "$SCRIPT_DIR/../benchmark/pyvrp*"
cd ../verl

# Update gcc
conda install -c conda-forge -y gcc=12.1.0

# # Evaluate baseline
python -m verl.utils.reward_score.vrp_evo --eval_baseline

# Run training
python3 -m verl.trainer.main_dapo \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$train_batch_size \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=$filter_overlong_prompts \
    data.truncation="$truncation" \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.lora_rank=$actor_lora_rank \
    actor_rollout_ref.model.lora_alpha=$actor_lora_alpha \
    actor_rollout_ref.model.target_modules=$actor_target_modules \
    actor_rollout_ref.model.exclude_modules=$actor_exclude_modules \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.optim.lr=$learning_rate \
    actor_rollout_ref.model.use_remove_padding=$use_remove_padding \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=$enable_gradient_checkpointing \
    actor_rollout_ref.actor.fsdp_config.param_offload=$param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.enable_chunked_prefill=$enable_chunked_prefill \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$rollout_log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=$param_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ref_log_prob_micro_batch_size_per_gpu \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    trainer.critic_warmup=$critic_warmup \
    trainer.logger=[$trainer_logger] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=$val_before_train \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    trainer.rebuild_dataloader=$rebuild_dataloader \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=$clip_ratio_c \
    actor_rollout_ref.actor.loss_agg_mode="$loss_agg_mode" \
    actor_rollout_ref.actor.loss_scale_factor=$loss_scale_factor \
    algorithm.filter_groups.enable=$filter_groups_enable \
    actor_rollout_ref.rollout.val_kwargs.top_k=$val_top_k \
    actor_rollout_ref.rollout.val_kwargs.top_p=$val_top_p \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
    actor_rollout_ref.rollout.val_kwargs.n=$val_n \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$val_do_sample \
    trainer.default_local_dir=$default_local_dir

# Clean up shared memory resources to avoid warnings
echo "Cleaning up shared memory resources..."
python3 -c "
import multiprocessing
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')
try:
    # Force cleanup of any remaining shared memory objects
    multiprocessing.resource_tracker._resource_tracker._cleanup()
except:
    pass
"
echo "Cleanup completed."
