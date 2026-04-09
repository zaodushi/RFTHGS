# Set these variables before running the script
export EXPERIMENT="${EXPERIMENT:-}"           # e.g. "2025xxxx_xxxxxx"
export STEP="${STEP:-}"                       # e.g. "global_step_700"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-}" # path to the base Qwen3-14B model
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-./checkpoints/$EXPERIMENT}"
export TARGET_DIR="${TARGET_DIR:-./Ckpt_to_be_evaluated/$EXPERIMENT/$STEP}"

if [ -z "$EXPERIMENT" ] || [ -z "$STEP" ] || [ -z "$BASE_MODEL_PATH" ]; then
    echo "ERROR: EXPERIMENT, STEP, and BASE_MODEL_PATH must be set."
    exit 1
fi

echo $TARGET_DIR
```# Function: copy tokenizer files
copy_tokenizer_files() {
    local ckpt_path=$1
    local BASE_model_path=$2
    local files_to_copy=(
        "added_tokens.json"
        "config.json"
        "generation_config.json"
        "special_tokens_map.json"
        "tokenizer_config.json"
        "tokenizer.json"
        "vocab.json"
    )
    if [ -f "$BASE_model_path/merges.txt" ]; then
        files_to_copy+=("merges.txt")
    fi
    # Create the target path if it does not exist
    if [ ! -d "$ckpt_path" ]; then
        mkdir -p "$ckpt_path"
        echo "Created checkpoint directory: $ckpt_path" >&2
    else
        echo "Checkpoint directory already exists: $ckpt_path" >&2
    fi

    # Copy each tokenizer file
    for filename in "${files_to_copy[@]}"; do
        src="$BASE_model_path/$filename"
        dst="$ckpt_path/$filename"
        if [ -e "$src" ]; then
            cp "$src" "$dst"
            echo "Copied $src to $dst"
        else
            echo "Warning: $src does not exist."
        fi
    done
}```

# upgrade to latest Transformers
pip install transformers==4.56.1

# Your conversion command goes here
python convert_pt_to_safetensors.py --backend fsdp --hf_model_path $BASE_MODEL_PATH --local_dir $CHECKPOINT_PATH/$STEP/actor --target_dir $TARGET_DIR
# call copy tokenizer to copy
# copy_tokenizer_files "$TARGET_DIR" "$BASE_MODEL_PATH"
cp $CHECKPOINT_PATH/$STEP/actor/huggingface/* $TARGET_DIR/
echo "Model convert done for $TARGET_DIR"