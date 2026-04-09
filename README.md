# RFTHGS: Refining Hybrid Genetic Search via Reinforcement Fine-Tuned LLM

[Paper link](https://openreview.net/pdf?id=aITKXFeivk)

---

## Environment Setup

**Requirements:** Python 3.10, CUDA 12.4, PyTorch 2.6.0

### 1. Create conda environment

```bash
conda create -n rfthgs python=3.10
conda activate rfthgs
```

### 2. Install build tools

```bash
python -m pip install -U meson meson-python ninja
sudo apt install -y ccache clang rsync
conda install -c conda-forge -y gcc=12.1.0
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Build PyVRP

```bash
cd pyvrp
meson setup build -Doptimization=3
meson compile -C build
cd ..
```

> **Important:** The training and evaluation pipelines use the local`pyvrp`.

### 5. Install training dependencies

```bash
cd verl/scripts
pip install wheel click==8.2.1 shortuuid docblock
pip install opentelemetry-sdk==1.26.0 opentelemetry-api==1.26.0
USE_MEGATRON=0 USE_SGLANG=0 bash install_vllm_sglang_mcore_qwen3.sh
cd ../..
```

### 6. (Optional) Set up Weights & Biases

```bash
export WANDB_API_KEY="<your_wandb_api_key>"
```

---

## Training

Training is launched from the `verl/` directory.

```bash
cd verl

# Minimal example (single node, 8 GPUs)
ray stop --force
bash RFTVRP_train_with_args.sh \
    --model /path/to/Qwen3-14B \
    --module-to-modify subpopulation \
    --run-name my_run \
    --gpus-per-node 8 \
    --epochs 2000 \
    2>&1 | tee training.log
```

Key arguments (see `bash RFTVRP_train_with_args.sh --help` for the full list):


| Argument              | Default          | Description                           |
| --------------------- | ---------------- | ------------------------------------- |
| `--model`             | `Qwen/Qwen3-14B` | Path to the base model                |
| `--module-to-modify`  | —                | `subpopulation` or `crossover`        |
| `--default-local-dir` | `./checkpoints`  | Root directory for saving checkpoints |
| `--rollout-n`         | `8`              | Number of operator samples per prompt |
| `--batch-size`        | `16`             | Training batch size                   |
| `--epochs`            | `2000`           | Total training epochs                 |
| `--save-freq`         | `-1`             | Checkpoint save interval (steps)      |
| `--test-freq`         | `-1`             | Validation interval (steps)           |


---

## Evaluation

```bash
cd evaluation

# Build PyVRP first (required before evaluation)
cd pyvrp && rm -rf build && meson setup build -Doptimization=3 && meson compile -C build && cd ..
pip uninstall -y pyvrp

bash evaluate.sh mtsp subpopulation /path/to/model/checkpoint 64 800 16 1 0.95 200 1 0
```

## License

The code in `verl/` is subject to the [verl license](verl/LICENSE).  
The code in `pyvrp/` is subject to the [PyVRP license](pyvrp/LICENSE.md).  
All other project code is released under the MIT License.