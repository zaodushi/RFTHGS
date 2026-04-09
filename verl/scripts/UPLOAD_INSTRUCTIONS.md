# Hugging Face Upload Instructions

## Setup

1. Install required dependencies:
```bash
pip install -r requirements_hf_upload.txt
```

2. Login to Hugging Face (one-time setup):
```bash
huggingface-cli login
```
Or set your token as an environment variable:
```bash
export HF_TOKEN="your_token_here"
```

## Usage

### Upload a single checkpoint

To upload the checkpoint at `Ckpt_to_be_evaluated/<run_id>/global_step_<N>/`:

```bash
python upload_to_huggingface.py \
    Ckpt_to_be_evaluated/<run_id>/global_step_<N>/ \
    your-username/your-model-name \
    --token $HF_TOKEN
```

### Upload all checkpoints to different branches

To upload all checkpoints found under `Ckpt_to_be_evaluated/` to different branches:

```bash
python upload_to_huggingface.py \
    Ckpt_to_be_evaluated/ \
    your-username/your-model-name \
    --multiple \
    --token $HF_TOKEN
```

### Command-line Options

- `local_dir`: Path to checkpoint directory
- `repo_id`: Hugging Face repository ID (format: "username/repo-name")
- `--token`: Your Hugging Face API token (optional if logged in)
- `--private`: Create a private repository (default)
- `--public`: Create a public repository
- `--commit-message`: Custom commit message
- `--revision`: Upload to a specific branch
- `--multiple`: Upload multiple checkpoints to different branches

## Examples

1. **Upload to a public repository:**
```bash
python upload_to_huggingface.py \
    Ckpt_to_be_evaluated/<run_id>/global_step_<N>/ \
    your-username/your-model-name \
    --public
```

2. **Upload with custom commit message:**
```bash
python upload_to_huggingface.py \
    Ckpt_to_be_evaluated/<run_id>/global_step_<N>/ \
    your-username/your-model-name \
    --commit-message "Add checkpoint from training run"
```

3. **Upload to a specific branch:**
```bash
python upload_to_huggingface.py \
    Ckpt_to_be_evaluated/<run_id>/global_step_<N>/ \
    your-username/your-model-name \
    --revision experiment-v2
```

## Getting Your Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "write" permissions
3. Copy the token and use it with the `--token` flag or set as `HF_TOKEN` environment variable