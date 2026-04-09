#!/usr/bin/env python3
"""
Upload SafeTensor checkpoints to Hugging Face Hub.

This script uploads model checkpoints and associated files from a local directory
to a Hugging Face repository.
"""

import os
import argparse
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError


def upload_checkpoint_to_hf(
    local_dir: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = True,
    commit_message: str = "Upload model checkpoint",
    revision: Optional[str] = None,
):
    """
    Upload a checkpoint directory to Hugging Face Hub.
    
    Args:
        local_dir: Path to the local checkpoint directory
        repo_id: Repository ID on Hugging Face (format: "username/repo-name")
        token: Hugging Face API token (optional if already logged in)
        private: Whether to create a private repository
        commit_message: Commit message for the upload
        revision: Branch/revision to upload to (default: main)
    """
    
    # Initialize the API
    api = HfApi(token=token)
    
    # Check if the local directory exists
    local_path = Path(local_dir)
    if not local_path.exists():
        raise ValueError(f"Local directory does not exist: {local_dir}")
    
    if not local_path.is_dir():
        raise ValueError(f"Path is not a directory: {local_dir}")
    
    # Check for required files
    safetensor_files = list(local_path.glob("*.safetensors"))
    if not safetensor_files:
        print(f"Warning: No .safetensors files found in {local_dir}")
    
    print(f"Found {len(safetensor_files)} SafeTensor files")
    
    # Try to create the repository (if it doesn't exist)
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            repo_type="model",
            exist_ok=True
        )
        print(f"Repository created/exists at: {repo_url}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return False
    
    # Upload the folder
    try:
        print(f"Uploading files from {local_dir} to {repo_id}...")
        
        # List files to be uploaded
        files_to_upload = []
        for file_path in local_path.iterdir():
            if file_path.is_file():
                files_to_upload.append(file_path.name)
        
        print(f"Files to upload: {', '.join(files_to_upload[:5])}" + 
              (f" and {len(files_to_upload) - 5} more..." if len(files_to_upload) > 5 else ""))
        
        # Upload the entire folder
        upload_info = upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            token=token,
            commit_message=commit_message,
            revision=revision,
        )
        
        print(f"Successfully uploaded to: https://huggingface.co/{repo_id}")
        if revision:
            print(f"Uploaded to branch: {revision}")
        
        return True
        
    except Exception as e:
        print(f"Error uploading files: {e}")
        return False


def upload_multiple_checkpoints(
    base_dir: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = True,
):
    """
    Upload multiple checkpoint directories to different branches.
    
    Args:
        base_dir: Base directory containing checkpoint subdirectories
        repo_id: Repository ID on Hugging Face
        token: Hugging Face API token
        private: Whether to create a private repository
    """
    
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    # Find all checkpoint directories
    checkpoint_dirs = []
    
    # Look for directories with SafeTensor files
    for item in base_path.rglob("*.safetensors"):
        checkpoint_dir = item.parent
        if checkpoint_dir not in checkpoint_dirs:
            checkpoint_dirs.append(checkpoint_dir)
    
    if not checkpoint_dirs:
        print(f"No checkpoint directories found in {base_dir}")
        return
    
    print(f"Found {len(checkpoint_dirs)} checkpoint directories:")
    for dir_path in checkpoint_dirs:
        print(f"  - {dir_path.relative_to(base_path)}")
    
    # Upload each checkpoint
    for checkpoint_dir in checkpoint_dirs:
        # Create a branch name from the directory structure
        relative_path = checkpoint_dir.relative_to(base_path)
        branch_name = str(relative_path).replace("/", "-").replace("_", "-")
        
        print(f"\nUploading {relative_path} to branch '{branch_name}'...")
        
        success = upload_checkpoint_to_hf(
            local_dir=str(checkpoint_dir),
            repo_id=repo_id,
            token=token,
            private=private,
            commit_message=f"Upload checkpoint: {relative_path}",
            revision=branch_name if len(checkpoint_dirs) > 1 else None
        )
        
        if success:
            print(f"✓ Successfully uploaded {relative_path}")
        else:
            print(f"✗ Failed to upload {relative_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload SafeTensor checkpoints to Hugging Face Hub"
    )
    
    parser.add_argument(
        "local_dir",
        type=str,
        help="Path to the checkpoint directory or base directory containing checkpoints"
    )
    
    parser.add_argument(
        "repo_id",
        type=str,
        help="Hugging Face repository ID (format: 'username/repo-name')"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (optional if already logged in via huggingface-cli)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Create a private repository (default: True)"
    )
    
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public repository"
    )
    
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload model checkpoint",
        help="Commit message for the upload"
    )
    
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Branch/revision to upload to (default: main)"
    )
    
    parser.add_argument(
        "--multiple",
        action="store_true",
        help="Upload multiple checkpoints from subdirectories to different branches"
    )
    
    args = parser.parse_args()
    
    # Handle public/private flag
    private = not args.public if args.public else args.private
    
    try:
        if args.multiple:
            # Upload multiple checkpoints from subdirectories
            upload_multiple_checkpoints(
                base_dir=args.local_dir,
                repo_id=args.repo_id,
                token=args.token,
                private=private,
            )
        else:
            # Upload a single checkpoint directory
            success = upload_checkpoint_to_hf(
                local_dir=args.local_dir,
                repo_id=args.repo_id,
                token=args.token,
                private=private,
                commit_message=args.commit_message,
                revision=args.revision,
            )
            
            if success:
                print("\n✓ Upload completed successfully!")
            else:
                print("\n✗ Upload failed!")
                exit(1)
                
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)


if __name__ == "__main__":
    
    main()