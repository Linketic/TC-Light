# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download NVIDIA Cosmos-1.0 Autoregressive models from Hugging Face")
    parser.add_argument(
        "--model_sizes",
        nargs="*",
        default=[
            "4B",
            "5B",
        ],  # Download all by default
        choices=["4B", "5B", "12B", "13B"],
        help="Which model sizes to download. Possible values: 4B, 5B, 12B, 13B.",
    )
    parser.add_argument(
        "--cosmos_version",
        type=str,
        default="1.0",
        choices=["1.0"],
        help="Which version of Cosmos to download. Only 1.0 is available at the moment.",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save the downloaded checkpoints."
    )
    args = parser.parse_args()
    return args


def main(args):
    ORG_NAME = "nvidia"

    # Mapping from size argument to Hugging Face repository name
    model_map = {
        "4B": "Cosmos-1.0-Autoregressive-4B",
        "5B": "Cosmos-1.0-Autoregressive-5B-Video2World",
        "12B": "Cosmos-1.0-Autoregressive-12B",
        "13B": "Cosmos-1.0-Autoregressive-13B-Video2World",
    }

    # Additional models that are always downloaded
    extra_models = [
        "Cosmos-1.0-Guardrail",
        "Cosmos-1.0-Diffusion-7B-Decoder-DV8x16x16ToCV8x8x8",
        "Cosmos-1.0-Tokenizer-CV8x8x8",
        "Cosmos-1.0-Tokenizer-DV8x16x16",
    ]

    # Create local checkpoints folder
    checkpoints_dir = Path(args.checkpoint_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    download_kwargs = dict(allow_patterns=["README.md", "model.pt", "config.json", "*.jit"])

    # Download the requested Autoregressive models
    for size in args.model_sizes:
        model_name = model_map[size]
        repo_id = f"{ORG_NAME}/{model_name}"
        local_dir = checkpoints_dir.joinpath(model_name)
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {repo_id} to {local_dir}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            **download_kwargs,
        )

    # Download the always-included models
    for model_name in extra_models:
        repo_id = f"{ORG_NAME}/{model_name}"
        local_dir = checkpoints_dir.joinpath(model_name)
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {repo_id} to {local_dir}...")
        # Download all files
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
