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

from cosmos1.scripts.convert_pixtral_ckpt import convert_pixtral_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Download NVIDIA Cosmos-1.0 Diffusion models from Hugging Face")
    parser.add_argument(
        "--model_sizes",
        nargs="*",
        default=[
            "7B",
            "14B",
        ],  # Download all by default
        choices=["7B", "14B"],
        help="Which model sizes to download. Possible values: 7B, 14B",
    )
    parser.add_argument(
        "--model_types",
        nargs="*",
        default=[
            "Text2World",
            "Video2World",
        ],  # Download all by default
        choices=["Text2World", "Video2World"],
        help="Which model types to download. Possible values: Text2World, Video2World",
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
        "7B": "Cosmos-1.0-Diffusion-7B",
        "14B": "Cosmos-1.0-Diffusion-14B",
    }

    # Additional models that are always downloaded
    extra_models = [
        "Cosmos-1.0-Guardrail",
        "Cosmos-1.0-Tokenizer-CV8x8x8",
    ]

    if "Text2World" in args.model_types:
        extra_models.append("Cosmos-1.0-Prompt-Upsampler-12B-Text2World")

    # Create local checkpoints folder
    checkpoints_dir = Path(args.checkpoint_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    download_kwargs = dict(allow_patterns=["README.md", "model.pt", "config.json", "*.jit"])

    # Download the requested Autoregressive models
    for size in args.model_sizes:
        for model_type in args.model_types:
            suffix = f"-{model_type}"
            model_name = model_map[size] + suffix
            repo_id = f"{ORG_NAME}/{model_name}"
            local_dir = checkpoints_dir.joinpath(model_name)
            local_dir.mkdir(parents=True, exist_ok=True)

            print(f"Downloading {repo_id} to {local_dir}...")
            snapshot_download(
                repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False, **download_kwargs
            )

    # Download the always-included models
    for model_name in extra_models:
        repo_id = f"{ORG_NAME}/{model_name}"
        local_dir = checkpoints_dir.joinpath(model_name)
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {repo_id} to {local_dir}...")
        # Download all files for Guardrail
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )

    if "Video2World" in args.model_types:
        # Prompt Upsampler for Cosmos-1.0-Diffusion-Video2World models
        convert_pixtral_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name="Pixtral-12B",
            vit_type="pixtral-12b-vit",
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
