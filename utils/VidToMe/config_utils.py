import argparse
from omegaconf import OmegaConf, DictConfig
import os

def load_config(print_config = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/tclight_default.yaml',
                        help="Config file path")
    parser.add_argument('--base_config', type=str,
                        default=None,
                        help="Base config file path to override")
    parser.add_argument('--input_path', '-i', type=str,
                        default=None,
                        help="path to video, for a fast usage")
    parser.add_argument('--prompt', '-p', type=str,
                        default=None,
                        help="prompt for video relighting, for a fast usage")
    parser.add_argument('--negative_prompt', '-n', type=str,
                        default=None,
                        help="negative prompt for video relighting, for a fast usage")
    parser.add_argument('--multi_axis', action='store_true',
                        help="use multi-axis denoising, for a fast usage")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # Recursively merge base configs
    cur_config_path = args.config
    cur_config = config
    if args.base_config is not None:
        cur_config.base_config = args.base_config
    while "base_config" in cur_config and cur_config.base_config != cur_config_path:
        base_config = OmegaConf.load(cur_config.base_config)
        config = OmegaConf.merge(base_config, config)
        cur_config_path = cur_config.base_config
        cur_config = base_config
    
    # overwrite config with command line arguments for fast usage
    if args.input_path is not None and config.data.scene_type.lower() == "video":
        config.data.rgb_path = args.input_path
    if args.prompt is not None:
        if isinstance(config.generation.prompt, dict):
            config.generation.prompt["edit"] = args.prompt
        else:
            config.generation.prompt = args.prompt
    if args.negative_prompt is not None:
        config.generation.negative_prompt = args.negative_prompt
    if args.multi_axis:
        config.generation.alpha_t = 0.01

    prompt = config.generation.prompt
    if isinstance(prompt, str):
        prompt = {"edit": prompt}
    config.generation.prompt = prompt
    OmegaConf.resolve(config)
    if print_config:
        print("[INFO] loaded config:")
        print(OmegaConf.to_yaml(config))
    
    return config

def save_config(config: DictConfig, path, gene = False, inv = False):
    os.makedirs(path, exist_ok = True)
    config = OmegaConf.create(config)
    if gene:
        config.pop("inversion")
    if inv:
        config.pop("generation")
    OmegaConf.save(config, os.path.join(path, "config.yaml"))