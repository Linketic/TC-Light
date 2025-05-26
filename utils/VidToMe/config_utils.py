import argparse
from omegaconf import OmegaConf, DictConfig
import os

def load_config(print_config = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/tea-pour.yaml',
                        help="Config file path")
    parser.add_argument('--base_config', type=str,
                        default=None,
                        help="Base config file path to override")
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