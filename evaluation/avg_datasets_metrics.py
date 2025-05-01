import sys
import yaml

import os
import ast
import glob
import argparse
import numpy as np
from tqdm import tqdm

def yaml_load(path):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dirs', type=str, nargs='+')
    parser.add_argument('--txt_name', type=str, default='iclight_vidtome_fastblend.txt')
    args = parser.parse_args()

    metrics_dict = {}

    for output_dir in tqdm(args.output_dirs, desc="Accumulating Results"):
        if not os.path.exists(os.path.join(output_dir, args.txt_name)):
            print(f'[INFO]: {os.path.join(output_dir, args.txt_name)} not exists, skipped.')
            continue

        with open(os.path.join(output_dir, args.txt_name), 'r') as f:
            lines = f.readlines()

        # Extract the line containing the metrics (third line)
        for dataline in lines[1:]:
            dataline = dataline.strip()
            metric_key = dataline.split(': ')[0]
            metric_val = float(dataline.split(': ')[-1])
            if metric_key not in metrics_dict:
                metrics_dict[metric_key] = []
            metrics_dict[metric_key].append(metric_val)

    for metric_key in metrics_dict:
        metrics_dict[metric_key] = np.mean(metrics_dict[metric_key])

    print(f"Averaged Metrics of {args.output_dirs}: \n", metrics_dict)


