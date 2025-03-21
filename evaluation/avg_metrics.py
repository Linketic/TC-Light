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
    args = parser.parse_args()

    metrics_dict = {}

    for output_dir in tqdm(args.output_dirs, desc="Accumulating Results"):
        with open(os.path.join(output_dir, 'result.txt'), 'r') as f:
            lines = f.readlines()

        # Extract the line containing the metrics (third line)
        data_line = lines[2].strip()

        # Split the line into components
        parts = [p.strip() for p in data_line.split(', ')]

        metric_dict = ast.literal_eval(data_line.split('>, ')[-1].split(')')[0])
        for metric_key in metric_dict:
            if metric_key not in metrics_dict:
                metrics_dict[metric_key] = []
            metrics_dict[metric_key].append(metric_dict[metric_key])

    for metric_key in metrics_dict:
        metrics_dict[metric_key] = np.mean(metrics_dict[metric_key])

    print(f"Averaged Metrics of {args.output_dirs}: \n", metrics_dict)


