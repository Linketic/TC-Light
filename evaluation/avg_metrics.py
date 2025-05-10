import sys
import yaml

import os
import json
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
    parser.add_argument('--result_file', type=str, default='result.txt')
    parser.add_argument('--start_row', type=int, default=1)
    parser.add_argument('--vbench', action='store_true')
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    metrics_dict = {}

    for output_dir in tqdm(args.output_dirs, desc="Accumulating Results"):
        with open(os.path.join(output_dir, args.result_file), 'r') as f:
            lines = f.readlines()

        # Extract the line containing the metrics (third line)
        for dataline in lines[args.start_row:]:
            dataline = dataline.strip()
            metric_key = dataline.split(': ')[0]
            metric_val = float(dataline.split(': ')[-1])
            if metric_key not in metrics_dict:
                metrics_dict[metric_key] = []
            metrics_dict[metric_key].append(metric_val)
        
        if args.vbench and os.path.exists(os.path.join(output_dir, 'vbench')):
            # Extract the line containing the vbench metrics (first line)
            result_file = sorted([file for file in glob.glob(os.path.join(output_dir, 'vbench', '*.json')) if file.endswith('_eval_results.json')])[-1]
            vbench_metrics = json.load(open(result_file, 'r'))
            for metric_key in vbench_metrics:
                if metric_key not in metrics_dict:
                    metrics_dict[metric_key] = []
                metrics_dict[metric_key].append(vbench_metrics[metric_key][0])

    for metric_key in metrics_dict:
        metrics_dict[metric_key] = np.mean(metrics_dict[metric_key])

    assert args.save_path.endswith('.txt'), "The save_path should be a txt file."
    with open(args.save_path, 'w') as f:
        f.write(f"Average Metrics of {args.output_dirs}: \n")
        for metric_key in metrics_dict:
            f.write(f"{metric_key}: {metrics_dict[metric_key]}\n")
    print(f"Averaged Metrics of {args.output_dirs}: \n", metrics_dict)


