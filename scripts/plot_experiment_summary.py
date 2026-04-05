import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description='Plot simple per-factor experiment summary charts.')
    parser.add_argument('--summary_csv', type=str, default='outputs/experiments/experiment_summary.csv')
    parser.add_argument('--output_dir', type=str, default='outputs/experiments/summary_plots')
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.summary_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    if df.empty:
        print('Summary CSV is empty, skipping plots.')
        return

    for factor_name, group in df.groupby('factor_name'):
        group = group.sort_values(by='best_val_acc', ascending=False)
        labels = group['factor_value'].astype(str).tolist()
        values = group['best_val_acc'].tolist()

        plt.figure(figsize=(8, 5))
        plt.bar(labels, values)
        plt.xlabel('Factor Value')
        plt.ylabel('Best Val Accuracy')
        plt.title(f'{factor_name} Experiments')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'{factor_name}_best_val_acc.png'))
        plt.close()

    print(f'Saved summary plots to {args.output_dir}')


if __name__ == '__main__':
    main()
