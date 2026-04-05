import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.experiments import collect_summary_paths, load_json, write_summary_csv


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate experiment summary.json files into a CSV report.')
    parser.add_argument('--input_root', type=str, default='outputs/experiments')
    parser.add_argument('--output_csv', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_csv = args.output_csv or os.path.join(args.input_root, 'experiment_summary.csv')
    summary_paths = collect_summary_paths(args.input_root)
    rows = [load_json(path) for path in summary_paths]
    write_summary_csv(rows, output_csv)
    print(f'Wrote {len(rows)} experiment rows to {output_csv}')


if __name__ == '__main__':
    main()
