import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import load_config
from common.config import save_config
from common.experiments import BASELINE_EXPERIMENT
from common.experiments import build_experiment_specs
from common.experiments import collect_summary_paths
from common.experiments import load_json
from common.experiments import write_summary_csv
from common.utils import ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description='Run single-factor batch experiments for leaf MLP.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--only', nargs='*', default=None, help='Optional subset of factors, e.g. --only lr hidden_dim')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--output_root', type=str, default='outputs/experiments')
    parser.add_argument('--device', type=str, default=None, help='Fallback device when GPUs are not specified')
    parser.add_argument('--gpus', type=str, default=None, help='Comma-separated GPU ids, e.g. 0,1,2')
    parser.add_argument('--max_parallel', type=int, default=None, help='Maximum concurrent experiments')
    return parser.parse_args()


def parse_gpu_list(gpu_arg):
    if not gpu_arg:
        return []
    return [item.strip() for item in gpu_arg.split(',') if item.strip()]


def append_log(log_path, message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'{timestamp} | {message}'
    print(line)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def build_command(spec, fallback_device=None):
    train_entry = os.path.join(PROJECT_ROOT, 'train.py')
    config_path = os.path.join(spec['output_dir'], 'config.yaml')
    command = [sys.executable, train_entry, '--config', config_path]
    if fallback_device:
        command.extend(['--opts', f'device={fallback_device}'])
    return command


def build_env(gpu_id):
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    return env


def run_dry_run(specs):
    print(f'Baseline: {BASELINE_EXPERIMENT}')
    print(f'Planned experiments: {len(specs)}')
    for spec in specs:
        print(f'- {spec["exp_name"]}: {spec["factor_name"]}={spec["factor_value"]} -> {spec["output_dir"]}')


def run_serial(specs, args, log_path):
    failures = []
    for spec in specs:
        command = build_command(spec, fallback_device=args.device)
        append_log(log_path, f'Start exp {spec["exp_name"]} on {"device " + args.device if args.device else "default device"}')
        result = subprocess.run(command, env=build_env(None), check=False)
        if result.returncode == 0:
            append_log(log_path, f'Finish exp {spec["exp_name"]}')
        else:
            failures.append(spec['exp_name'])
            append_log(log_path, f'Fail exp {spec["exp_name"]} exit_code={result.returncode}')
    return failures


def run_parallel(specs, args, gpu_ids, log_path):
    max_parallel = args.max_parallel or len(gpu_ids)
    max_parallel = max(1, min(max_parallel, len(gpu_ids)))
    available_gpus = gpu_ids[:]
    pending = list(specs)
    running = []
    failures = []

    while pending or running:
        while pending and available_gpus and len(running) < max_parallel:
            gpu_id = available_gpus.pop(0)
            spec = pending.pop(0)
            command = build_command(spec, fallback_device=args.device)
            process = subprocess.Popen(
                command,
                env=build_env(gpu_id),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            running.append({'process': process, 'gpu_id': gpu_id, 'spec': spec})
            append_log(log_path, f'Start exp {spec["exp_name"]} on GPU {gpu_id}')

        time.sleep(2)
        still_running = []
        for item in running:
            return_code = item['process'].poll()
            if return_code is None:
                still_running.append(item)
                continue

            spec = item['spec']
            gpu_id = item['gpu_id']
            available_gpus.append(gpu_id)
            available_gpus.sort(key=lambda value: int(value))
            if return_code == 0:
                append_log(log_path, f'Finish exp {spec["exp_name"]} on GPU {gpu_id}')
            else:
                failures.append(spec['exp_name'])
                append_log(log_path, f'Fail exp {spec["exp_name"]} on GPU {gpu_id} exit_code={return_code}')
        running = still_running

    return failures


def main():
    args = parse_args()
    ensure_dir(args.output_root)
    log_path = os.path.join(args.output_root, 'parallel_run.log')

    base_cfg = load_config(args.config)
    specs = build_experiment_specs(base_cfg, args.output_root, only_factors=args.only)
    for spec in specs:
        ensure_dir(spec['output_dir'])
        save_config(spec['config'], os.path.join(spec['output_dir'], 'config.yaml'))
    save_json({'baseline': BASELINE_EXPERIMENT, 'experiments': specs}, os.path.join(args.output_root, 'experiment_plan.json'))

    if args.dry_run:
        run_dry_run(specs)
        return

    gpu_ids = parse_gpu_list(args.gpus)
    if gpu_ids:
        failures = run_parallel(specs, args, gpu_ids, log_path)
    else:
        failures = run_serial(specs, args, log_path)

    summary_rows = [load_json(path) for path in collect_summary_paths(args.output_root)]
    summary_csv = os.path.join(args.output_root, 'experiment_summary.csv')
    write_summary_csv(summary_rows, summary_csv)
    append_log(log_path, f'Wrote summary CSV with {len(summary_rows)} rows to {summary_csv}')
    append_log(log_path, f'Completed {len(specs) - len(failures)}/{len(specs)} experiments')
    if failures:
        append_log(log_path, f'Failed experiments: {", ".join(failures)}')


if __name__ == '__main__':
    main()
