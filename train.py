import argparse
import os

import torch

from common.config import load_config, save_config
from common.experiments import build_summary
from common.logging import get_logger
from common.loss import build_loss
from common.optim import build_optimizer
from common.scheduler import build_scheduler
from common.seed import seed_everything
from common.utils import ensure_dir, plot_metrics, save_history_csv, save_json
from datasets import build_dataloaders
from models import MLPClassifier
from trainers import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train pure MLP for leaf classification')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--opts', nargs='*', default=[], help='Override config key-values, e.g. model.hidden_dim=256')
    parser.add_argument('--output_dir', type=str, default=None, help='Optional output directory override')
    return parser.parse_args()


def setup_device(device_str):
    if device_str.startswith('cuda') and not torch.cuda.is_available():
        print('[Warning] CUDA requested but unavailable. Falling back to CPU.')
        return torch.device('cpu')
    return torch.device(device_str)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.opts)
    if args.output_dir is not None:
        cfg['output_dir'] = args.output_dir

    seed_everything(cfg.get('seed', 42))

    output_dir = cfg.get('output_dir', './outputs/mlp_baseline')
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, 'plots'))

    logger = get_logger(output_dir)
    save_config(cfg, os.path.join(output_dir, 'config.yaml'))
    exp_name = cfg.get('experiment', {}).get('name', os.path.basename(os.path.abspath(output_dir)))
    logger.info(f'Experiment: {exp_name}')

    device = setup_device(cfg.get('device', 'cpu'))
    logger.info(f'Using device: {device}')

    loaders, data_meta = build_dataloaders(cfg)
    logger.info(f"Split sizes: {data_meta['split_sizes']}")

    model_cfg = cfg['model']
    input_dim = model_cfg.get('input_dim')
    num_classes = model_cfg.get('num_classes')

    if input_dim is None:
        input_dim = data_meta['input_dim']
        logger.info(f'Auto-inferred input_dim={input_dim}')
    if num_classes is None:
        num_classes = data_meta['num_classes']
        logger.info(f'Auto-inferred num_classes={num_classes}')

    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        num_hidden_layers=model_cfg.get('num_hidden_layers', 2),
        hidden_dim=model_cfg.get('hidden_dim', 512),
        hidden_dims=model_cfg.get('hidden_dims', []),
        activation=model_cfg.get('activation', 'relu'),
        use_batchnorm=model_cfg.get('use_batchnorm', False),
        dropout=model_cfg.get('dropout', 0.0),
    ).to(device)

    criterion = build_loss(cfg['loss'])
    optimizer = build_optimizer(model.parameters(), cfg['optimizer'])
    scheduler = build_scheduler(optimizer, cfg['scheduler'])

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        logger=logger,
        monitor=cfg['train'].get('monitor', 'val_acc'),
        grad_clip_norm=cfg['train'].get('grad_clip_norm', None),
        save_best_only=cfg['train'].get('save_best_only', True),
        early_stopping_patience=cfg['train'].get('early_stopping_patience', None),
    )

    history, best_info = trainer.fit(
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        epochs=cfg['train'].get('epochs', 30),
    )

    history_path_json = os.path.join(output_dir, 'history.json')
    history_path_csv = os.path.join(output_dir, 'history.csv')
    save_json(history, history_path_json)
    save_history_csv(history, history_path_csv)
    plot_metrics(history, os.path.join(output_dir, 'plots'))

    save_json(data_meta, os.path.join(output_dir, 'data_meta.json'))
    summary = build_summary(cfg, history, best_info)
    save_json(summary, os.path.join(output_dir, 'summary.json'))

    logger.info(f"Training finished. Best {best_info['monitor']}={best_info['best_metric']:.4f} at epoch {best_info['best_epoch']}")
    logger.info(f'Best checkpoint: {os.path.join(output_dir, "checkpoints", "best.pth")}')
    return summary


if __name__ == '__main__':
    main()
