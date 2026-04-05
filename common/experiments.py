import copy
import csv
import json
import os


BASELINE_EXPERIMENT = {
    'layers': 2,
    'hidden_dim': 512,
    'lr': 1e-3,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'batchnorm': False,
    'dropout': 0.0,
    'epochs': 30,
}


EXPERIMENT_FACTORS = {
    'lr': [1e-2, 1e-3, 1e-4],
    'hidden_dim': [128, 256, 512, 1024],
    'layers': [1, 2, 4, 6],
    'batchnorm': [False, True],
    'dropout': [0.0, 0.2, 0.5],
    'optimizer': ['adam', 'sgd'],
    'weight_decay': [0.0, 1e-5, 1e-4, 1e-3],
    'epochs': [10, 20, 50, 100],
}


def _format_factor_value(value):
    if isinstance(value, bool):
        return 'on' if value else 'off'
    text = str(value)
    return text.replace('.', 'p')


def _experiment_name(factor_name, value):
    aliases = {
        'lr': 'lr',
        'hidden_dim': 'hd',
        'layers': 'layers',
        'batchnorm': 'bn',
        'dropout': 'drop',
        'optimizer': 'opt',
        'weight_decay': 'wd',
        'epochs': 'ep',
    }
    prefix = aliases.get(factor_name, factor_name)
    return f'{prefix}_{_format_factor_value(value)}'


def _set_nested(cfg, dotted_key, value):
    keys = dotted_key.split('.')
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def build_experiment_specs(base_cfg, output_root, only_factors=None):
    factors = only_factors or list(EXPERIMENT_FACTORS.keys())
    specs = []

    for factor_name in factors:
        if factor_name not in EXPERIMENT_FACTORS:
            raise ValueError(f'Unknown factor: {factor_name}. Expected one of {list(EXPERIMENT_FACTORS)}')

        for value in EXPERIMENT_FACTORS[factor_name]:
            exp_cfg = copy.deepcopy(base_cfg)
            exp_name = _experiment_name(factor_name, value)

            exp_cfg['output_dir'] = os.path.join(output_root, exp_name)
            exp_cfg.setdefault('experiment', {})
            exp_cfg['experiment'].update({
                'name': exp_name,
                'factor_name': factor_name,
                'factor_value': value,
                'baseline': copy.deepcopy(BASELINE_EXPERIMENT),
            })

            _apply_single_factor(exp_cfg, factor_name, value)

            specs.append({
                'exp_name': exp_name,
                'factor_name': factor_name,
                'factor_value': value,
                'config': exp_cfg,
                'output_dir': exp_cfg['output_dir'],
            })

    return specs


def _apply_single_factor(cfg, factor_name, value):
    mapping = {
        'lr': [('optimizer.lr', value)],
        'hidden_dim': [('model.hidden_dim', value)],
        'layers': [('model.num_hidden_layers', value)],
        'batchnorm': [('model.use_batchnorm', value)],
        'dropout': [('model.dropout', value)],
        'weight_decay': [('optimizer.weight_decay', value)],
        'epochs': [('train.epochs', value)],
    }

    if factor_name == 'optimizer':
        updates = [
            ('optimizer.name', value),
            ('optimizer.lr', 1e-2 if str(value).lower() == 'sgd' else 1e-3),
        ]
    else:
        updates = mapping[factor_name]

    for dotted_key, update_value in updates:
        _set_nested(cfg, dotted_key, update_value)


def _best_epoch_index(best_info, history):
    best_epoch = best_info.get('best_epoch', -1)
    if best_epoch is None or best_epoch <= 0:
        if history.get('val_acc'):
            return max(range(len(history['val_acc'])), key=lambda idx: history['val_acc'][idx])
        return None
    return best_epoch - 1


def build_summary(cfg, history, best_info):
    idx = _best_epoch_index(best_info, history)
    exp_meta = cfg.get('experiment', {})
    model_cfg = cfg.get('model', {})
    optimizer_cfg = cfg.get('optimizer', {})
    train_cfg = cfg.get('train', {})

    summary = {
        'exp_name': exp_meta.get('name', os.path.basename(os.path.abspath(cfg.get('output_dir', '')))),
        'factor_name': exp_meta.get('factor_name', 'manual'),
        'factor_value': exp_meta.get('factor_value', None),
        'best_epoch': best_info.get('best_epoch', None),
        'best_val_acc': None,
        'best_val_macro_f1': None,
        'optimizer': optimizer_cfg.get('name'),
        'lr': optimizer_cfg.get('lr'),
        'weight_decay': optimizer_cfg.get('weight_decay'),
        'hidden_dim': model_cfg.get('hidden_dim'),
        'num_hidden_layers': model_cfg.get('num_hidden_layers'),
        'batchnorm': model_cfg.get('use_batchnorm'),
        'dropout': model_cfg.get('dropout'),
        'epochs': train_cfg.get('epochs'),
        'monitor': best_info.get('monitor'),
        'output_dir': cfg.get('output_dir'),
    }

    if idx is not None:
        if idx < len(history.get('val_acc', [])):
            summary['best_val_acc'] = history['val_acc'][idx]
        if idx < len(history.get('val_macro_f1', [])):
            summary['best_val_macro_f1'] = history['val_macro_f1'][idx]

    return summary


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_summary_paths(root_dir):
    summary_paths = []
    for current_root, _, files in os.walk(root_dir):
        if 'summary.json' in files:
            summary_paths.append(os.path.join(current_root, 'summary.json'))
    return sorted(summary_paths)


def write_summary_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not rows:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write('')
        return

    fieldnames = [
        'exp_name',
        'factor_name',
        'factor_value',
        'best_epoch',
        'best_val_acc',
        'best_val_macro_f1',
        'optimizer',
        'lr',
        'weight_decay',
        'hidden_dim',
        'num_hidden_layers',
        'batchnorm',
        'dropout',
        'epochs',
        'monitor',
        'output_dir',
    ]

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
