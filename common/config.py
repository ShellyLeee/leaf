import copy
from pathlib import Path

import yaml


def _parse_value(value):
    low = value.lower() if isinstance(value, str) else value
    if low == 'null' or low == 'none':
        return None
    if low == 'true':
        return True
    if low == 'false':
        return False
    try:
        return int(value)
    except Exception:
        try:
            return float(value)
        except Exception:
            return value


def _set_by_dotted_key(d, dotted_key, value):
    keys = dotted_key.split('.')
    cur = d
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def load_config(config_path, overrides=None):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    overrides = overrides or []
    for item in overrides:
        if '=' not in item:
            raise ValueError(f'Invalid override: {item}. Use key=value format')
        k, v = item.split('=', 1)
        _set_by_dotted_key(cfg, k, _parse_value(v))
    return cfg


def save_config(cfg, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
