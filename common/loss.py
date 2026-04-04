import torch.nn as nn


def build_loss(loss_cfg):
    name = loss_cfg.get('name', 'cross_entropy').lower()
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    raise ValueError(f'Unsupported loss: {name}')
