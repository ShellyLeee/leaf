import torch.optim as optim


def build_optimizer(params, optim_cfg):
    name = optim_cfg.get('name', 'adam').lower()
    lr = optim_cfg.get('lr', 1e-3)
    weight_decay = optim_cfg.get('weight_decay', 0.0)

    if name == 'sgd':
        return optim.SGD(
            params,
            lr=lr,
            momentum=optim_cfg.get('momentum', 0.9),
            weight_decay=weight_decay,
        )
    if name == 'adam':
        return optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(optim_cfg.get('betas', [0.9, 0.999])),
        )
    if name == 'adamw':
        return optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(optim_cfg.get('betas', [0.9, 0.999])),
        )
    raise ValueError(f'Unsupported optimizer: {name}')
