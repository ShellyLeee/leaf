from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def build_scheduler(optimizer, scheduler_cfg):
    name = scheduler_cfg.get('name', 'none').lower()
    if name == 'none':
        return None
    if name == 'steplr':
        return StepLR(
            optimizer,
            step_size=scheduler_cfg.get('step_size', 10),
            gamma=scheduler_cfg.get('gamma', 0.1),
        )
    if name == 'cosineannealinglr':
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get('t_max', 50),
        )
    raise ValueError(f'Unsupported scheduler: {name}')
