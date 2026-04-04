import os

import torch
from tqdm import tqdm

from common.metrics import accuracy, macro_f1


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        output_dir,
        logger,
        monitor='val_acc',
        grad_clip_norm=None,
        save_best_only=True,
        early_stopping_patience=None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.logger = logger
        self.monitor = monitor
        self.grad_clip_norm = grad_clip_norm
        self.save_best_only = save_best_only
        self.early_stopping_patience = early_stopping_patience

        self.ckpt_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _run_one_epoch(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        y_true, y_pred = [], []

        pbar = tqdm(loader, desc='train' if train else 'val', leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                    self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy(y_true, y_pred)
        m_f1 = macro_f1(y_true, y_pred)
        return avg_loss, acc, m_f1

    def fit(self, train_loader, val_loader, epochs):
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_macro_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_macro_f1': [],
            'lr': [],
        }

        best_metric = -float('inf')
        best_epoch = -1
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_f1 = self._run_one_epoch(train_loader, train=True)
            val_loss, val_acc, val_f1 = self._run_one_epoch(val_loader, train=False)

            if self.scheduler is not None:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_macro_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_macro_f1'].append(val_f1)
            history['lr'].append(lr)

            monitor_map = {
                'val_acc': val_acc,
                'val_macro_f1': val_f1,
            }
            if self.monitor not in monitor_map:
                raise ValueError(f'Unsupported monitor: {self.monitor}. Use one of {list(monitor_map.keys())}')
            monitored = monitor_map[self.monitor]
            is_best = monitored > best_metric
            if is_best:
                best_metric = monitored
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            self._save_checkpoint(epoch, is_best)

            self.logger.info(
                f"Epoch [{epoch}/{epochs}] | lr={lr:.6f} | "
                f"train loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
                f"val loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
                f"best_{self.monitor}={best_metric:.4f} (epoch {best_epoch})"
            )

            if self.early_stopping_patience is not None and epochs_no_improve >= self.early_stopping_patience:
                self.logger.info(f'Early stopping triggered at epoch {epoch}')
                break

        return history, {'best_metric': best_metric, 'best_epoch': best_epoch, 'monitor': self.monitor}

    def _save_checkpoint(self, epoch, is_best):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': None if self.scheduler is None else self.scheduler.state_dict(),
        }

        latest_path = os.path.join(self.ckpt_dir, 'latest.pth')
        torch.save(ckpt, latest_path)

        if is_best:
            best_path = os.path.join(self.ckpt_dir, 'best.pth')
            torch.save(ckpt, best_path)
        elif not self.save_best_only:
            epoch_path = os.path.join(self.ckpt_dir, f'epoch_{epoch:03d}.pth')
            torch.save(ckpt, epoch_path)
