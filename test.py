import argparse
import json
import os

import pandas as pd
import torch

from common.config import load_config
from common.loss import build_loss
from common.metrics import compute_classification_metrics, macro_f1
from datasets import build_dataloaders
from models import MLPClassifier


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate pure MLP on test set')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--submission_path', type=str, default=None)
    parser.add_argument('--opts', nargs='*', default=[], help='Override config key-values, e.g. loader.batch_size=128')
    return parser.parse_args()


def setup_device(device_str):
    if device_str.startswith('cuda') and not torch.cuda.is_available():
        print('[Warning] CUDA requested but unavailable. Falling back to CPU.')
        return torch.device('cpu')
    return torch.device(device_str)


def evaluate(model, criterion, loader, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * images.size(0)
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, y_true, y_pred


def predict_unlabeled(model, loader, device, idx_to_class):
    model.eval()
    image_names, pred_labels = [], []

    with torch.no_grad():
        for images, names in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
            labels = [idx_to_class[p] for p in preds]
            image_names.extend(list(names))
            pred_labels.extend(labels)
    return image_names, pred_labels


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    cfg = load_config(args.config, args.opts)
    device = setup_device(cfg.get('device', 'cpu'))

    loaders, data_meta = build_dataloaders(cfg)

    model_cfg = cfg['model']
    input_dim = model_cfg.get('input_dim') or data_meta['input_dim']
    num_classes = model_cfg.get('num_classes') or data_meta['num_classes']

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

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    criterion = build_loss(cfg['loss'])
    test_loss, y_true, y_pred = evaluate(model, criterion, loaders['test'], device)

    test_acc = sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)
    test_macro_f1 = macro_f1(y_true, y_pred)
    report, cm = compute_classification_metrics(y_true, y_pred, class_names=data_meta['class_names'])

    print(f'test_loss: {test_loss:.6f}')
    print(f'test_accuracy: {test_acc:.6f}')
    print(f'test_macro_f1: {test_macro_f1:.6f}')

    output_dir = cfg.get('output_dir', './outputs/mlp_baseline')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'test_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'test_macro_f1': test_macro_f1,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Detailed metrics saved to: {os.path.join(output_dir, 'test_metrics.json')}")

    predict_loader = loaders.get('predict', None)
    if predict_loader is not None:
        idx_to_class = {int(k): v for k, v in data_meta['idx_to_class'].items()}
        image_names, pred_labels = predict_unlabeled(model, predict_loader, device, idx_to_class)
        submission_df = pd.DataFrame({'image': image_names, 'label': pred_labels})

        submission_path = args.submission_path
        if submission_path is None:
            submission_path = os.path.join(output_dir, 'sample_submission.csv')
        submission_dir = os.path.dirname(submission_path)
        if submission_dir:
            os.makedirs(submission_dir, exist_ok=True)
        submission_df.to_csv(submission_path, index=False)
        print(f'Submission saved to: {submission_path}')


if __name__ == '__main__':
    main()
