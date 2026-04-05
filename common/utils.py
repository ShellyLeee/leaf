import json
import os

import pandas as pd


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_history_csv(history, path):
    pd.DataFrame(history).to_csv(path, index=False)


def plot_metrics(history, save_dir):
    import matplotlib.pyplot as plt

    ensure_dir(save_dir)

    epochs = list(range(1, len(history['train_loss']) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_acc'], label='train_acc')
    plt.plot(epochs, history['val_acc'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'acc_curve.png'))
    plt.close()

    if 'train_macro_f1' in history and len(history['train_macro_f1']) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history['train_macro_f1'], label='train_macro_f1')
        plt.plot(epochs, history['val_macro_f1'], label='val_macro_f1')
        plt.xlabel('Epoch')
        plt.ylabel('Macro-F1')
        plt.title('Macro-F1 Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'macro_f1_curve.png'))
        plt.close()
