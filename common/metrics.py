import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0))


def compute_classification_metrics(y_true, y_pred, class_names=None):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    return report, cm


def safe_mean(values):
    if not values:
        return 0.0
    return float(np.mean(values))
