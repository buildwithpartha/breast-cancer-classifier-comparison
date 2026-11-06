"""Shared validation and evaluation; positive class 1 means malignant."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)


def validate_data(frame, features):
    required = features + ['target']
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('The CSV contains no rows.')
    values = frame[required].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('All features and targets must be finite numbers without missing values.')
    if not values.target.isin([0, 1]).all():
        raise ValueError('Target must be 0 (benign) or 1 (malignant).')
    return values[features], values.target.astype(int)


def evaluate(model, X, y):
    predicted = model.predict(X)
    positive_index = list(model.classes_).index(1)
    scores = model.predict_proba(X)[:, positive_index]
    metrics = {
        'Accuracy': accuracy_score(y, predicted),
        'AUC': roc_auc_score(y, scores) if y.nunique() == 2 else None,
        'Precision': precision_score(y, predicted, zero_division=0),
        'Recall': recall_score(y, predicted, zero_division=0),
        'F1': f1_score(y, predicted, zero_division=0),
        'MCC': matthews_corrcoef(y, predicted),
    }
    return metrics, predicted
