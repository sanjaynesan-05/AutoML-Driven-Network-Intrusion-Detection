import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report

def evaluate(model, X, y):
    t0 = time.perf_counter(); pred = model.predict(X); latency = time.perf_counter() - t0
    try:
        prob = model.predict_proba(X)[:, 1]
        roc = roc_auc_score(y, prob); pr = average_precision_score(y, prob)
    except (AttributeError, IndexError, ValueError):
        roc = pr = np.nan
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc, "pr_auc": pr, "prediction_seconds": latency,
            "confusion_matrix": confusion_matrix(y, pred).tolist(), "per_class": classification_report(y, pred, output_dict=True, zero_division=0)}
