import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score
from transformers import EvalPrediction


def compute_metrics(p: EvalPrediction) -> dict:
    n_examples = p.label_ids.shape[0]
    n_labels   = p.label_ids.shape[1] #9

    # y_true
    y_true = np.zeros((n_examples, n_labels + 1), dtype=np.int32)
    y_true[:, :-1] = p.label_ids
    y_true[:, -1]  = (np.sum(p.label_ids, axis=1) == 0).astype(np.int32)

    # predizioni
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds  = (expit(logits) > 0.5).astype(np.int32)

    # y_pred
    y_pred = np.zeros((n_examples, n_labels + 1), dtype=np.int32)
    y_pred[:, :-1] = preds
    y_pred[:, -1]  = (np.sum(preds, axis=1) == 0).astype(np.int32)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    return {"macro-f1": macro_f1, "micro-f1": micro_f1}

def apply_threshold(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (expit(logits) > threshold).astype(np.int32)