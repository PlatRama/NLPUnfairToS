import json
import logging
import os

import numpy as np
import pandas as pd
from scipy.special import expit

logger = logging.getLogger(__name__)

NUM_LABELS    = 9
LABEL_COLUMNS = ['label_a', 'label_ch', 'label_cr', 'label_j', 'label_law',
                 'label_ltd', 'label_ter', 'label_use', 'label_pinc']


def setup_logging(log_level: int = logging.INFO):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log_level,
    )


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["sentence"] + LABEL_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} rows to {path}")


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding line {i}: {e}")
    logger.info(f"Loaded {len(records)} records from {path}")
    return records


def logits_to_binary(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (expit(logits) > threshold).astype(np.int32)


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    return expit(logits).astype(np.float32)


def print_label_distribution(df: pd.DataFrame, dataset_name: str = ""):
    header = f"Distribution of labels — {dataset_name}" if dataset_name else "Distribution of labels"
    logger.info(header)
    total = len(df)
    for col in LABEL_COLUMNS:
        if col in df.columns:
            count = df[col].sum()
            pct = count / total * 100
            logger.info(f"  {col}: {int(count):>5} / {total}  ({pct:.1f}%)")
    no_label = (df[LABEL_COLUMNS].sum(axis=1) == 0).sum()
    logger.info(f"  none  : {int(no_label):>5} / {total}  ({no_label/total*100:.1f}%)")


def print_dataset_stats(df: pd.DataFrame, dataset_name: str = ""):
    logger.info(f"\n{'='*50}")
    logger.info(f"Dataset: {dataset_name} — {len(df)} examples")
    labels_per_example = df[LABEL_COLUMNS].sum(axis=1)
    logger.info(f"  Labels per example: mean={labels_per_example.mean():.2f}, "
                f"min={labels_per_example.min()}, max={labels_per_example.max()}")
    print_label_distribution(df, dataset_name)
    logger.info(f"{'='*50}\n")