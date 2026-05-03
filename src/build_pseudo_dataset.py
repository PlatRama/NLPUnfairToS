import argparse
import logging

import numpy as np
import pandas as pd
from scipy.special import expit

from utils import setup_logging, save_csv, print_dataset_stats

logger = logging.getLogger(__name__)

LOGIT_COLUMNS = [f"logit_{i}" for i in range(9)]
LABEL_COLUMNS = ['label_a', 'label_ch', 'label_cr', 'label_j', 'label_law',
                 'label_ltd', 'label_ter', 'label_use', 'label_pinc']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labeled dataset from raw logits"
    )
    parser.add_argument("--input",  type=str, required=True,
                        help="CSV with raw logits produced by inference.py")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV with pseudo-labels")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold to classify a label as active")
    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help=(
                            "Minimum confidence on the most probable label to include a "
                            "positive example. 'none' examples are always kept."
                        ))
    parser.add_argument("--max_samples", type=int, default=None,
                        help=(
                            "Maximum number of samples in the final dataset. "
                            "Sampling maintains proportions between 'none' and positive examples."
                        ))
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for sampling reproducibility")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    logger.info(f"Loading raw logits from {args.input}...")
    df = pd.read_csv(args.input)
    logger.info(f"Examples loaded: {len(df)}")

    missing = [col for col in LOGIT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    logits        = df[LOGIT_COLUMNS].values.astype(np.float32)
    probabilities = expit(logits)
    binary_labels = (probabilities > args.threshold).astype(np.int32)

    if args.min_confidence > 0.0:
        max_positive_prob = np.max(probabilities, axis=1)

        # "none" examples: all prob < threshold → always kept
        # Positive examples: kept only if max_prob >= min_confidence
        is_none      = max_positive_prob < args.threshold
        is_confident = max_positive_prob >= args.min_confidence
        confident_mask = is_none | is_confident

        n_before = len(df)
        df            = df[confident_mask].reset_index(drop=True)
        binary_labels = binary_labels[confident_mask]
        probabilities = probabilities[confident_mask]

        logger.info(f"\nConfidence filter (min={args.min_confidence}, threshold={args.threshold}):")
        logger.info(f"  Total examples          : {n_before}")
        logger.info(f"  'none' examples         : {is_none.sum()} → all kept")
        logger.info(f"  Total positive examples : {(~is_none).sum()}")
        logger.info(f"  Positive examples kept  : {is_confident.sum()} (prob >= {args.min_confidence})")
        logger.info(f"  Examples removed        : {n_before - len(df)}")
        logger.info(f"  Examples after filter   : {len(df)}")
    else:
        logger.info("No confidence filter applied (min_confidence=0.0)")

    if args.max_samples is not None and args.max_samples < len(df):
        is_none_after  = (binary_labels.sum(axis=1) == 0)
        n_none         = is_none_after.sum()
        n_positive     = (~is_none_after).sum()
        total          = len(df)

        ratio_none     = n_none / total
        ratio_positive = n_positive / total

        n_none_sample     = int(args.max_samples * ratio_none)
        n_positive_sample = args.max_samples - n_none_sample

        logger.info(f"\nSampling to {args.max_samples} examples (seed={args.seed}):")
        logger.info(f"  Original proportion : {ratio_none*100:.1f}% none, {ratio_positive*100:.1f}% positive")
        logger.info(f"  'none' examples sampled     : {n_none_sample} / {n_none}")
        logger.info(f"  Positive examples sampled   : {n_positive_sample} / {n_positive}")

        rng = np.random.default_rng(args.seed)

        none_indices     = np.where(is_none_after)[0]
        positive_indices = np.where(~is_none_after)[0]

        sampled_none     = rng.choice(none_indices,     size=n_none_sample,     replace=False)
        sampled_positive = rng.choice(positive_indices, size=n_positive_sample, replace=False)

        sampled_indices = np.concatenate([sampled_none, sampled_positive])
        rng.shuffle(sampled_indices)

        df            = df.iloc[sampled_indices].reset_index(drop=True)
        binary_labels = binary_labels[sampled_indices]

        logger.info(f"  Total examples after sampling: {len(df)}")
    else:
        if args.max_samples is not None:
            logger.info(
                f"max_samples={args.max_samples} >= dataset size={len(df)}, "
                f"no sampling necessary"
            )

    output_df = pd.DataFrame()
    output_df["document"] = df["company"]
    output_df["sentence"] = df["text"]

    for i, col in enumerate(LABEL_COLUMNS):
        output_df[col] = binary_labels[:, i]

    logger.info("\n--- Final dataset statistics ---")
    print_dataset_stats(output_df, "Pseudo-labeled dataset")

    logger.info("\nExamples per company (top 20):")
    for company, count in output_df["document"].value_counts().head(20).items():
        logger.info(f"  {company}: {count} sentences")

    save_csv(output_df, args.output)
    logger.info(f"\nSaved to {args.output} — {len(output_df)} total examples")
    logger.info(f"Ready for training: add --extra_data {args.output} to the train.py command")

if __name__ == "__main__":
    main()