import os
import argparse
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import setup_logging

logger = logging.getLogger(__name__)

CSV_SEP = ";"

LABEL_COLUMNS = [
    "label_a",  # Arbitration
    "label_ch",  # Content removal
    "label_cr",  # Contract by using
    "label_j",  # Jurisdiction
    "label_law",  # Choice of law
    "label_ltd",  # Limitation of liability
    "label_ter",  # Unilateral termination
    "label_use",  # Unilateral change
    "label_pinc",  # Privacy
]

# Columns to keep in final CSVs
COLUMNS_TO_KEEP = ["document", "sentence"] + LABEL_COLUMNS

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test splits for the UNFAIR-ToS dataset"
    )
    parser.add_argument(
        "--input_csv", type=str, required=True,
        help="Path to the original CSV with the 'split' column"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/original/",
        help="Folder where train.csv, val.csv, and test.csv will be saved"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Percentage of training to use as test"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    return parser.parse_args()


def log_split_stats(df: pd.DataFrame, split_name: str, output_path: str):
    logger.info(f"\n── {split_name.upper()} → {output_path}")
    logger.info(f"   Rows             : {len(df)}")
    logger.info(f"   Unique companies : {df['document'].nunique()}")

    for col in LABEL_COLUMNS:
        count = int(df[col].sum())
        pct = count / len(df) * 100
        logger.info(f"   {col:<12}: {count:>5} positive ({pct:.1f}%)")

    none_count = int((df[LABEL_COLUMNS].sum(axis=1) == 0).sum())
    logger.info(f"   {'none':<12}: {none_count:>5} ({none_count / len(df) * 100:.1f}%)")


def save_split(df: pd.DataFrame, path: str):
    df = df[COLUMNS_TO_KEEP].reset_index(drop=True)
    df.to_csv(path, index=False, sep=",")


def main():
    setup_logging()
    args = parse_args()

    logger.info(f"Loading CSV from {args.input_csv}...")
    df = pd.read_csv(args.input_csv, sep=CSV_SEP)
    logger.info(f"Total rows    : {len(df)}")
    logger.info(f"Found columns : {df.columns.tolist()}")

    required = ["split", "sentence", "document"] + LABEL_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in CSV: {missing}\n"
            f"Columns present: {df.columns.tolist()}"
        )

    for col in LABEL_COLUMNS:
        bad = [v for v in df[col].unique() if v not in [0, 1]]
        if bad:
            logger.warning(f"Column {col} contains non-binary values: {bad}")

    n_before = len(df)
    df = df.dropna(subset=["sentence"])
    df = df[df["sentence"].str.strip() != ""]
    if len(df) < n_before:
        logger.warning(f"Removed {n_before - len(df)} rows with empty or NaN sentence")

    found_splits = set(df["split"].unique())
    logger.info(f"Splits found in CSV: {sorted(found_splits)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Case where the test set already exists
    if "test" in found_splits:
        logger.info("\nDetected CASE B: CSV already contains the 'test' split.")
        logger.info("Using splits as they are, no modifications.")

        split_map = {
            "train": "train.csv",
            "validation": "val.csv",
            "test": "test.csv",
        }

        for split_value, filename in split_map.items():
            split_df = df[df["split"] == split_value].copy()
            if len(split_df) == 0:
                logger.warning(f"No rows found for split='{split_value}', skipping file.")
                continue
            output_path = os.path.join(args.output_dir, filename)
            save_split(split_df, output_path)
            log_split_stats(split_df, split_value, output_path)

    # Case where the test set does not exist
    else:
        logger.info("\nDetected CASE A: CSV does not contain the 'test' split.")
        logger.info(
            f"The test set will be derived from training "
            f"({args.test_size * 100:.0f}% of the original train)."
        )

        train_full = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "eval"].copy()

        if len(train_full) == 0:
            raise ValueError("No rows with split='train' found in CSV.")
        if len(val_df) == 0:
            raise ValueError("No rows with split='eval' found in CSV.")

        logger.info(f"Original training   : {len(train_full)} rows")
        logger.info(f"Original validation : {len(val_df)} rows")

        # The dataset is highly imbalanced: most sentences have
        # no active label ("none" class). To ensure train and test
        # have the same proportion of positive/negative examples:
        #
        # DO NOT stratify on all 9 labels simultaneously because some
        # combinations have very few examples and sklearn would raise an error.
        stratify_col = train_full[LABEL_COLUMNS].sum(axis=1).clip(0, 1)

        logger.info(
            f"Stratification distribution — "
            f"negative (none): {(stratify_col == 0).sum()}, "
            f"positive: {(stratify_col == 1).sum()}"
        )

        train_df, test_df = train_test_split(
            train_full,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=stratify_col,  # maintains positive/negative proportion
        )

        logger.info(
            f"\nSplit completed with seed={args.seed}:"
            f"\n  New train  : {len(train_df)} rows"
            f"\n  Test       : {len(test_df)} rows"
            f"\n  Validation : {len(val_df)} rows (unchanged)"
        )

        splits_to_save = [
            ("train", train_df, "train.csv"),
            ("test", test_df, "test.csv"),
            ("validation", val_df, "val.csv"),
        ]

        for split_name, split_df, filename in splits_to_save:
            output_path = os.path.join(args.output_dir, filename)
            save_split(split_df, output_path)
            log_split_stats(split_df, split_name, output_path)

        logger.info("\nVerification of stratification (proportions should be similar):")
        for name, split_df in [("Train", train_df), ("Test", test_df)]:
            pos = (split_df[LABEL_COLUMNS].sum(axis=1) > 0).sum()
            pct = pos / len(split_df) * 100
            logger.info(f"  {name}: {pos}/{len(split_df)} positive ({pct:.1f}%)")

    logger.info(f"\nFiles saved in: {args.output_dir}")
    logger.info("Next step → python src/train.py ...")


if __name__ == "__main__":
    main()