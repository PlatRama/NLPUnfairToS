import os
import argparse
import logging
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
)

from trainer import MultilabelTrainer
from dataset import TosDataset, CombinedDataset
from utils import setup_logging, load_csv, print_dataset_stats
from metrics import compute_metrics

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning LEGAL-BERT on UNFAIR-ToS")

    parser.add_argument("--model_name", type=str, default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data",   type=str, required=True)
    parser.add_argument("--test_data",  type=str, default=None)
    parser.add_argument("--extra_data", type=str, default=None)

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Starting training LEGAL-BERT on UNFAIR-ToS")
    logger.info(f"Model      : {args.model_name}")
    logger.info(f"Train data : {args.train_data}")
    logger.info(f"Extra data : {args.extra_data or 'None (Step 1)'}")
    logger.info(f"Val data   : {args.val_data}")
    logger.info(f"Output dir : {args.output_dir}")
    logger.info("=" * 60)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=True)

    logger.info("Loading data...")
    train_dataset = TosDataset(args.train_data, tokenizer, args.max_length)
    print_dataset_stats(load_csv(args.train_data), "Original train")

    if args.extra_data is not None:
        logger.info(f"Loading pseudo-labeled data from {args.extra_data}...")
        pseudo_dataset = TosDataset(args.extra_data, tokenizer, args.max_length)
        print_dataset_stats(load_csv(args.extra_data), "Pseudo-labeled")
        train_dataset = CombinedDataset(train_dataset, pseudo_dataset)
        logger.info(f"Combined dataset: {len(train_dataset)} total examples")
    else:
        logger.info(f"Dataset: {len(train_dataset)} examples (Step 1, no extra data)")

    val_dataset = TosDataset(args.val_data, tokenizer, args.max_length)
    logger.info(f"Validation set: {len(val_dataset)} examples")

    logger.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=9,
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro-f1",
        greater_is_better=True,

        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,

        seed=args.seed,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_model()
    logger.info(f"Model saved at {args.output_dir}")

    logger.info("Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    trainer.log_metrics("eval", val_metrics)
    trainer.save_metrics("eval", val_metrics)
    logger.info(f"Validation — macro-f1: {val_metrics['eval_macro-f1']:.4f} | "
                f"micro-f1: {val_metrics['eval_micro-f1']:.4f}")

    if args.test_data is not None:
        logger.info("Evaluating on test set...")
        test_dataset = TosDataset(args.test_data, tokenizer, args.max_length)
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        logger.info(f"Test — macro-f1: {test_metrics['test_macro-f1']:.4f} | "
                    f"micro-f1: {test_metrics['test_micro-f1']:.4f}")

    logger.info("Training finished.")


if __name__ == "__main__":
    main()