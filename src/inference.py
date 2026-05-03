import os
import argparse
import logging

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataset import NewTosDataset
from utils import setup_logging, load_jsonl

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LEGAL-BERT inference on new ToS")

    parser.add_argument(
        "--model_path", type=str, required=True, default="models/finetuned_v1",
        help="Path to the fine-tuned model folder"
    )
    parser.add_argument(
        "--input", type=str, required=True, default="data/new_tos/new_tos_segmented.jsonl",
        help="Path to the new ToS JSONL file"
    )
    parser.add_argument(
        "--output", type=str, required=True, default="data/pseudo_labeled/raw_logits.csv",
        help="Path where the CSV with raw logits will be saved"
    )
    parser.add_argument(
        "--max_length", type=int, default=128,
        help="Maximum tokenized sequence length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for inference"
    )

    return parser.parse_args()

def main():
    setup_logging()
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)

    model.to(device)
    model.eval()

    logger.info(f"Loading new ToS from {args.input}...")
    dataset = NewTosDataset(args.input, tokenizer, args.max_length)
    logger.info(f"Sentences to process: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_logits   = []  # collects raw logits for each batch
    all_texts    = []  # collects original texts
    all_companies = [] # collects company names

    logger.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            texts     = batch.pop("text")
            companies = batch.pop("company")

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # outputs.logits shape: [batch_size, 8]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits

            all_logits.append(logits.cpu().numpy())
            all_texts.extend(texts)
            all_companies.extend(companies)

            if (batch_idx + 1) % 10 == 0:
                processed = min((batch_idx + 1) * args.batch_size, len(dataset))
                logger.info(f"  Processed {processed} / {len(dataset)} examples")

    all_logits = np.concatenate(all_logits, axis=0)
    logger.info(f"Inference completed. Logit shape: {all_logits.shape}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = pd.DataFrame({
        "company": all_companies,
        "text":    all_texts,
    })

    for i in range(all_logits.shape[1]):
        df[f"logit_{i}"] = all_logits[:, i]

    df.to_csv(args.output, index=False)
    logger.info(f"Raw logits saved to {args.output} ({len(df)} rows)")

def collate_fn(batch: list[dict]) -> dict:
    return {
        "input_ids":      torch.stack([item["input_ids"]      for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "text":    [item["text"]    for item in batch],
        "company": [item["company"] for item in batch],
    }

if __name__ == "__main__":
    main()