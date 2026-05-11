# NLPUnfairToS

**Semi-supervised Fine-tuning of LEGAL-BERT for Unfair Terms of Service Detection using Pseudo-labeling**

This project investigates whether pseudo-labeling on unlabeled Terms of Service (ToS) documents can improve a LEGAL-BERT model fine-tuned for multi-label classification of potentially unfair clauses. We experiment with three training strategies across multiple dataset sizes and compare them against a supervised baseline.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Label Distribution & Class Imbalance](#label-distribution--class-imbalance)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Pipeline](#pipeline)
7. [Experiments](#experiments)
8. [Results](#results)
9. [Confidence Threshold Ablation](#confidence-threshold-ablation)
10. [Key Findings](#key-findings)
11. [References](#references)

---

## Project Overview

Online platforms frequently embed potentially unfair clauses in their Terms of Service. Manually reviewing these documents is time-consuming and requires legal expertise. This project builds on the CLAUDETTE dataset (Lippi et al., 2019) and LEGAL-BERT to automatically detect and classify such clauses.

**The core research question:** Can pseudo-labeling on a large collection of unlabeled ToS documents improve model performance beyond supervised fine-tuning alone?

**Three training strategies are compared:**
- **V1 — Supervised baseline**: Fine-tune LEGAL-BERT on the labeled dataset only
- **V2 — Combined**: Fine-tune on labeled + pseudo-labeled data simultaneously
- **V3 — Pseudo-label only**: Fine-tune on pseudo-labeled data only (no labeled examples)
- **V1→Pseudo — Continual learning**: Fine-tune V1 further on pseudo-labeled data

---

## Dataset

### Original Labeled Dataset (CLAUDETTE)

The labeled dataset is derived from the CLAUDETTE corpus (Lippi et al., 2019), containing Terms of Service from major online platforms. Each sentence is annotated with one or more of 9 unfairness categories.

| Split      | Examples | Positive examples | None (%) |
|------------|----------|-------------------|----------|
| Train      | 27,130   | 2,148             | 92.1%    |
| Validation | 3,949    | 364               | 90.8%    |
| Test       | 6,783    | 537               | 92.1%    |

**Label distribution per split:**

| Label                     | Train        | Validation  | Test        |
|---------------------------|-------------|-------------|-------------|
| Arbitration               | 103 (0.4%)  | 20 (0.5%)   | 33 (0.5%)   |
| Content Removal           | 366 (1.3%)  | 61 (1.5%)   | 79 (1.2%)   |
| Contract by Using         | 187 (0.7%)  | 33 (0.8%)   | 41 (0.6%)   |
| Jurisdiction              | 119 (0.4%)  | 27 (0.7%)   | 34 (0.5%)   |
| Choice of Law             | 125 (0.5%)  | 27 (0.7%)   | 40 (0.6%)   |
| Limitation of Liability   | 687 (2.5%)  | 109 (2.8%)  | 175 (2.6%)  |
| Unilateral Termination    | 443 (1.6%)  | 75 (1.9%)   | 106 (1.6%)  |
| Unilateral Change         | 273 (1.0%)  | 37 (0.9%)   | 60 (0.9%)   |
| Privacy                   | 84 (0.3%)   | 11 (0.3%)   | 20 (0.3%)   |
| **None**                  | **24,982 (92.1%)** | **3,585 (90.8%)** | **6,246 (92.1%)** |

**Labels per sentence distribution (Train):**

| Labels per sentence | Count  | Percentage |
|--------------------|--------|------------|
| 0 (none)           | 24,982 | 92.1%      |
| 1                  | 1,929  | 7.1%       |
| 2                  | 199    | 0.7%       |
| 3                  | 20     | 0.1%       |

The original CSV uses `;` as separator and contains the following columns:
```
label_a;label_ch;label_cr;label_j;label_law;label_ltd;label_ter;label_use;label_pinc;split;label;document;sentence
```

### New ToS Documents (Pseudo-labeling Source)

New unlabeled ToS documents were sourced from the **ToSDR Terms of Service Corpus** available on Kaggle:

> [https://www.kaggle.com/datasets/sonu1607/tosdr-terms-of-service-corpus](https://www.kaggle.com/datasets/sonu1607/tosdr-terms-of-service-corpus)

This Kaggle dataset contains documents of different types from online platforms, including Terms of Service, Cookie Policies, Privacy Policies, and others. **For this project, only files classified as Terms of Service were used**, filtering out all other document types to maintain consistency with the original CLAUDETTE corpus.

The selected ToS documents were segmented into sentences using spaCy (`en_core_web_sm`), discarding sentences shorter than 5 words following the methodology of Lippi et al. (2019). The segmented output is provided directly in the repository at `data/new_tos/new_tos_segmented.jsonl`, so **Step 1 of the pipeline can be skipped** if you use the provided file.

### Pseudo-labeled Dataset

The segmented ToS sentences were labeled using the fine-tuned V1 model. Only predictions with confidence ≥ 0.9 on positive labels were retained to ensure pseudo-label quality. The `none` class (no active label) is always retained regardless of confidence.

| Dataset                       | Examples | Positive | None (%)  |
|-------------------------------|----------|----------|-----------|
| Pseudo-labeled (conf ≥ 0.9)   | 154,409  | 13,827   | 91.0%     |

**Label distribution in pseudo-labeled dataset:**

| Label                     | Count        | Percentage |
|---------------------------|-------------|------------|
| Arbitration               | 64          | 0.0%       |
| Content Removal           | 3,075       | 2.0%       |
| Contract by Using         | 1,212       | 0.8%       |
| Jurisdiction              | 850         | 0.6%       |
| Choice of Law             | 993         | 0.6%       |
| Limitation of Liability   | 4,134       | 2.7%       |
| Unilateral Termination    | 3,104       | 2.0%       |
| Unilateral Change         | 1,427       | 0.9%       |
| **Privacy**               | **0**       | **0.0%**   |
| None                      | 140,582     | 91.0%      |

**Labels per sentence distribution (Pseudo-labeled):**

| Labels per sentence | Count   | Percentage |
|--------------------|---------|------------|
| 0 (none)           | 140,582 | 91.0%      |
| 1                  | 12,810  | 8.3%       |
| 2                  | 1,002   | 0.6%       |
| 3                  | 15      | 0.0%       |

**Important note on pseudo-label quality:** Two labels have near-zero representation:
- `Privacy` (label_pinc): **0 examples** — the V1 model never predicted this label with ≥ 0.9 confidence on the new documents
- `Arbitration` (label_a): only **64 examples** (0.04%) — extremely rare in the new corpus

This directly impacts the performance of models trained on pseudo-labels for these categories, as discussed in the Key Findings section.

**Comparison of label distributions across datasets:**

| Label                     | Train  | Val    | Test   | Pseudo-labeled |
|---------------------------|--------|--------|--------|----------------|
| Arbitration               | 0.4%   | 0.5%   | 0.5%   | **0.0%**       |
| Content Removal           | 1.3%   | 1.5%   | 1.2%   | 2.0%           |
| Contract by Using         | 0.7%   | 0.8%   | 0.6%   | 0.8%           |
| Jurisdiction              | 0.4%   | 0.7%   | 0.5%   | 0.6%           |
| Choice of Law             | 0.5%   | 0.7%   | 0.6%   | 0.6%           |
| Limitation of Liability   | 2.5%   | 2.8%   | 2.6%   | 2.7%           |
| Unilateral Termination    | 1.6%   | 1.9%   | 1.6%   | 2.0%           |
| Unilateral Change         | 1.0%   | 0.9%   | 0.9%   | 0.9%           |
| Privacy                   | 0.3%   | 0.3%   | 0.3%   | **0.0%**       |
| None                      | 92.1%  | 90.8%  | 92.1%  | 91.0%          |

---

## Label Distribution & Class Imbalance

### The 9 Unfairness Categories

| Label        | Full Name                | Description |
|--------------|--------------------------|-------------|
| `label_a`    | Arbitration              | Clause requiring disputes to be resolved through arbitration |
| `label_ch`   | Content Removal          | Provider can unilaterally remove or modify user content |
| `label_cr`   | Contract by Using        | User accepts the contract simply by using the service |
| `label_j`    | Jurisdiction             | Disputes must be resolved in a specific (foreign) jurisdiction |
| `label_law`  | Choice of Law            | Contract governed by a specific (foreign) law |
| `label_ltd`  | Limitation of Liability  | Provider limits or excludes liability for damages |
| `label_ter`  | Unilateral Termination   | Provider can terminate the contract/service unilaterally |
| `label_use`  | Unilateral Change        | Provider can unilaterally modify the terms |
| `label_pinc` | Privacy                  | Clauses concerning personal data and privacy |

### Class Imbalance

The dataset is **heavily imbalanced**: 92.1% of sentences have no unfair label. Among positive labels, `label_ltd` (Limitation of Liability) is the most frequent at 2.5%, while `label_pinc` (Privacy) is the rarest at 0.3% — an **8x imbalance** between the most and least frequent positive labels.

This imbalance is a key challenge: the model must learn to detect rare categories from very few examples, and pseudo-labeling further amplifies this problem by generating zero examples for the rarest label (Privacy).

---

## Project Structure

```
NLPUnfairToS/
│
├── data/
│   ├── original/
│   │   ├── train.csv                  # labeled training set (produced by prepare_data.py)
│   │   ├── val.csv                    # validation set
│   │   └── test.csv                   # test set (held out, never seen during training)
│   ├── new_tos/
│   │   └── new_tos_segmented.jsonl    # ToS sentences from Kaggle corpus (provided)
│   └── pseudo_labeled/
│       ├── raw_logits.csv             # raw logits from inference (produced by inference.py)
│       ├── pseudo_25k_09.csv          # 25K pseudo-labeled (conf≥0.9)
│       ├── pseudo_50k_09.csv          # 50K pseudo-labeled (conf≥0.9)
│       └── pseudo_all_09.csv          # all pseudo-labeled (conf≥0.9)
│
├── models/
│   ├── finetuned_v1/                  # baseline — original only
│   ├── finetuned_v2/                  # combined — original + pseudo 50K
│   ├── finetuned_v1_25k/              # continual learning — V1 + pseudo 25K
│   ├── finetuned_v1_50k/              # continual learning — V1 + pseudo 50K
│   ├── finetuned_v3_25k/              # pseudo only — 25K
│   ├── finetuned_v3_50k/              # pseudo only — 50K
│   └── finetuned_v3_all/              # pseudo only — all (154K)
│
├── src/
│   ├── prepare_data.py                # splits original CSV into train/val/test
│   ├── dataset.py                     # PyTorch Dataset classes
│   ├── trainer.py                     # MultilabelTrainer with BCE loss
│   ├── metrics.py                     # macro-F1 / micro-F1 computation
│   ├── utils.py                       # shared utilities
│   ├── train.py                       # main training script
│   ├── inference.py                   # inference on new ToS
│   └── build_pseudo_dataset.py        # generates pseudo-labeled CSV
│
├── requirements.txt
└── README.md
```

---

## Pipeline

The full pipeline consists of 5 steps. Run all scripts from the project root directory.

> **Note:** The segmented ToS file (`data/new_tos/new_tos_segmented.jsonl`) is already provided in the repository. You can skip Step 1 and proceed directly to Step 2.

### Step 0 — Prepare the original dataset

The original CSV contains all splits in a single file with a `split` column. This script automatically detects whether a test split exists. If not (our case), it creates one from the training data using stratified sampling (80/20), stratifying on the binary variable "has at least one active label" to preserve class proportions.

```bash
python src/prepare_data.py \
    --input_csv  data/original/unfair_tos.csv \
    --output_dir data/original/
```

### Step 1 — Segment new ToS documents (optional, file already provided)

If you want to reproduce the segmentation from scratch, run the spaCy script on your `.txt` files. Sentences shorter than 5 words are discarded following Lippi et al. (2019). The source documents come from the [ToSDR Terms of Service Corpus on Kaggle](https://www.kaggle.com/datasets/sonu1607/tosdr-terms-of-service-corpus) — only files classified as Terms of Service were used.

```bash
python src/segment_tos.py \
    --input_folder data/new_tos/raw/ \
    --output_file  data/new_tos/new_tos_segmented.jsonl
```

### Step 2 — Fine-tune on the original labeled dataset (V1 baseline)

```bash
python src/train.py \
    --model_name    nlpaueb/legal-bert-base-uncased \
    --train_data    data/original/train.csv \
    --val_data      data/original/val.csv \
    --test_data     data/original/test.csv \
    --output_dir    models/finetuned_v1 \
    --num_epochs    10 \
    --batch_size    8 \
    --max_length    128 \
    --learning_rate 2e-5
```

### Step 3 — Generate pseudo-labels

**Step 3a** — Run inference with V1 on the segmented ToS:

```bash
python src/inference.py \
    --model_path models/finetuned_v1 \
    --input      data/new_tos/new_tos_segmented.jsonl \
    --output     data/pseudo_labeled/raw_logits.csv \
    --batch_size 32
```

**Step 3b** — Convert logits to pseudo-labels with confidence filtering:

```bash
# Full dataset (confidence >= 0.9)
python src/build_pseudo_dataset.py \
    --input          data/pseudo_labeled/raw_logits.csv \
    --output         data/pseudo_labeled/pseudo_all_09.csv \
    --min_confidence 0.9

# Capped at 25K examples (stratified sampling, seed=42)
python src/build_pseudo_dataset.py \
    --input          data/pseudo_labeled/raw_logits.csv \
    --output         data/pseudo_labeled/pseudo_25k_09.csv \
    --min_confidence 0.9 \
    --max_samples    25000

# Capped at 50K examples
python src/build_pseudo_dataset.py \
    --input          data/pseudo_labeled/raw_logits.csv \
    --output         data/pseudo_labeled/pseudo_50k_09.csv \
    --min_confidence 0.9 \
    --max_samples    50000
```

**Confidence filtering logic:**
- `none` examples (no active label) are **always kept** regardless of confidence
- Positive examples are kept only if `max(sigmoid(logits)) >= min_confidence`
- Sampling with `--max_samples` preserves the original none/positive ratio

### Step 4 — Run all experiments

**V2 — Combined (original + pseudo-labeled):**

```bash
python src/train.py \
    --model_name  nlpaueb/legal-bert-base-uncased \
    --train_data  data/original/train.csv \
    --extra_data  data/pseudo_labeled/pseudo_50k_09.csv \
    --val_data    data/original/val.csv \
    --test_data   data/original/test.csv \
    --output_dir  models/finetuned_v2 \
    --num_epochs  10 --batch_size 8 --max_length 128
```

**V3 — Pseudo-label only (ablation on dataset size):**

```bash
# 25K
python src/train.py \
    --model_name nlpaueb/legal-bert-base-uncased \
    --train_data data/pseudo_labeled/pseudo_25k_09.csv \
    --val_data   data/original/val.csv \
    --test_data  data/original/test.csv \
    --output_dir models/finetuned_v3_25k \
    --num_epochs 10 --batch_size 8 --max_length 128

# 50K
python src/train.py \
    --model_name nlpaueb/legal-bert-base-uncased \
    --train_data data/pseudo_labeled/pseudo_50k_09.csv \
    --val_data   data/original/val.csv \
    --test_data  data/original/test.csv \
    --output_dir models/finetuned_v3_50k \
    --num_epochs 10 --batch_size 8 --max_length 128

# All (154K)
python src/train.py \
    --model_name nlpaueb/legal-bert-base-uncased \
    --train_data data/pseudo_labeled/pseudo_all_09.csv \
    --val_data   data/original/val.csv \
    --test_data  data/original/test.csv \
    --output_dir models/finetuned_v3_all \
    --num_epochs 10 --batch_size 8 --max_length 128
```

**V1→Pseudo — Continual learning (start from V1, fine-tune on pseudo-labels):**

```bash
# 25K
python src/train.py \
    --model_name    models/finetuned_v1 \
    --train_data    data/pseudo_labeled/pseudo_25k_09.csv \
    --val_data      data/original/val.csv \
    --test_data     data/original/test.csv \
    --output_dir    models/finetuned_v1_25k \
    --num_epochs    5 --batch_size 8 --max_length 128 \
    --learning_rate 5e-6

# 50K
python src/train.py \
    --model_name    models/finetuned_v1 \
    --train_data    data/pseudo_labeled/pseudo_50k_09.csv \
    --val_data      data/original/val.csv \
    --test_data     data/original/test.csv \
    --output_dir    models/finetuned_v1_50k \
    --num_epochs    5 --batch_size 8 --max_length 128 \
    --learning_rate 5e-6
```

---

## Experiments

| Model      | Training Data          | LR    | Epochs (early stop) |
|------------|------------------------|-------|---------------------|
| V1         | Original (27K)         | 2e-5  | 6                   |
| V2         | Original + Pseudo 50K  | 2e-5  | 8                   |
| V3 25K     | Pseudo 25K             | 2e-5  | 9                   |
| V3 50K     | Pseudo 50K             | 2e-5  | 6                   |
| V3 All     | Pseudo 154K            | 2e-5  | 7                   |
| V1→25K     | Pseudo 25K             | 5e-6  | 4                   |
| V1→50K     | Pseudo 50K             | 5e-6  | 5                   |

**Common hyperparameters for all models:**
- Base model: `nlpaueb/legal-bert-base-uncased`
- Max sequence length: 128 tokens
- Batch size: 8
- Warmup ratio: 0.1
- Early stopping patience: 3 epochs
- Loss function: Binary Cross-Entropy (BCE) with sigmoid activation
- Primary evaluation metric: macro-F1 (computed over all 9 labels + `none` class)

---

## Results

### Global Metrics on Test Set

| Model                      | Macro-F1    | Micro-F1 | Test Loss |
|---------------------------|-------------|----------|-----------|
| **V1 — Original only**    | **0.7881**  | 0.9613   | 0.0145    |
| V2 — Combined (orig+50K)  | 0.7771      | 0.9659   | 0.0221    |
| V3 25K — Pseudo only      | 0.7295      | 0.9659   | 0.0337    |
| V3 50K — Pseudo only      | 0.7136      | 0.9637   | 0.0308    |
| V3 All — Pseudo only      | 0.7183      | 0.9633   | 0.0406    |
| V1→25K — Continual        | 0.7194      | 0.9659   | 0.0199    |
| V1→50K — Continual        | 0.7144      | 0.9643   | 0.0285    |

### Comparison with CLAUDETTE Paper (Lippi et al., 2019)

| System                           | F1    |
|----------------------------------|-------|
| Ensemble C8 — best system        | 0.806 |
| Combined SVM C2                  | 0.784 |
| **V1 — LEGAL-BERT (ours)**       | **0.788** |

Our V1 baseline achieves **0.788 macro-F1**, surpassing the combined SVM from the original paper and approaching the ensemble of 5 systems using a single model.

---

## Confidence Threshold Ablation

This experiment investigates how the confidence threshold used during pseudo-label generation affects model performance when training from scratch on the combined dataset (original 27K + 25K pseudo-labeled examples).

All models in this ablation are trained with:
- Base model: `nlpaueb/legal-bert-base-uncased` (from scratch, not from V1)
- Training data: original labeled dataset (27K) + 25K pseudo-labeled examples
- Pseudo-labeled examples sampled with `--max_samples 25000 --seed 42`
- Confidence threshold varies: 0.5, 0.6, 0.7, 0.8, 0.9
- All other hyperparameters identical to other experiments

To reproduce this ablation:

```bash
# Generate pseudo-labeled datasets at different confidence thresholds
for CONF in 0.5 0.6 0.7 0.8 0.9; do
    python src/build_pseudo_dataset.py \
        --input          data/pseudo_labeled/raw_logits.csv \
        --output         data/pseudo_labeled/pseudo_25k_0${CONF//./}.csv \
        --min_confidence $CONF \
        --max_samples    25000 \
        --seed           42
done

# Train a model for each threshold
for CONF in 0.5 0.6 0.7 0.8 0.9; do
    python src/train.py \
        --model_name  nlpaueb/legal-bert-base-uncased \
        --train_data  data/original/train.csv \
        --extra_data  data/pseudo_labeled/pseudo_25k_0${CONF//./}.csv \
        --val_data    data/original/val.csv \
        --test_data   data/original/test.csv \
        --output_dir  models/conf_ablation_${CONF//./} \
        --num_epochs  10 --batch_size 8 --max_length 128
done
```

### Validation Set Results

| Confidence | Macro-F1 | Micro-F1 | Eval Loss | Epochs |
|------------|----------|----------|-----------|--------|
| 0.5        | 0.7902   | 0.9529   | 0.0336    | 10     |
| 0.6        | 0.7857   | 0.9524   | 0.0260    | 8      |
| 0.7        | 0.7761   | 0.9544   | 0.0205    | 6      |
| 0.8        | 0.7863   | 0.9542   | 0.0256    | 8      |
| 0.9        | 0.7854   | 0.9576   | 0.0276    | 8      |
| **V1 baseline (no pseudo)** | **0.7656** | **0.9493** | **0.0187** | **6** |

### Test Set Results

| Confidence | Macro-F1    | Micro-F1 | Test Loss | Epochs |
|------------|-------------|----------|-----------|--------|
| 0.5        | 0.7771      | 0.9616   | 0.0262    | 10     |
| 0.6        | 0.7765      | 0.9646   | 0.0200    | 8      |
| 0.7        | 0.7682      | 0.9647   | 0.0161    | 6      |
| 0.8        | 0.7753      | 0.9671   | 0.0202    | 8      |
| 0.9        | 0.7771      | 0.9659   | 0.0221    | 8      |
| **V1 baseline (no pseudo)** | **0.7881** | **0.9613** | **0.0145** | **6** |

### Observations

**No confidence threshold improves over V1 on the test set.** All combined models (original + 25K pseudo) score below the supervised baseline (0.7881) on macro-F1, confirming that the noise introduced by pseudo-labels partially offsets any benefit from the additional data volume.

**Confidence 0.5 achieves the best validation macro-F1 (0.7902)** — higher than V1's validation score (0.7656) — but does not generalise as well to the test set (0.7771 vs 0.7881). It also required all 10 epochs to converge without triggering early stopping, suggesting the model keeps slowly improving on validation without reaching a clear peak.

**Confidence 0.7 is the worst performer** on both validation (0.7761) and test (0.7682). At this threshold the pseudo-labeled positives are filtered enough to reduce volume, but not enough to eliminate noisy examples — a middle ground that yields the worst of both worlds.

**Confidence 0.5 and 0.9 tie on test macro-F1 (0.7771)**, while confidence 0.8 and 0.9 are the closest competitors at 0.7753 and 0.7771 respectively. The relationship between threshold and performance is not monotonic: 0.5 ≥ 0.9 > 0.6 > 0.8 > 0.7 on test macro-F1.

**Micro-F1 increases with higher confidence thresholds** (0.9616 at conf=0.5 up to 0.9671 at conf=0.8), reflecting that stricter filtering keeps only the most unambiguous positive examples, which helps the model learn frequent labels more precisely without affecting the dominant `none` class.

**The gap between validation and test macro-F1 is largest for confidence 0.5** (0.7902 val vs 0.7771 test, delta=0.013), suggesting slight overfitting to the validation distribution when more pseudo-label noise is included.

---

## Key Findings

### 1. Labeled data quality beats pseudo-label quantity
V1 trained on 27K labeled examples outperforms all pseudo-label strategies, including V3 All trained on 154K pseudo-labeled examples (0.788 vs 0.718). High-quality labeled data cannot be replaced by volume alone.

### 2. The Privacy label reveals the limits of pseudo-labeling
`label_pinc` (Privacy) has **0 examples** in the pseudo-labeled dataset — the V1 model never reached 0.9 confidence on this label for any of the new documents. As a consequence, V3 models score F1=0.000 on Privacy, while V1 achieves F1=0.533. This demonstrates that **pseudo-labeling inherits and amplifies the teacher model's biases**: if the teacher is uncertain about a category, that category disappears from the pseudo-labeled data.

### 3. Combined training does not improve over the baseline
V2 (original + 50K pseudo-labeled) scores macro-F1=0.777, slightly below V1 (0.788). Adding pseudo-labeled data introduces noise that partially offsets any benefit from the increased data volume. Micro-F1 does improve marginally (0.966 vs 0.961), suggesting pseudo-labels help on frequent labels but hurt performance on rare ones.

### 4. No clear benefit from increasing pseudo-label dataset size
Comparing V3 25K (0.7295), V3 50K (0.7136), and V3 All (0.7183) shows no monotonic improvement as more pseudo-labeled examples are added. The model saturates quickly and begins to overfit to pseudo-label noise, as evidenced by the rising test loss from V3 25K (0.034) to V3 All (0.041).

### 5. Continual learning suffers from catastrophic forgetting
V1→25K (0.7194) and V1→50K (0.7144) both underperform V1 (0.788) despite starting from the strong supervised baseline. Fine-tuning on pseudo-labels overwrites knowledge about rare labels acquired during supervised training, particularly Privacy (F1 drops to near 0) and Arbitration.

### 6. Micro-F1 is stable across all strategies
All models achieve micro-F1 between 0.961 and 0.966, showing that the dominant `none` class and frequent positive labels are consistently learned. The critical challenge lies in **rare label detection**, which only supervised fine-tuning on gold-labeled data handles effectively.

### 7. Confidence threshold has no clear optimal value
The confidence threshold ablation (0.5, 0.6, 0.7, 0.9 tested) shows no monotonic relationship between threshold and test macro-F1. Confidence 0.5 performs best on validation (0.790) but not on test (0.777), while confidence 0.7 is the worst on both sets. All thresholds produce models below the V1 supervised baseline on test macro-F1, suggesting that **the problem lies in the pseudo-label quality and coverage rather than the filtering threshold**.

---

## References

Lippi, M., Palka, P., Contissa, G., Lagioia, F., Micklitz, H. W., Sartor, G., & Torroni, P. (2019). *CLAUDETTE: an Automated Detector of Potentially Unfair Clauses in Online Terms of Service*. Artificial Intelligence and Law. https://doi.org/10.1007/s10506-019-09243-2

Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I. (2020). *LEGAL-BERT: The Muppets straight out of Law School*. EMNLP Findings.

ToSDR Terms of Service Corpus. Kaggle. https://www.kaggle.com/datasets/sonu1607/tosdr-terms-of-service-corpus