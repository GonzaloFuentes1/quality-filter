# Quality Classifier

This repository provides tools to label, fine-tune, evaluate, and test text quality across multiple dimensions using large language models (LLMs). It supports high-throughput evaluation, multi-language detection, and robust model training for multi-label classification.

## Repository Structure

```
quality-classifier/
├── src/
│   ├── label_quality.py                # LLM-based labeling
│   ├── split_languajes.py             # Language detection & splitting
│   ├── train_bert.py                  # Fine-tuning BERT for multilabel classification
│   ├── test_bert.py                   # Evaluate trained BERT models on test set
├── prompts/
│   └── base_prompt.txt                # Base prompt with placeholder
├── data/
│   └── textos_500k/                   # Saved truncated evaluation texts
├── requirements.txt
└── README.md
```

## Installation

1. **Create virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

## Usage

### 1. LLM-based Quality Labeling

The `label_quality.py` script uses LLMs (via vLLM) to evaluate texts based on various quality metrics:

- `coherencia`
- `desinformacion`
- `representacion_latinoamericana`
- `nivel_educacional`
- `originalidad`
- `etapa_1_valida`
- `score_final`

The prompt must include the placeholder `[PEGUE AQUÍ EL TEXTO]` which is replaced for each input.

**Example:**

```bash
python src/label_quality.py \
    --prompt_path prompts/base_prompt.txt \
    --dataset_path <path_to_hf_dataset> \
    --model_path nvidia/Llama-3_3-Nemotron-Super-49B-v1 \
    --download_dir <hf_model_dir> \
    --output_path outputs/resultados.json \
    --text_column texto \
    --batch_size 50 \
    --tensor_parallel_size 8
```

### 2. Language Detection & Splitting

Split a JSON file by language (Spanish, English, Portuguese) using `langdetect` in parallel:

```bash
python src/split_languajes.py
```

**Modifies inside:**

- Input: `data/test.json`
- Output: `test_textos_es_.json`, `test_textos_en_.json`, etc.

### 3. BERT-based Multilabel Classification

Use `train_bert.py` to train a BERT-based model that predicts quality scores per category. Each label is in [0.0, 1.0] and thresholded to create binary labels.

**Example:**

```bash
python src/train_bert.py \
    --json_path outputs/resultados.json \
    --output_dir models/ \
    --logging_dir logs/ \
    --model_name answerdotai/ModernBERT-base \
    --label_ids 0 1 2 3 4 5 \
    --threshold 0.5 \
    --balance_method oversample
```

#### Labels and IDs

| ID  | Label                          |
| --- | ------------------------------ |
| 0   | coherencia                     |
| 1   | desinformacion                 |
| 2   | representacion_latinoamericana |
| 3   | nivel_educacional              |
| 4   | originalidad                   |
| 5   | score_final                    |

**Balance Options:** `oversample`, `undersample`, or leave empty.

**Model Options:** Any HuggingFace Transformer model compatible with `AutoModelForSequenceClassification`.

### 4. Testing Trained Models

The script `test_bert.py` allows for batch evaluation of multiple fine-tuned models against the same labeled test set.

**Example:**

```bash
python src/test_bert.py \
  --json_path outputs/resultados.json \
  --model_dirs models/ModernBERT-base_oversample models/Llama2_undersample \
  --label_keys coherencia desinformacion representacion_latinoamericana nivel_educacional originalidad score_final \
  --threshold 0.5
```

Outputs include macro F1 score and class-wise precision, recall, and F1 per model.

### Output Files

- Fine-tuned model and tokenizer (in `output_dir/...`)
- Evaluation metrics (validation + test) in `test_metrics.json`
- Truncated input texts in `data/textos_500k/`

## Features

- Multi-label classification with score thresholding
- LLM evaluation with JSON parsing and defensive validation
- Language-aware preprocessing
- Class balancing (under/over-sampling)
- Distributed GPU support via vLLM (e.g. Llama-3, Mistral)

## License

This project is released under the MIT License.
