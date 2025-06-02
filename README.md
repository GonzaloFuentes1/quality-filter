# quality-filter

Repository for processing, cleaning, evaluating, and training multilingual text quality classification models, with a special focus on linguistic, educational, and cultural criteria.

## Repository Structure

```
src/
  clean_data/
    detect_outliers.py
    undersampling.py
  inference/
    label_quality.py
    promt.txt
  training/
    test_bert.py
    train_bert.py
  utils/
    bootstrap.py
    split_languajes.py
```

## Main Scripts Description

### `src/training/train_bert.py`

Trains a BERT (or compatible) model for multilabel text classification, using custom metrics and balancing methods (`undersample`/`oversample`). Allows selecting which labels to train and saves validation and test metrics.

### `src/training/test_bert.py`

Evaluates one or more trained models on a test set, calculating F1, precision, and recall metrics per class, and saves the results in a JSON file.

### `src/utils/bootstrap.py`

Filters a text dataset using a trained model, selecting those that exceed a probability threshold for at least one label. Supports parallel processing with multiple GPUs and generates probability histograms.

### `src/inference/label_quality.py`

Evaluates texts using an LLM (via vLLM) and a detailed FSM prompt (see `promt.txt`). Extracts and validates the JSON structure of the response, ensuring compliance with formatting and scoring logic rules.

### `src/utils/split_languajes.py`

Detects the language of each text in a JSON file and splits the texts into separate files by language (`es`, `en`, `pt`). Uses parallel processing for faster detection.

### `src/clean_data/detect_outliers.py`

Detects and removes outliers in multilabel datasets using cross-validation and binary cross-entropy (BCE) loss as the outlier score. Saves a new clean JSON file.

### `src/clean_data/undersampling.py`

Balances multilingual datasets by the originality label, ensuring an equal number of positive and negative examples per language.

### `src/inference/promt.txt`

Detailed prompt for FSM text quality evaluation, used by `label_quality.py`. Defines criteria, output format, and justification rules.

## Dependency Installation

It is recommended to create a virtual environment and then install the required dependencies, for example:

```sh
pip install -r requirements.txt
```

Main dependencies:

- `transformers`
- `datasets`
- `scikit-learn`
- `torch`
- `matplotlib`
- `langdetect`
- `tqdm`
- `vllm` (for LLM evaluation)
- `pandas`
- `numpy`

## Usage Example

### Training

```sh
python src/training/train_bert.py \
  --json_path datos.json \
  --output_dir modelos/ \
  --logging_dir logs/ \
  --model_name xlm-roberta-base \
  --label_ids 0 1 2 3 4 5 \
  --threshold 0.5 \
  --balance_method undersample
```

### Evaluation

```sh
python src/training/test_bert.py \
  --json_path test.json \
  --model_dirs modelos/xlm-roberta-base_undersample \
  --threshold 0.5 \
  --label_keys coherencia desinformacion representacion_latinoamericana nivel_educacional originalidad score_final
```

### Bootstrap Filtering

```sh
python src/utils/bootstrap.py \
  --input_path textos.json \
  --model_path modelos/xlm-roberta-base_undersample \
  --label_ids 0 1 2 3 4 5 \
  --threshold 0.3 \
  --output_path filtrados/
```

### LLM Evaluation

```sh
python src/inference/label_quality.py \
  --prompt_path src/inference/promt.txt \
  --dataset_path datos_filtrados/ \
  --model_path llama3-70b-instruct \
  --output_path resultados_llm.json
```

## Pre-commit

The repository includes configuration for `black`, `isort`, `flake8`, and other code quality hooks. Install pre-commit and activate it with:

```sh
pre-commit install
```

## License

MIT License.

---

For questions or improvements, contact the author or open an issue.
