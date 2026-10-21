# Quality Filter

LLM-labelled multilabel text quality classifier and batch filtering pipeline for Spanish/LatAm pretraining data.

## Overview

The repository contains the scripts used to build a text quality filter in three stages:

1. **Labelling.** A large language model served with vLLM scores each text against a rubric (`src/inference/prompt.txt`) and returns continuous scores in the range 0.000 to 1.000.
2. **Training.** The scored texts are used to fine-tune a transformer encoder (default `bert-base-uncased`, any Hugging Face sequence-classification model can be passed) as a multilabel classifier.
3. **Filtering.** The trained classifier is applied to large datasets, either as a batch pipeline run with Hugging Face Accelerate (`coordinator.py` and `predict_and_save.py`) or as a multi-GPU bootstrap filter that uses `torch.multiprocessing` (`bootstrap.py`).

No evaluation results are reported in this repository.

## Project structure

```
quality-filter/
|-- src/
|   |-- config.py               # Centralised paths and parameters (env-var driven)
|   |-- training/
|   |   |-- train_bert.py       # Fine-tune a multilabel classifier
|   |   `-- test_bert.py        # Evaluate trained models on a labelled JSON file
|   |-- inference/
|   |   |-- label_quality.py    # LLM-based labelling (vLLM)
|   |   |-- prompt.txt          # Labelling prompt (Spanish)
|   |   |-- tokenize_dataset.py # Tokenise a Hugging Face dataset
|   |   |-- predict_and_save.py # Classifier inference with Accelerate
|   |   `-- coordinator.py      # Tokenise + predict over many sub-datasets
|   |-- utils/
|   |   |-- split_languajes.py  # Language detection and splitting (es/en/pt)
|   |   `-- bootstrap.py        # Multi-GPU filtering by label probability
|   `-- clean_data/
|       |-- undersampling.py    # Balance positives and negatives
|       `-- detect_outliers.py  # Cross-validated outlier removal
|-- tests/test_config.py        # Checks on src/config.py (no GPU needed)
|-- .github/workflows/ci.yml    # compileall + pytest
`-- requirements.txt
```

## Pipeline

```
raw texts (JSON list of objects with a "texto" field)
   |
   v
src/utils/split_languajes.py ........ es / en / pt JSON files (langdetect)
   |
   v
src/inference/label_quality.py ...... LLM scores per text, JSON keyed by entry id
   |                                  (prompt: src/inference/prompt.txt)
   |   (conversion to a JSON list of {"texto": ..., <label keys>: ...}
   |    is not provided by this repository)
   v
src/clean_data/undersampling.py ..... balanced JSON (optional)
src/clean_data/detect_outliers.py ... cleaned JSON (optional)
   |
   v
src/training/train_bert.py .......... fine-tuned multilabel classifier
src/training/test_bert.py ........... per-label precision / recall / F1
   |
   v
src/inference/tokenize_dataset.py
   |
   v
src/inference/predict_and_save.py ... dataset + one probability column per label
   (src/inference/coordinator.py runs the two steps above for every sub-dataset
    found under DATA_DIR)

src/utils/bootstrap.py .............. keeps texts whose probability for any
                                      selected label is >= threshold
```

## Labels

The labelling prompt, the training script, the evaluation script and the cleaning scripts use these Spanish keys:

| Key | Index in `train_bert.py --label_ids` | Meaning (from `prompt.txt`) |
|---|---|---|
| `coherencia` | 0 | Well written, coherent and understandable |
| `desinformacion` | 1 | Free of false claims, biased ideology or conspiratorial language |
| `representacion_latinoamericana` | 2 | Reflects Latin American contexts, actors or issues |
| `nivel_educacional` | 3 | Clear educational value |
| `originalidad` | 4 | Distinctive voice or content |
| `score_final` | 5 | Overall decision derived from the other scores |

The LLM output additionally contains `etapa_1_valida` and `justificacion`. `label_quality.py` checks each output in two steps:

- `extract_json_block` rejects the output (the entry is stored as an `error`) if a required key is missing, a score is outside 0.000 to 1.000, or `justificacion` breaks its rule: it must be the string `sin justificacion` when `score_final` is not `0.000`, and a real explanation when it is `0.000`.
- `validar_score_final` recomputes the derived scores and, if they differ from the LLM output, keeps the entry but adds an `errores_validacion` list:
  - `etapa_1_valida = 1.000` if `coherencia > 0.5` and `desinformacion > 0.5`, otherwise `0.000`.
  - `score_final = 1.000` if `etapa_1_valida == 1.000` and at least one of `representacion_latinoamericana`, `nivel_educacional`, `originalidad` is `> 0.5`, otherwise `0.000`.

Note that `src/inference/predict_and_save.py` names its five output columns differently: `coherence`, `desinformation`, `latam_representation`, `education_level`, `originality` (task `quality`). They are assigned by position, so they correspond to a model trained with `--label_ids 0 1 2 3 4`.

## Setup

Python 3.10 or newer is required (`coordinator.py` uses `X | None` type hints).

```bash
pip install -r requirements.txt
```

`requirements.txt` does not include `vllm`, which `label_quality.py` needs; install it separately. For multi-GPU inference configure Accelerate once:

```bash
accelerate config
accelerate env     # verify
```

All scripts do `import config`, which lives in `src/`. Run them with `src` on the Python path:

```bash
export PYTHONPATH=src
python src/training/train_bert.py --help
```

The commands below assume `PYTHONPATH=src` is set.

## Usage

Paths in the examples are placeholders (`checkpoint-STEP` stands for a checkpoint directory written by `train_bert.py`). Input JSON files are lists of objects with a `texto` field and, where labels are needed, one numeric field per label key.

### `src/utils/split_languajes.py`

Detects the language of every entry with `langdetect` (100 worker processes) and writes `test_textos_es_<N_SAMPLES>.json`, `test_textos_en_<N_SAMPLES>.json` and `test_textos_pt_<N_SAMPLES>.json` to the current directory. It takes no arguments; the input is `SPLIT_LANG_INPUT_PATH` and the suffix is `N_SAMPLES` (see [Configuration](#configuration)).

```bash
SPLIT_LANG_INPUT_PATH=data/raw.json N_SAMPLES=1000 python src/utils/split_languajes.py
```

### `src/inference/label_quality.py`

Labels a Hugging Face dataset (`load_from_disk`) with an LLM through vLLM. The text column is inserted into the prompt in place of `[PEGUE AQUÍ EL TEXTO]` after truncation to 5000 characters. The result is a JSON file keyed by entry id; each entry holds `evaluacion` (the keys listed in [Labels](#labels)) or an `error` with the raw LLM output. The truncated text of each entry is also written as `<id>.txt` under `LABEL_QUALITY_OUTPUT_DIR`.

```bash
python src/inference/label_quality.py \
    --prompt_path src/inference/prompt.txt \
    --dataset_path data/input_dataset \
    --model_path meta-llama/Meta-Llama-3.1-70B-Instruct \
    --output_path results/llm_labels.json \
    --tensor_parallel_size 8 \
    --batch_size 50
```

Other options: `--text_column` (default `texto`), `--download_dir`.

### `src/clean_data/undersampling.py`

Loads the file at `SPLIT_LANG_INPUT_PATH`, finds the label with the fewest values `>= CLEAN_DATA_THRESHOLD` among `coherencia`, `desinformacion`, `representacion_latinoamericana`, `nivel_educacional`, `originalidad`, `score_final`, and writes a balanced file with equal numbers of positive and negative examples to `DATA_DIR/balanced_texts_<N_SAMPLES>.json`. No arguments.

```bash
python src/clean_data/undersampling.py
```

### `src/clean_data/detect_outliers.py`

Trains a classifier with 5-fold cross-validation (2 epochs per fold), scores every text by its mean binary cross-entropy on the held-out fold, drops the highest-scoring fraction and writes `datos_limpios.json` to the output directory.

```bash
python src/clean_data/detect_outliers.py \
    --json_path data/input.json \
    --output_dir results/clean \
    --label_keys coherencia desinformacion \
    --outlier_fraction 0.05
```

Options: `--model_name` (default `CLEAN_DATA_MODEL_NAME`), `--output_dir` (default `CLEAN_DATA_OUTPUT_DIR`), `--outlier_fraction` (default 0.05).

### `src/training/train_bert.py`

Fine-tunes a multilabel classifier. The input JSON is split 80/10/10 into train, validation and test with seed 42; `--balance_method` is applied to the training split only. Labels are selected by index (see [Labels](#labels)). Results are written to `<output_dir>/<model_name>_<balance_method|no-balance>/` (with `/` in the model name replaced by `-`), including `test_metrics.json`; Trainer checkpoints (model and tokenizer) are saved there each epoch as `checkpoint-<step>/`. The script does not call `trainer.save_model()`, so the run directory itself holds no final model: pass one of the `checkpoint-<step>/` directories to `test_bert.py`, `predict_and_save.py`, `coordinator.py` and `bootstrap.py`.

```bash
python src/training/train_bert.py \
    --json_path data/training_data.json \
    --output_dir models/quality_model \
    --logging_dir logs/training \
    --model_name xlm-roberta-base \
    --label_ids 0 1 2 3 4 \
    --threshold 0.5 \
    --balance_method undersample
```

`--label_ids` is required. Without `--model_name` the default is `bert-base-uncased`. Learning rate (2e-5), number of epochs (5), weight decay and `bf16=True` are fixed in the script; `MAX_LENGTH` and `BATCH_SIZE` come from `config.py`.

### `src/training/test_bert.py`

Evaluates one or more model directories on a labelled JSON file (inputs truncated to 512 tokens) and writes `metrics.json` (macro F1 and per-label precision, recall and F1) into each model directory. `--label_keys` must be given in the same order as the labels the model was trained with.

```bash
python src/training/test_bert.py \
    --json_path data/test_data.json \
    --model_dirs models/quality_model/xlm-roberta-base_undersample/checkpoint-STEP \
    --threshold 0.5 \
    --label_keys coherencia desinformacion representacion_latinoamericana nivel_educacional originalidad
```

### `src/inference/tokenize_dataset.py`

Tokenises the `texto` column of a Hugging Face dataset (padding to `max_length`) and saves it with the original columns kept.

```bash
python src/inference/tokenize_dataset.py \
    --dataset_path data/raw_dataset \
    --tokenizer_path xlm-roberta-base \
    --output_path data/tokenized_dataset \
    --max_length 512 \
    --num_workers 4
```

### `src/inference/predict_and_save.py`

Runs the classifier on a tokenised dataset and saves the original dataset with five added float columns holding sigmoid probabilities (column names in [Labels](#labels)). `--threshold` is accepted but not applied: the output stores probabilities.

```bash
accelerate launch src/inference/predict_and_save.py \
    --task quality \
    --tokenized_dataset_path data/tokenized_dataset \
    --original_dataset_path data/raw_dataset \
    --model_dir models/quality_model/xlm-roberta-base_undersample/checkpoint-STEP \
    --output_path results/predictions \
    --threshold 0.5 \
    --batch_size 64 \
    --num_workers 4
```

### `src/inference/coordinator.py`

Finds every sub-dataset (directory containing `dataset_info.json`) under `DATA_DIR`, tokenises it into `PIPE_ROOT/tokens/<relative path>` if missing, and runs `predict_and_save.py` through `accelerate launch --main_process_port 1235`, writing to `PIPE_ROOT/calidad/<relative path>`. Existing outputs are skipped. `QUALITY_MODEL`, `TOKENIZE_SCRIPT` and `PREDICT_SCRIPT` have no default and must be set. Because the scan is recursive, keep `PIPE_ROOT` outside `DATA_DIR` so earlier outputs are not picked up as inputs. No arguments.

```bash
QUALITY_MODEL=models/quality_model/xlm-roberta-base_undersample/checkpoint-STEP \
TOKENIZE_SCRIPT=src/inference/tokenize_dataset.py \
PREDICT_SCRIPT=src/inference/predict_and_save.py \
DATA_DIR=data/datasets PIPE_ROOT=data/pipeline \
python src/inference/coordinator.py
```

### `src/utils/bootstrap.py`

Splits the dataset across the GPUs listed in `CUDA_VISIBLE_DEVICES` (read from the environment, not from `config.py`), scores each text, and keeps those whose probability is `>= --threshold` for at least one of the selected labels. It saves a dataset with a `texto` column to `--output_path` and histograms of the probabilities to `graficos_bootstrap/`. The input is a JSON list with `texto` fields or a Hugging Face dataset directory with a `text`, `texto` or `content` column.

```bash
CUDA_VISIBLE_DEVICES=0,1 python src/utils/bootstrap.py \
    --input_path data/input.json \
    --model_path models/quality_model/xlm-roberta-base_undersample/checkpoint-STEP \
    --label_ids 0 1 \
    --threshold 0.3 \
    --batch_size 16 \
    --output_path results/filtered
```

## Configuration

`src/config.py` holds the paths and parameters shared by the scripts. Every value is read from an environment variable of the same name; if the variable is unset, or cannot be cast to the expected type, the default is used. Example:

```bash
export DATA_DIR=/path/to/data
export CLEAN_DATA_CUDA_VISIBLE_DEVICES=0,1
```

Relative defaults are resolved from `src/`, so `DATA_DIR` defaults to `<repo>/data` and `MODEL_PATH` to `<repo>/model`. Command-line options take precedence over the config values where a script exposes both.

| Variable | Default | Type | Read by |
|---|---|---|---|
| `DATA_DIR` | `<repo>/data` | path | `coordinator.py`, `label_quality.py`, `train_bert.py`, `undersampling.py` |
| `MODEL_PATH` | `<repo>/model` | path | `train_bert.py` |
| `CUDA_VISIBLE_DEVICES` | `0` | str | `train_bert.py`, `label_quality.py` |
| `QUALITY_MODEL` | empty | path | `coordinator.py` |
| `TOKENIZE_SCRIPT` | empty | path | `coordinator.py` |
| `PREDICT_SCRIPT` | empty | path | `coordinator.py` |
| `PIPE_ROOT` | `DATA_DIR` | path | `coordinator.py` |
| `COORDINATOR_MAX_LEN` | `512` | int | `coordinator.py` |
| `COORDINATOR_BATCH_SIZE` | `2048` | int | `coordinator.py` |
| `COORDINATOR_TOKENIZE_NUM_WORKERS` | `120` | int | `coordinator.py` |
| `COORDINATOR_DATA_LOADER_NUM_WORKERS` | `120` | int | `coordinator.py` |
| `COORDINATOR_THRESHOLD` | `0.5` | float | `coordinator.py` |
| `BATCH_SIZE` | `32` | int | `train_bert.py`, `detect_outliers.py` |
| `MAX_LENGTH` | `128` | int | `train_bert.py`, `detect_outliers.py` |
| `INFERENCE_CUDA_VISIBLE_DEVICES` | `0` | str | `test_bert.py` |
| `CLEAN_DATA_THRESHOLD` | `0.5` | float | `undersampling.py` |
| `CLEAN_DATA_CUDA_VISIBLE_DEVICES` | `0` | str | `detect_outliers.py` |
| `CLEAN_DATA_MODEL_NAME` | `xlm-roberta-base` | str | `detect_outliers.py` |
| `CLEAN_DATA_OUTPUT_DIR` | `DATA_DIR/clean_data_outputs` | path | `detect_outliers.py` |
| `LABEL_QUALITY_OUTPUT_DIR` | `DATA_DIR/label_quality_outputs` | path | `label_quality.py` |
| `SPLIT_LANG_INPUT_PATH` | `DATA_DIR/split_lang_input.json` | path | `split_languajes.py`, `undersampling.py` |
| `N_SAMPLES` | empty | str | `split_languajes.py`, `undersampling.py` |

The following variables are defined in `config.py` but are not read by any script yet: `COORDINATOR_CUDA_VISIBLE_DEVICES`, `COORDINATOR_OUTPUT_DIR`, `COORDINATOR_INPUT_PATH`, `NUM_EPOCHS`, `TRAIN_LOG_DIR`, `TRAINING_CUDA_VISIBLE_DEVICES`, `TRAINING_OUTPUT_DIR`, `TRAINING_INPUT_PATH`, `INFER_BATCH_SIZE`, `INFER_NUM_WORKERS`, `INFERENCE_OUTPUT_DIR`, `INFERENCE_INPUT_PATH`, `CLEAN_DATA_INPUT_PATH`, `LABEL_QUALITY_CUDA_VISIBLE_DEVICES`, `LABEL_QUALITY_INPUT_PATH`.

## Development

```bash
pip install pytest
python -m compileall src
python -m pytest tests
```

CI (`.github/workflows/ci.yml`) runs the same two commands. The tests check that `config.py` defines each setting once and that every `config.<NAME>` used by the scripts exists.

## License

This project is licensed under the MIT License.
