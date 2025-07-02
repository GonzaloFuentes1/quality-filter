# Quality Filter for Text Data

A comprehensive pipeline for training and evaluating text quality models using transformer-based architectures. This project provides tools for data preprocessing, model training, inference, and evaluation of text quality metrics.

## Project Structure

```
quality-filter/
├── src/
│   ├── training/           # Model training scripts
│   │   ├── train_bert.py   # Main training script for BERT-like models
│   │   └── test_bert.py    # Model evaluation script
│   ├── inference/          # Inference and prediction scripts
│   │   ├── coordinator.py  # Pipeline coordinator for batch processing
│   │   ├── tokenize_dataset.py    # Dataset tokenization utility
│   │   ├── predict_and_save.py    # Model inference and result saving
│   │   ├── label_quality.py       # LLM-based quality labeling
│   │   └── promt.txt              # Quality evaluation prompt template
│   ├── utils/              # Utility scripts
│   │   ├── split_languajes.py     # Language detection and splitting
│   │   └── bootstrap.py           # Bootstrap filtering with multi-GPU
│   └── clean_data/         # Data cleaning and preprocessing
│       ├── undersampling.py       # Data balancing utilities
│       └── detect_outliers.py     # Outlier detection and removal
└── README.md
```

## Setup and Installation

### Requirements

- Python 3.8+
- PyTorch with CUDA support
- Transformers library
- Accelerate library
- Additional dependencies: datasets, sklearn, pandas, numpy, tqdm

### Environment Configuration

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Configure Accelerate for multi-GPU inference:

```bash
accelerate config
```

Follow the prompts to set up your hardware configuration. For multi-GPU setups, choose:

- Multi-GPU training: Yes
- How many processes: Number of GPUs available
- Use DeepSpeed: No (unless specifically needed)
- Use FullyShardedDataParallel: No
- Use CPU only: No

3. Verify configuration:

```bash
accelerate env
```

**Note**: For LLM-based quality labeling, you may need to install additional dependencies like `vllm`. Uncomment the relevant lines in `requirements.txt` if needed.

## Usage

### 1. Data Preparation

#### Language Detection and Splitting

```bash
python src/utils/split_languajes.py
```

Configure input paths in the script before running.

#### LLM-based Quality Labeling

Generate quality labels using large language models (this step creates the training labels):

```bash
python src/inference/label_quality.py \
    --prompt_path src/inference/promt.txt \
    --dataset_path data/input_dataset \
    --model_path meta-llama/Meta-Llama-3.1-70B-Instruct \
    --output_path results/llm_labels.json \
    --tensor_parallel_size 8 \
    --batch_size 50
```

#### Data Cleaning and Balancing

```bash
python src/clean_data/undersampling.py
```

#### Outlier Detection

```bash
python src/clean_data/detect_outliers.py \
    --json_path data/input.json \
    --output_dir results/clean \
    --label_keys coherencia desinformacion \
    --outlier_fraction 0.05
```

### 2. Model Training

Train a multilabel classification model:

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

### 3. Model Evaluation

Evaluate trained models:

```bash
python src/training/test_bert.py \
    --json_path data/test_data.json \
    --model_dirs models/quality_model \
    --threshold 0.5 \
    --label_keys coherencia desinformacion representacion_latinoamericana nivel_educacional originalidad
```

### 4. Inference Pipeline

#### Dataset Tokenization

```bash
python src/inference/tokenize_dataset.py \
    --dataset_path data/raw_dataset \
    --tokenizer_path xlm-roberta-base \
    --output_path data/tokenized_dataset \
    --max_length 512 \
    --num_workers 4
```

#### Model Inference with Accelerate

```bash
accelerate launch src/inference/predict_and_save.py \
    --task quality \
    --tokenized_dataset_path data/tokenized_dataset \
    --original_dataset_path data/raw_dataset \
    --model_dir models/quality_model \
    --output_path results/predictions \
    --threshold 0.5 \
    --batch_size 64 \
    --num_workers 4
```

#### Batch Processing Pipeline

Configure paths in `src/inference/coordinator.py` and run:

```bash
python src/inference/coordinator.py
```

## Configuration

### Model Labels

The project supports the following quality metrics:

- `coherence`: Text coherence and readability
- `desinformation`: Freedom from false information
- `latam_representation`: Latin American context representation
- `education_level`: Educational value
- `originality`: Content originality

### Hardware Requirements

- Multi-GPU setup recommended for inference
- Minimum 16GB VRAM for model training
- SSD storage recommended for large datasets

## Accelerate Configuration Notes

For optimal performance with Accelerate:

1. **Multi-GPU Setup**: Configure Accelerate to use all available GPUs
2. **Memory Management**: Use `keep_in_memory=True` for small datasets, `False` for large ones
3. **Batch Size**: Adjust based on GPU memory (start with 32-64 per GPU)
4. **Port Configuration**: Use different ports for multiple concurrent jobs

Example accelerate config for 4 GPUs:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: "no"
gpu_ids: "0,1,2,3"
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## License

This project is licensed under the MIT License.
