
import os


def get_env(var, default, cast=None):
    """
    Helper to get environment variable with type casting and default.
    Usage:
        get_env('VAR', 123, int)
        get_env('VAR', 'default')
    """
    val = os.getenv(var, default)
    if cast:
        try:
            return cast(val)
        except Exception:
            return cast(default)
    return val


"""
config.py
Centralized configuration file for paths and global parameters.
All scripts should import from this file to avoid hardcoded values.

Sections:
    - General paths
    - Coordinator pipeline configuration
    - Training configuration
    - Inference configuration
    - Clean data configuration
    - Label quality configuration
    - Utils configuration
"""

# =====================
# General paths
# =====================
DATA_DIR = get_env(
    'DATA_DIR', os.path.join(os.path.dirname(__file__), '../data')
)
MODEL_PATH = get_env(
    'MODEL_PATH', os.path.join(os.path.dirname(__file__), '../model')
)

# =====================
# Coordinator pipeline configuration
# =====================
QUALITY_MODEL = get_env('QUALITY_MODEL', '')
TOKENIZE_SCRIPT = get_env('TOKENIZE_SCRIPT', '')
PREDICT_SCRIPT = get_env('PREDICT_SCRIPT', '')
PIPE_ROOT = get_env('PIPE_ROOT', DATA_DIR)
COORDINATOR_MAX_LEN = get_env('COORDINATOR_MAX_LEN', 512, int)
COORDINATOR_BATCH_SIZE = get_env('COORDINATOR_BATCH_SIZE', 2048, int)
COORDINATOR_TOKENIZE_NUM_WORKERS = get_env(
    'COORDINATOR_TOKENIZE_NUM_WORKERS', 120, int
)
COORDINATOR_DATA_LOADER_NUM_WORKERS = get_env(
    'COORDINATOR_DATA_LOADER_NUM_WORKERS', 120, int
)
COORDINATOR_THRESHOLD = get_env('COORDINATOR_THRESHOLD', 0.5, float)
COORDINATOR_CUDA_VISIBLE_DEVICES = get_env(
    'COORDINATOR_CUDA_VISIBLE_DEVICES', '0'
)
COORDINATOR_OUTPUT_DIR = get_env(
    'COORDINATOR_OUTPUT_DIR', os.path.join(DATA_DIR, 'coordinator_outputs')
)
COORDINATOR_INPUT_PATH = get_env(
    'COORDINATOR_INPUT_PATH', os.path.join(DATA_DIR, 'coordinator_input.json')
)

# =====================
# Training configuration
# =====================
BATCH_SIZE = get_env('BATCH_SIZE', 32, int)
MAX_LENGTH = get_env('MAX_LENGTH', 128, int)
NUM_EPOCHS = get_env('NUM_EPOCHS', 5, int)
TRAIN_LOG_DIR = get_env(
    'TRAIN_LOG_DIR', os.path.join(MODEL_PATH, 'logs')
)
TRAINING_CUDA_VISIBLE_DEVICES = get_env(
    'TRAINING_CUDA_VISIBLE_DEVICES', '0'
)
TRAINING_OUTPUT_DIR = get_env(
    'TRAINING_OUTPUT_DIR', os.path.join(DATA_DIR, 'training_outputs')
)
TRAINING_INPUT_PATH = get_env(
    'TRAINING_INPUT_PATH', os.path.join(DATA_DIR, 'train_input.json')
)

# =====================
# Inference configuration
# =====================
INFER_BATCH_SIZE = get_env('INFER_BATCH_SIZE', 64, int)
INFER_NUM_WORKERS = get_env('INFER_NUM_WORKERS', 4, int)
INFERENCE_CUDA_VISIBLE_DEVICES = get_env(
    'INFERENCE_CUDA_VISIBLE_DEVICES', '0'
)
INFERENCE_OUTPUT_DIR = get_env(
    'INFERENCE_OUTPUT_DIR', os.path.join(DATA_DIR, 'inference_outputs')
)
INFERENCE_INPUT_PATH = get_env(
    'INFERENCE_INPUT_PATH', os.path.join(DATA_DIR, 'inference_input.json')
)

# =====================
# Clean data configuration
# =====================
CLEAN_DATA_THRESHOLD = get_env('CLEAN_DATA_THRESHOLD', 0.5, float)
CLEAN_DATA_CUDA_VISIBLE_DEVICES = get_env(
    'CLEAN_DATA_CUDA_VISIBLE_DEVICES', '0'
)
CLEAN_DATA_OUTPUT_DIR = get_env(
    'CLEAN_DATA_OUTPUT_DIR', os.path.join(DATA_DIR, 'clean_data_outputs')
)
CLEAN_DATA_INPUT_PATH = get_env(
    'CLEAN_DATA_INPUT_PATH', os.path.join(DATA_DIR, 'clean_data_input.json')
)

# =====================
# Label quality configuration
# =====================
LABEL_QUALITY_OUTPUT_DIR = get_env(
    'LABEL_QUALITY_OUTPUT_DIR',
    os.path.join(DATA_DIR, 'label_quality_outputs')
)
LABEL_QUALITY_CUDA_VISIBLE_DEVICES = get_env(
    'LABEL_QUALITY_CUDA_VISIBLE_DEVICES', '0'
)
LABEL_QUALITY_INPUT_PATH = get_env(
    'LABEL_QUALITY_INPUT_PATH', os.path.join(DATA_DIR, 'label_quality_input.json')
)


# =====================
# Add more global parameters below as needed
# =====================

"""
config.py
Centralized configuration file for paths and global parameters.
All scripts should import from this file to avoid hardcoded values.

Sections:
    - General paths
    - Training configuration
    - Inference configuration
    - Clean data configuration
    - Label quality configuration
    - GPU configuration
    - Utils configuration
"""

# =====================
# General paths
# =====================
DATA_DIR = os.getenv('DATA_DIR', os.path.join(os.path.dirname(__file__), '../data'))
MODEL_PATH = os.getenv('MODEL_PATH',
                       os.path.join(os.path.dirname(__file__), '../model'))

# =====================
# Training configuration
# =====================
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))  # Default batch size
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 128))  # Default max sequence length
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 5))  # Default number of epochs
TRAIN_LOG_DIR = os.getenv('TRAIN_LOG_DIR', os.path.join(MODEL_PATH, 'logs'))

# =====================
# Inference configuration
# =====================
INFER_BATCH_SIZE = int(os.getenv('INFER_BATCH_SIZE', 64))
INFER_NUM_WORKERS = int(os.getenv('INFER_NUM_WORKERS', 4))

# =====================
# Clean data configuration
# =====================
CLEAN_DATA_THRESHOLD = float(os.getenv('CLEAN_DATA_THRESHOLD', 0.5))

# =====================
# Label quality configuration
# =====================
LABEL_QUALITY_OUTPUT_DIR = os.getenv('LABEL_QUALITY_OUTPUT_DIR',
                                     os.path.join(DATA_DIR, 'label_quality_outputs'))

# =====================
# GPU configuration
# =====================
CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0')

# =====================
# Add more global parameters below as needed
# =====================
