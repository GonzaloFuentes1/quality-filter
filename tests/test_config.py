"""Lightweight checks for src/config.py (no GPU or ML dependencies)."""

import ast
import importlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
CONFIG_PATH = SRC / "config.py"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXPECTED_NAMES = [
    # general paths
    "DATA_DIR",
    "MODEL_PATH",
    # coordinator
    "QUALITY_MODEL",
    "TOKENIZE_SCRIPT",
    "PREDICT_SCRIPT",
    "PIPE_ROOT",
    "COORDINATOR_MAX_LEN",
    "COORDINATOR_BATCH_SIZE",
    "COORDINATOR_TOKENIZE_NUM_WORKERS",
    "COORDINATOR_DATA_LOADER_NUM_WORKERS",
    "COORDINATOR_THRESHOLD",
    "COORDINATOR_CUDA_VISIBLE_DEVICES",
    "COORDINATOR_OUTPUT_DIR",
    "COORDINATOR_INPUT_PATH",
    # training
    "BATCH_SIZE",
    "MAX_LENGTH",
    "NUM_EPOCHS",
    "TRAIN_LOG_DIR",
    "TRAINING_CUDA_VISIBLE_DEVICES",
    "TRAINING_OUTPUT_DIR",
    "TRAINING_INPUT_PATH",
    # inference
    "INFER_BATCH_SIZE",
    "INFER_NUM_WORKERS",
    "INFERENCE_CUDA_VISIBLE_DEVICES",
    "INFERENCE_OUTPUT_DIR",
    "INFERENCE_INPUT_PATH",
    # clean data
    "CLEAN_DATA_THRESHOLD",
    "CLEAN_DATA_CUDA_VISIBLE_DEVICES",
    "CLEAN_DATA_MODEL_NAME",
    "CLEAN_DATA_OUTPUT_DIR",
    "CLEAN_DATA_INPUT_PATH",
    # label quality
    "LABEL_QUALITY_OUTPUT_DIR",
    "LABEL_QUALITY_CUDA_VISIBLE_DEVICES",
    "LABEL_QUALITY_INPUT_PATH",
    # GPU and utils
    "CUDA_VISIBLE_DEVICES",
    "N_SAMPLES",
    "SPLIT_LANG_INPUT_PATH",
]


def _load_config():
    import config

    return importlib.reload(config)


def _imports_config(tree):
    return any(
        isinstance(node, ast.Import)
        and any(alias.name == "config" for alias in node.names)
        for node in ast.walk(tree)
    )


def _names_used_from_config(tree):
    """Names accessed as config.NAME or getattr(config, 'NAME', ...)."""
    names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "config"
        ):
            names.add(node.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "config"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            names.add(node.args[1].value)
    return names


def test_expected_names_exist():
    config = _load_config()
    missing = [name for name in EXPECTED_NAMES if not hasattr(config, name)]
    assert not missing, f"config.py is missing: {missing}"


def test_names_defined_once():
    """Guards against re-introducing a duplicated copy of the settings."""
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
    seen = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    seen[target.id] = seen.get(target.id, 0) + 1
    duplicated = sorted(name for name, count in seen.items() if count > 1)
    assert not duplicated, f"names assigned more than once: {duplicated}"


def test_every_name_used_by_scripts_exists():
    config = _load_config()
    problems = []
    for path in sorted(SRC.rglob("*.py")):
        if path == CONFIG_PATH:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not _imports_config(tree):
            continue
        for name in sorted(_names_used_from_config(tree)):
            if not hasattr(config, name):
                problems.append(f"{path.relative_to(SRC)}: config.{name}")
    assert not problems, f"names missing from config.py: {problems}"


def test_env_override_and_fallback(monkeypatch):
    monkeypatch.setenv("BATCH_SIZE", "16")
    monkeypatch.setenv("COORDINATOR_THRESHOLD", "not-a-number")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    config = _load_config()
    assert config.BATCH_SIZE == 16
    assert config.COORDINATOR_THRESHOLD == 0.5
    assert config.CUDA_VISIBLE_DEVICES == "1"


@pytest.fixture(autouse=True)
def _restore_config_module():
    yield
    sys.modules.pop("config", None)
