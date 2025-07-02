"""
Pipeline con Accelerate — filtro_idioma ➜ calidad ➜ toxicidad  (encadenado)
"""
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import List

# ---------- CONFIG ----------
QUALITY_MODEL = Path(
    "")
TOKENIZE_SCRIPT = Path(
    "")
PREDICT_SCRIPT = Path(
    "")

DATA_DIR = Path("")
PIPE_ROOT = Path("")
CALIDAD_DIR = PIPE_ROOT / "calidad"
TOKENS_DIR = PIPE_ROOT / "tokens"

MAX_LEN = 512
BATCH_SIZE = 2048
TOKENIZE_NUM_WORKERS = 120
DATA_LOADER_NUM_WORKERS = 120
THRESHOLD = 0.5
# ---------------------------------


# ---------- helpers ----------
def run(cmd: List[str], *, env: dict | None = None):
    """Ejecuta cmd en un nuevo grupo de procesos y propaga SIGINT/SIGTERM."""
    print("Ejecutando:", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)

    def _forward(sig, _frame):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)
    ret = proc.wait()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if ret:
        raise RuntimeError(f"Comando {' '.join(cmd)} terminó con código {ret}")


def tokenize_if_needed(raw_ds: Path, tok_ds: Path, tokenizer_model: Path):
    if tok_ds.exists():
        return
    tok_ds.parent.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, str(TOKENIZE_SCRIPT),
        "--dataset_path", str(raw_ds),
        "--tokenizer_path", str(tokenizer_model),
        "--output_path", str(tok_ds),
        "--max_length", str(MAX_LEN),
        "--num_workers", str(TOKENIZE_NUM_WORKERS),
    ])


def predict_stage(
    raw_ds: Path,          # dataset sin tokenizar (el que recibirá las nuevas columnas)
    tok_ds: Path,          # tokenización existente
    out_dir: Path,         # carpeta donde se guarda dataset_con_predicciones
    model_dir: Path,
    task: str,
):
    pred_ds = out_dir
    if pred_ds.exists():
        return pred_ds   # ya calculado
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    run([
        "accelerate", "launch", "--main_process_port", "1235",
        str(PREDICT_SCRIPT),
        "--task", task,
        "--tokenized_dataset_path", str(tok_ds),
        "--original_dataset_path", str(raw_ds),
        "--model_dir", str(model_dir),
        "--output_path", str(out_dir),
        "--threshold", str(THRESHOLD),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", str(DATA_LOADER_NUM_WORKERS),
    ], env=env)
    return pred_ds


# ---------------------------------

def main():
    subdatasets = [p.parent for p in DATA_DIR.rglob("dataset_info.json")]
    print(f"Se procesarán {len(subdatasets)} sub-datasets.\n")

    for raw_ds in subdatasets:
        rel = raw_ds.relative_to(DATA_DIR)
        print(f"\n=== Procesando: {rel} ===")

        tok_ds = TOKENS_DIR / rel
        qual_out = CALIDAD_DIR / rel

        # 0) tokenización (si falta)
        tokenize_if_needed(raw_ds, tok_ds, QUALITY_MODEL)

        # 1) CALIDAD  → agrega columnas de calidad al dataset original
        qual_pred_ds = predict_stage(
            raw_ds=raw_ds,
            tok_ds=tok_ds,
            out_dir=qual_out,
            model_dir=QUALITY_MODEL,
            task="quality",
        )

        print(f"Terminado: {qual_pred_ds}")

    print("\nPipeline completo.")


if __name__ == "__main__":
    main()
