"""
Tokeniza un dataset de Hugging Face y lo guarda en disco
sin alterar las columnas originales (por ejemplo 'text' o etiquetas).
"""

import argparse
import tempfile
import time
from functools import partial
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

# Configurar tmpdir para evitar problemas de permisos en entornos compartidos
tempfile.tempdir = ""


def tokenize_fn(batch, tokenizer, max_length):
    return tokenizer(
        batch["texto"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                        help="Ruta del dataset (load_from_disk)")
    parser.add_argument("--tokenizer_path", required=True,
                        help="Nombre o carpeta del tokenizer")
    parser.add_argument("--output_path", required=True,
                        help="Carpeta destino del dataset tokenizado")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    t0 = time.time()

    print(f"Cargando dataset desde: {args.dataset_path}")
    ds = load_from_disk(args.dataset_path)

    print(f"Cargando tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        use_fast=True)
    if not tokenizer.is_fast:
        print("Advertencia: el tokenizer no es rápido (use_fast=False)")

    print(f"Tokenizando con {args.num_workers} procesos...")
    ds_tok = ds.map(
        partial(tokenize_fn, tokenizer=tokenizer, max_length=args.max_length),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_workers,
        desc="Tokenizando",
        keep_in_memory=True,
    )

    out_path = Path(args.output_path)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Guardando en: {out_path}")
    ds_tok.save_to_disk(str(out_path))

    print(f"Listo. Tiempo total: {time.time() - t0:.1f} segundos.")


if __name__ == "__main__":
    main()
