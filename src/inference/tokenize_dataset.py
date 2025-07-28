"""
Tokenizes a Hugging Face dataset and saves it to disk
without altering the original columns (e.g. 'text' or labels).
"""

import argparse
import tempfile
import time
from functools import partial
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

# Set tmpdir to avoid permission issues in shared environments
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
                        help="Path to the dataset (load_from_disk)")
    parser.add_argument("--tokenizer_path", required=True,
                        help="Tokenizer name or folder")
    parser.add_argument("--output_path", required=True,
                        help="Destination folder for the tokenized dataset")
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
        print("Warning: the tokenizer is not fast (use_fast=False)")

    print(f"Tokenizing with {args.num_workers} processes...")
    ds_tok = ds.map(
        partial(tokenize_fn, tokenizer=tokenizer, max_length=args.max_length),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_workers,
        desc="Tokenizing",
        keep_in_memory=True,
    )

    out_path = Path(args.output_path)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {out_path}")
    ds_tok.save_to_disk(str(out_path))

    print(f"Done. Total time: {time.time() - t0:.1f} seconds.")


if __name__ == "__main__":
    main()
