import argparse
import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification

# ---------------- etiquetas ----------------
TASK_LABELS = {
    "quality": [
        "coherence",
        "desinformation",
        "latam_representation",
        "education_level",
        "originality",
    ]
}
# -------------------------------------------


class Predictor:
    def __init__(
        self,
        task: str,
        tokenized_dataset_path: str,
        model_dir: str,
        output_path: str,
        threshold: Union[float, List[float]] = 0.5,
        batch_size: int = 64,
        num_workers: int = 4,
        original_dataset_path: Optional[str] = None,
    ):
        if task not in TASK_LABELS:
            raise ValueError(f"Tarea desconocida {task!r}")

        self.labels = TASK_LABELS[task]
        self.ds_path = Path(tokenized_dataset_path)
        self.model_dir = Path(model_dir)
        self.out_path = Path(output_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.acc = Accelerator()
        self.out_path.mkdir(parents=True, exist_ok=True)

        # ruta al dataset “crudo”
        if original_dataset_path:
            self.orig_path = Path(original_dataset_path)
        else:
            parts = list(self.ds_path.parts)
            self.orig_path = (
                Path(*parts[: parts.index("tokens")],
                     *parts[parts.index("tokens") + 1 :])
                if "tokens" in parts
                else None
            )

    def _setup(self):
        self.ds_tok: Dataset = load_from_disk(str(self.ds_path))
        self.ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

        if self.orig_path and self.orig_path.exists():
            self.ds_orig: Dataset = load_from_disk(str(self.orig_path))
        else:
            self.ds_orig = self.ds_tok.remove_columns(
                [c for c in ("input_ids", "attention_mask", "token_type_ids")
                 if c in self.ds_tok.column_names]
            )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, trust_remote_code=True
        )
        dl = DataLoader(
            self.ds_tok,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        self.model, self.dl = self.acc.prepare(model, dl)

    # ---------- infer ----------
    @torch.no_grad()
    def _infer(self):
        self.model.eval()
        local_logits = []

        for batch in tqdm(
            self.dl,
            desc="Inferencia",
            disable=not self.acc.is_local_main_process,
        ):
            outs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            local_logits.append(outs.logits.detach())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logits_local = torch.cat(local_logits, dim=0)
        logits_global = self.acc.gather_for_metrics(logits_local)

        if not self.acc.is_main_process:
            return None

        logits_global = logits_global[: len(self.ds_tok)]
        probs = torch.sigmoid(logits_global).cpu().numpy().astype(np.float32)
        return probs

    # ---------- save ----------
    def _save(self, preds: np.ndarray):
        ds_plain = self.ds_orig
        # borra columnas con mismo nombre (si existen)
        for label in self.labels:
            if label in ds_plain.column_names:
                ds_plain = ds_plain.remove_columns(label)
        # añade columnas → valores python float, no tensores/arrays
        for i, label in enumerate(self.labels):
            column_vals = [float(x) for x in preds[:, i]]
            ds_plain = ds_plain.add_column(label, column_vals)

        out_ds = self.out_path
        ds_plain.save_to_disk(str(out_ds))
        print(f"Guardado: {out_ds}")

    # ---------- run ----------
    def run(self):
        self._setup()
        preds = self._infer()
        if preds is not None:
            self._save(preds)


# ---------------- CLI ----------------
def parse_threshold(txt: str):
    txt = txt.strip()
    if txt.startswith("["):
        return json.loads(txt)
    return float(txt)


def main():
    p = argparse.ArgumentParser(
        description="Inferencia multietiqueta + guardado de resultados")
    p.add_argument("--task", required=True, choices=list(TASK_LABELS))
    p.add_argument("--tokenized_dataset_path", required=True)
    p.add_argument("--original_dataset_path",
                   required=False, help="Ruta al dataset sin tokenizar")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--threshold", type=parse_threshold, default=0.5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    Predictor(
        task=args.task,
        tokenized_dataset_path=args.tokenized_dataset_path,
        original_dataset_path=args.original_dataset_path,
        model_dir=args.model_dir,
        output_path=args.output_path,
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    ).run()


if __name__ == "__main__":
    main()
