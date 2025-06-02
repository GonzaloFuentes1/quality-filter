import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


def load_json_data(json_path, label_keys):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item["texto"] for item in data]
    labels = [
        [float(item[key]) for key in label_keys]
        for item in data
    ]
    return pd.DataFrame({
        'text': texts,
        'labels': labels,
        'original': data
    })


def tokenize_dataset(df, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    dataset = Dataset.from_pandas(df)
    return dataset.map(tokenize_function, batched=True)


def compute_outlier_scores(preds, labels):
    probs = torch.sigmoid(torch.tensor(preds))
    labels = torch.tensor(labels)
    loss = F.binary_cross_entropy(
        probs, labels, reduction='none'
    ).mean(dim=1)
    return loss.numpy()


def detect_outliers(
    df, label_keys, model_name,
    outlier_fraction, output_path
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    outlier_scores = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\nFold {fold + 1}/5")
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_dataset = tokenize_dataset(train_df, tokenizer)
        val_dataset = tokenize_dataset(val_df, tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label_keys),
            problem_type="multi_label_classification"
        )

        args = TrainingArguments(
            output_dir=f"{output_path}/fold_{fold}",
            num_train_epochs=2,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=16,
            eval_strategy='no',
            save_strategy='no',
            disable_tqdm=True,
            logging_steps=50,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset
        )
        trainer.train()

        preds_logits = trainer.predict(val_dataset).predictions
        preds = torch.sigmoid(
            torch.tensor(preds_logits)
        ).numpy()

        val_labels = val_df["labels"].tolist()
        outlier_scores[val_idx] = compute_outlier_scores(
            preds, val_labels
        )

    df["outlier_score"] = outlier_scores
    threshold = np.quantile(outlier_scores, 1 - outlier_fraction)
    df_clean = df[df["outlier_score"] <= threshold].copy()

    output_file = os.path.join(output_path, "datos_limpios.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            df_clean["original"].tolist(),
            f,
            indent=2,
            ensure_ascii=False
        )

    removed = len(df) - len(df_clean)
    pct = outlier_fraction * 100
    print(f"\nSe eliminaron {removed} posibles outliers ({pct:.0f}%).")
    print(f"Dataset limpio guardado en {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Detección de outliers multilabel por BCE."
    )
    parser.add_argument(
        "--json_path", type=str, required=True,
        help="Ruta al archivo JSON original."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directorio de salida."
    )
    parser.add_argument(
        "--model_name", type=str, default="xlm-roberta-base",
        help="Modelo preentrenado."
    )
    parser.add_argument(
        "--outlier_fraction", type=float, default=0.05,
        help="Fracción a eliminar."
    )
    parser.add_argument(
        "--label_keys", nargs="+", required=True,
        help="Etiquetas multilabel a usar."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_json_data(args.json_path, args.label_keys)
    detect_outliers(
        df,
        args.label_keys,
        args.model_name,
        args.outlier_fraction,
        args.output_dir
    )


if __name__ == "__main__":
    main()
