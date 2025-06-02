import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support
)

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


def load_json_data(json_path, label_keys):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['texto'] for item in data]
    labels = [
        [float(item[key]) for key in label_keys]
        for item in data
    ]
    return pd.DataFrame({'text': texts, 'labels': labels})


def tokenize_dataset(df, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    dataset = Dataset.from_pandas(df)
    return dataset.map(tokenize_function, batched=True)


def evaluate(model, dataset, threshold, device):
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset.remove_columns(['text']),
        batch_size=64
    )
    all_logits, all_labels = [], []

    for batch in dataloader:
        with torch.no_grad():
            inputs = {
                k: v.to(device)
                for k, v in batch.items() if k != 'labels'
            }
            logits = model(**inputs).logits
            all_logits.append(logits.cpu())
            all_labels.append(batch['labels'])

    preds = torch.sigmoid(torch.cat(all_logits)).numpy()
    labels = torch.cat(all_labels).numpy()

    pred_binary = (preds >= threshold).astype(int)
    labels_binary = (labels >= threshold).astype(int)

    f1 = f1_score(labels_binary, pred_binary, average='macro')
    precision, recall, f1_class, _ = precision_recall_fscore_support(
        labels_binary, pred_binary, average=None
    )

    return f1, precision.tolist(), recall.tolist(), f1_class.tolist()


def save_metrics(model_dir, label_keys, f1, precision, recall, f1_class):
    metrics = {
        "macro_f1": f1,
        "class_metrics": [
            {
                "label": label_keys[i],
                "f1": f1_class[i],
                "precision": precision[i],
                "recall": recall[i]
            }
            for i in range(len(label_keys))
        ]
    }
    output_path = os.path.join(model_dir, "metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--model_dirs', nargs='+', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--label_keys', nargs='+', required=True)
    args = parser.parse_args()

    df = load_json_data(args.json_path, args.label_keys)

    for model_dir in args.model_dirs:
        print(f"\nEvaluando modelo: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            trust_remote_code=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print(f" - Usando {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        model.to(device)

        tokenized_dataset = tokenize_dataset(df, tokenizer)
        tokenized_dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels']
        )

        f1, precision, recall, f1_class = evaluate(
            model, tokenized_dataset, args.threshold, device
        )

        print(f"Macro F1: {f1:.4f}")
        for i, label in enumerate(args.label_keys):
            print(
                f" - {label}: F1={f1_class[i]:.4f}, "
                f"Precision={precision[i]:.4f}, Recall={recall[i]:.4f}"
            )

        save_metrics(model_dir, args.label_keys, f1, precision, recall, f1_class)


if __name__ == "__main__":
    main()
