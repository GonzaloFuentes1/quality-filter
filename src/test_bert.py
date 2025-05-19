import argparse
import json

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_json_data(json_path, label_keys):
    """ 
    Carga el JSON con etiquetas.

    Tokeniza los textos.

    EvalÃºa todos los modelos indicados con F1 macro y F1 por clase.

    Todas las etiquetas son evaluadas especificadas en label_keys .
    """

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = [item['texto'] for item in data]
    labels = [[float(item[key]) for key in label_keys] for item in data]
    return pd.DataFrame({'text': texts, 'labels': labels})


def tokenize_dataset(df, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", 
                         truncation=True, max_length=512)

    dataset = Dataset.from_pandas(df)
    return dataset.map(tokenize_function, batched=True)


def evaluate(model, dataset, threshold):
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset.remove_columns(['text']), 
                                             batch_size=16)

    all_logits, all_labels = [], []
    for batch in dataloader:
        with torch.no_grad():
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
            logits = model(**inputs).logits
            all_logits.append(logits.cpu())
            all_labels.append(batch['labels'])

    preds = torch.sigmoid(torch.cat(all_logits)).numpy()
    labels = torch.cat(all_labels).numpy()
    pred_binary = (preds >= threshold).astype(int)
    labels_binary = (labels >= threshold).astype(int)

    f1 = f1_score(labels_binary, pred_binary, average='macro')
    precision, recall, f1_class, _ = precision_recall_fscore_support(
        labels_binary, pred_binary, average=None)

    return f1, precision, recall, f1_class


def main():
    """
    Ejecuta la evaluación de modelos BERT para clasificación multi-etiqueta.

    python evaluation_models.py \\
        --json_path /ruta/al/test.json \\
        --model_dirs /ruta/a/modelo1 /ruta/a/modelo2 \\
        --label_keys coherencia desinformacion rep_latinoamericana \\
                    nivel_educacional originalidad score_final \\
        --threshold 0.5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--model_dirs', nargs='+', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--label_keys', nargs='+', required=True)
    args = parser.parse_args()

    df = load_json_data(args.json_path, args.label_keys)

    for model_dir in args.model_dirs:
        print(f"Evaluando modelo: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir).to('cuda' if torch.cuda.is_available() else 'cpu')

        tokenized_dataset = tokenize_dataset(df, tokenizer)
        tokenized_dataset.set_format(type='torch', columns=['input_ids',
                                                            'attention_mask', 'labels'])

        f1, precision, recall, f1_class = evaluate(model, tokenized_dataset,
                                                   args.threshold)

        print(f"Macro F1: {f1:.4f}")
        for i, f1_c in enumerate(f1_class):
            print(
                f"Clase {i+1} | F1: {f1_c:.4f}",
                f"| Precision: {precision[i]:.4f}",
                f"| Recall: {recall[i]:.4f}")
        print("\n")


if __name__ == "__main__":
    main()