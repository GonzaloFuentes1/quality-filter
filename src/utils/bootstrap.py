import argparse
import os
import json
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)
from torch.nn.functional import sigmoid
from tqdm import tqdm
from datasets import load_from_disk, Dataset


def load_model_and_tokenizer(model_path, label_ids):
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = len(label_ids)
    config.problem_type = "multi_label_classification"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True
    )
    return model, tokenizer


def load_input_dataset(input_path):
    if os.path.isdir(input_path):
        print(f"Cargando dataset desde disco: {input_path}")
        dataset = load_from_disk(input_path)
        print(len(dataset))

        if "text" not in dataset.column_names:
            if "texto" in dataset.column_names:
                dataset = dataset.rename_column("texto", "text")
            elif "content" in dataset.column_names:
                dataset = dataset.rename_column("content", "text")
            else:
                raise ValueError(
                    "El dataset no tiene una columna reconocida como texto."
                )
        return dataset

    elif input_path.endswith(".json"):
        print(f"📄 Cargando dataset JSON: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not all("texto" in entry for entry in data):
            raise ValueError("El JSON debe tener objetos con la clave 'texto'.")
        return Dataset.from_list([{"text": item["texto"]} for item in data])

    else:
        raise ValueError(
            "Input debe ser un directorio (load_from_disk) o archivo .json"
        )


def predict_and_filter(
    dataset, model_path, label_ids, threshold,
    batch_size, local_gpu_id, return_dict
):
    torch.cuda.set_device(local_gpu_id)
    model, tokenizer = load_model_and_tokenizer(model_path, label_ids)
    model.eval().to(local_gpu_id)

    selected_texts = []
    class_counts = [0 for _ in label_ids]
    class_probs = [[] for _ in label_ids]

    for i in tqdm(
        range(0, len(dataset), batch_size),
        desc=f"[GPU {local_gpu_id}]"
    ):
        batch = dataset[i:i + batch_size]
        texts = batch["text"]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(local_gpu_id)

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = sigmoid(logits).cpu().numpy()

        for text, prob in zip(texts, probs):
            keep = False
            for idx in range(len(label_ids)):
                class_probs[idx].append(prob[idx])
                if prob[idx] >= threshold:
                    class_counts[idx] += 1
                    keep = True
            if keep:
                selected_texts.append({"texto": text})

    return_dict[local_gpu_id] = {
        "texts": selected_texts,
        "counts": class_counts,
        "probs": class_probs
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filtrado bootstrap con múltiples GPUs via CUDA_VISIBLE_DEVICES"
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Ruta al dataset (JSON o load_from_disk)"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Ruta al modelo Hugging Face entrenado"
    )
    parser.add_argument(
        "--label_ids", nargs="+", type=int, default=[0, 1],
        help="IDs de las etiquetas de interés"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Umbral mínimo para filtrar un texto"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Tamaño de batch por GPU"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Ruta de salida para el dataset filtrado (columna 'texto')"
    )
    args = parser.parse_args()

    dataset = load_input_dataset(args.input_path)

    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = list(range(len(visible_gpus.split(",")))) if visible_gpus \
        else list(range(torch.cuda.device_count()))

    if not gpu_ids:
        raise RuntimeError(
            "No se detectaron GPUs disponibles "
            "(¿estableciste CUDA_VISIBLE_DEVICES?)."
        )

    print(f"GPUs visibles: {gpu_ids}")
    print(f"Total de textos: {len(dataset)}")

    chunk_size = len(dataset) // len(gpu_ids)
    datasets_split = [
        dataset.select(range(i * chunk_size, (i + 1) * chunk_size))
        for i in range(len(gpu_ids) - 1)
    ]
    datasets_split.append(
        dataset.select(range((len(gpu_ids) - 1) * chunk_size, len(dataset)))
    )

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for local_id, global_id in enumerate(gpu_ids):
        p = mp.Process(
            target=predict_and_filter,
            args=(
                datasets_split[local_id],
                args.model_path,
                args.label_ids,
                args.threshold,
                args.batch_size,
                global_id,
                return_dict
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    all_texts = []
    total_counts = [0 for _ in args.label_ids]
    all_probs = [[] for _ in args.label_ids]

    for result in return_dict.values():
        all_texts.extend(result["texts"])
        for idx in range(len(args.label_ids)):
            total_counts[idx] += result["counts"][idx]
            all_probs[idx].extend(result["probs"][idx])

    if all_texts:
        final_dataset = Dataset.from_list(all_texts)
        final_dataset.save_to_disk(args.output_path)

        print("\n📊 Textos seleccionados por clase:")
        for idx, label in enumerate(args.label_ids):
            print(
                f"   🔹 Clase {label} (posición {idx}): "
                f"{total_counts[idx]} textos"
            )

        os.makedirs("graficos_bootstrap", exist_ok=True)
        for idx, label in enumerate(args.label_ids):
            plt.figure()
            plt.hist(all_probs[idx], bins=50, alpha=0.75)
            plt.axvline(
                args.threshold,
                color="red",
                linestyle="--",
                label="Threshold"
            )
            plt.title(f"Distribución de probabilidades - Clase {label}")
            plt.xlabel("Probabilidad")
            plt.ylabel("Frecuencia")
            plt.legend()
            plt.savefig(
                f"graficos_bootstrap/distribucion_clase_{label}.png"
            )

        print("Histogramas guardados en: graficos_bootstrap/")
        print(
            f"Dataset guardado en {args.output_path} con "
            f"{len(final_dataset)} textos."
        )
    else:
        print("No se encontraron textos que cumplan el umbral.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
