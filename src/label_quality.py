import argparse
import json
import logging
import os
import re

from datasets import Dataset
from vllm import LLM, SamplingParams

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MAX_CHARACTERS = 5000
TEXT_OUTPUT_DIR = "/workspace1/gonzalo.fuentes/Latamgpt/llm_calidad/data/textos_500k"


def load_base_prompt(prompt_path):
    with open(prompt_path, encoding="utf-8") as f:
        return f.read()


def insert_prompt(base_prompt: str, prompt_to_insert: str):
    return base_prompt.replace("[PEGUE AQUÍ EL TEXTO]", prompt_to_insert)


def update_text(example, base_prompt, text_column):
    texto = example[text_column][:MAX_CHARACTERS]
    example["eval_prompt"] = insert_prompt(base_prompt, texto)
    example["texto_truncado"] = texto
    return example


def extract_json_block(text):
    try:
        # Paso 1: Buscar el bloque delimitado entre <<JSON>> y <<FIN>>
        match = re.search(r"<<JSON>>\s*({[\s\S]+?})\s*<<FIN>>", text)
        if not match:
            logging.warning(
                "No se encontró bloque JSON delimitado correctamente con",
                "<<JSON>> y <<FIN>>."
            )
            return None

        json_str = match.group(1).strip()

        # Paso 2: Arreglos defensivos de formato

        # a) Reemplazar comillas simples por dobles
        json_str = json_str.replace("'", '"')

        # b) Asegurar comillas en claves no entrecomilladas (solo por seguridad extra)
        json_str = re.sub(
            r"(?<=\n|\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'"\1":', json_str
        )

        # Paso 3: Intentar parsear como JSON
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.error(f"Error al decodificar JSON: {e}")
            return None

        # Paso 4: Validación de claves obligatorias
        required_keys = {
            "coherencia",
            "desinformacion",
            "etapa_1_valida",
            "representacion_latinoamericana",
            "nivel_educacional",
            "originalidad",
            "score_final",
        }

        missing = required_keys - parsed.keys()
        if missing:
            logging.warning(f"Faltan claves obligatorias en el JSON: {missing}")
            return None

        # Paso 5: Validar y normalizar valores numéricos
        for k in required_keys:
            try:
                parsed[k] = round(float(parsed[k]), 3)
                if not (0.000 <= parsed[k] <= 1.000):
                    logging.warning(f"Valor fuera de rango para {k}: {parsed[k]}")
                    return None
                if not re.match(r"^\d\.\d{3}$", f"{parsed[k]:.3f}"):
                    logging.warning(
                        "Formato inválido (no tiene tres decimales) para",
                        f"{k}: {parsed[k]}"
                    )
                    return None
            except Exception as e:
                logging.warning(
                    f"No se pudo convertir {k} a float: {parsed[k]} | Error: {e}"
                )
                return None

        # Paso 6: Validación opcional del campo "justificacion"
        if "justificacion" in parsed:
            justif = parsed["justificacion"]
            if not isinstance(justif, str):
                logging.warning("El campo 'justificacion' no es una cadena de texto.")
                return None

            if parsed["score_final"] == 0.000:
                if justif.strip().lower() == "sin justificacion":
                    logging.warning(
                        "Se esperaba una justificación real, pero se recibió",
                        "'sin justificacion'"
                    )
                    return None
            else:
                if justif.strip().lower() != "sin justificacion":
                    logging.warning(
                        "Se esperaba 'sin justificacion', pero se recibió otra cosa."
                    )
                    return None
        else:
            logging.warning("Falta el campo 'justificacion' en el JSON.")
            return None

        return parsed

    except Exception as e:
        logging.error(f"Error inesperado al extraer JSON: {e}")
        return None

        return parsed

    except Exception as e:
        logging.error(f"Error al extraer JSON: {e}")
        return None

        return parsed

    except Exception as e:
        logging.error(f"Error al parsear JSON: {e}")
        return None

        return parsed

    except Exception as e:
        logging.error(f"Error al parsear JSON: {e}")
        return None


def validar_score_final(evaluacion):
    try:
        c = evaluacion["coherencia"]
        d = evaluacion["desinformacion"]
        etapa = evaluacion["etapa_1_valida"]
        latam = evaluacion["representacion_latinoamericana"]
        edu = evaluacion["nivel_educacional"]
        orig = evaluacion["originalidad"]
        score = evaluacion["score_final"]

        etapa_calc = 1.000 if c > 0.5 and d > 0.5 else 0.000
        score_calc = (
            1.000 if etapa_calc == 1.000 and max(latam, edu, orig) > 0.5 else 0.000
        )

        errores = []
        if etapa != etapa_calc:
            errores.append(f"etapa_1_valida mal calculado (esperado: {etapa_calc:.3f})")
        if score != score_calc:
            errores.append(f"score_final mal calculado (esperado: {score_calc:.3f})")

        return errores if errores else None
    except Exception as e:
        return [f"Error de validación: {e}"]


def save_text_as_file(output_dir, idx, text):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{idx}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def process_batch(llm, batch, sampling_params, start_idx):
    outputs = llm.generate(list(batch["eval_prompt"]), sampling_params)
    results = {}

    for i, (vllm_output, input_text) in enumerate(
        zip(outputs, list(batch["texto_truncado"]))
    ):
        entry_id = start_idx + i
        generated_text = vllm_output.outputs[0].text.strip()
        result_json = extract_json_block(generated_text)

        save_text_as_file(TEXT_OUTPUT_DIR, entry_id, input_text)

        if result_json:
            errores = validar_score_final(result_json)
            if errores:
                logging.warning(f"[{entry_id}] Validación lógica fallida: {errores}")
                result_json["errores_validacion"] = errores

            logging.info(
                f"[{entry_id}] Evaluado → score_final =",
                f"{result_json.get('score_final')}"
            )
            results[entry_id] = {"id": entry_id, "evaluacion": result_json}
        else:
            logging.warning(f"[{entry_id}] JSON inválido o ausente.")
            results[entry_id] = {
                "id": entry_id,
                "evaluacion": {
                    "error": "No se pudo extraer JSON del output",
                    "output_llm": generated_text,
                },
            }

    return results


def main(args):
    try:
        base_prompt = load_base_prompt(args.prompt_path)
        logging.info(f"Cargando dataset desde {args.dataset_path}")
        dataset = Dataset.load_from_disk(args.dataset_path)
        dataset = dataset.map(lambda x: update_text(x, base_prompt, args.text_column))

        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=700,
            truncate_prompt_tokens=130000,
        )

        logging.info("Inicializando LLM")
        llm = LLM(
            model=args.model_path,
            download_dir=args.download_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_prefix_caching=True,
        )

        results_dict = {}
        curr_idx = 0

        while len(dataset) > curr_idx:
            batch = dataset[curr_idx : curr_idx + args.batch_size]
            batch_results = process_batch(llm, batch, sampling_params, curr_idx)
            results_dict.update(batch_results)

            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=4)

            curr_idx += args.batch_size
            logging.info(f"Procesados {curr_idx} ejemplos")

        logging.info(f"Evaluación completada. Resultados en {args.output_path}")

    except Exception as e:
        logging.error(f"Error general: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación FSM con LLM y puntuación continua"
    )
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--download_dir", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct"
    )
    parser.add_argument("--output_path", type=str, default="resultados_continuos.json")
    parser.add_argument("--text_column", type=str, default="texto")
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=50)

    args = parser.parse_args()
    main(args)
