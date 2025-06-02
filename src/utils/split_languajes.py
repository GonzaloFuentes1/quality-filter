import json
from langdetect import detect
from multiprocessing import Pool
from tqdm import tqdm

n_samples = ""
input_path = (
    ""
)


# ------------------------------
# Función para detección de idioma
# ------------------------------
def detectar_idioma(entry):
    try:
        lang = detect(entry["texto"])
        return lang, entry
    except Exception as e:
        return "error", {
            "id": entry.get("id", None),
            "error": str(e)
        }


# ------------------------------
# Carga del JSON
# ------------------------------
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)


# ------------------------------
# Paralelización
# ------------------------------
with Pool(processes=100) as pool:
    results = list(
        tqdm(pool.imap(detectar_idioma, data), total=len(data))
    )


# ------------------------------
# Agrupar por idioma
# ------------------------------
es, en, pt = [], [], []
for lang, entry in results:
    if lang == "es":
        es.append(entry)
    elif lang == "en":
        en.append(entry)
    elif lang == "pt":
        pt.append(entry)
    elif lang == "error":
        print(
            f"Error en entrada con id {entry['id']}: {entry['error']}"
        )


# ------------------------------
# Guardar resultados
# ------------------------------
with open(
    f"test_textos_es_{n_samples}.json", "w", encoding="utf-8"
) as f:
    json.dump(es, f, ensure_ascii=False, indent=2)

with open(
    f"test_textos_en_{n_samples}.json", "w", encoding="utf-8"
) as f:
    json.dump(en, f, ensure_ascii=False, indent=2)

with open(
    f"test_textos_pt_{n_samples}.json", "w", encoding="utf-8"
) as f:
    json.dump(pt, f, ensure_ascii=False, indent=2)


print(
    f"Separados: {len(es)} español, {len(en)} inglés, {len(pt)} portugués."
)
