
import json
import os

# Import config for utils
import sys
from multiprocessing import Pool

from langdetect import detect
from tqdm import tqdm

import config

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Use config for input path and n_samples
n_samples = getattr(config, 'N_SAMPLES', "")
input_path = getattr(config, 'SPLIT_LANG_INPUT_PATH', "")


def detectar_idioma(entry):
    try:
        lang = detect(entry["texto"])
        return lang, entry
    except Exception as e:
        return "error", {
            "id": entry.get("id", None),
            "error": str(e)
        }


with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)


with Pool(processes=100) as pool:
    results = list(
        tqdm(pool.imap(detectar_idioma, data), total=len(data))
    )


################################
# Group by language
################################
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


################################
# Save results
################################
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
    f"Separated: {len(es)} Spanish, {len(en)} English, {len(pt)} Portuguese."
)
