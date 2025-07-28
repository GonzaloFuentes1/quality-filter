import os

import pandas as pd

import config

# List of allowed columns (adjust according to your dataset)
COLUMNAS_PERMITIDAS = [
    "coherencia",
    "desinformacion",
    "representacion_latinoamericana",
    "nivel_educacional",
    "originalidad",
    "score_final",
    "justificacion",
    "texto"
]


# Numeric metrics that can be used for balancing
METRICAS_NUMERICAS = [
    "coherencia",
    "desinformacion",
    "representacion_latinoamericana",
    "nivel_educacional",
    "originalidad",
    "score_final"
]


def load_json(path):
    df = pd.read_json(path)
    cols = [col for col in COLUMNAS_PERMITIDAS if col in df.columns]
    df = df[cols]
    df = df.dropna(axis=1, how='all')
    return df


def count_positivos(df, metrica, threshold=0.5):
    return (df[metrica] >= threshold).sum()


def encontrar_metrica_limitante(dfs, threshold=0.5):
    """Finds the metric with the fewest positive values in total."""
    conteos_por_metrica = {}

    for metrica in METRICAS_NUMERICAS:
        # Check that the metric exists in all dataframes
        if all(metrica in df.columns for df in dfs.values()):
            total_positivos = sum(
                count_positivos(df, metrica, threshold) for df in dfs.values()
            )
            conteos_por_metrica[metrica] = total_positivos

    if not conteos_por_metrica:
        raise ValueError("No valid metrics found in all dataframes")

    metrica_limitante = min(conteos_por_metrica, key=conteos_por_metrica.get)
    return metrica_limitante, conteos_por_metrica


def balancear_por_metrica(df, n_pos, metrica, threshold=0.5):
    positivos = df[df[metrica] >= threshold].sample(
        n=n_pos, random_state=42
    )
    negativos = df[df[metrica] < threshold].sample(
        n=n_pos, random_state=42
    )
    return pd.concat([positivos, negativos]).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)


def balancear_todos_los_idiomas(json_paths: dict, threshold=0.5):
    dfs = {lang: load_json(path) for lang, path in json_paths.items()}

    # Encontrar la métrica limitante
    metrica_limitante, conteos_por_metrica = encontrar_metrica_limitante(dfs, threshold)

    print(f"Conteo de valores >= {threshold} por métrica:")
    for metrica, count in conteos_por_metrica.items():
        print(f"  {metrica}: {count}")
    print(f"\nMétrica limitante: {metrica_limitante}")

    positivos_por_idioma = {
        lang: count_positivos(df, metrica_limitante, threshold)
        for lang, df in dfs.items()
    }
    n_minimos = min(positivos_por_idioma.values())

    print(f"\n{metrica_limitante} >= {threshold} por idioma:")
    for lang, count in positivos_por_idioma.items():
        print(f"  {lang}: {count}")
    print(f"\nUsaremos {n_minimos} positivos + {n_minimos} negativos por idioma.")

    balanceados = {
        lang: balancear_por_metrica(df, n_pos=n_minimos, metrica=metrica_limitante,
                                    threshold=threshold)
        for lang, df in dfs.items()
    }

    df_total = pd.concat(balanceados.values()).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    return df_total


if __name__ == "__main__":

    # Use config for input and output paths
    rutas = {
        "balanceado": config.SPLIT_LANG_INPUT_PATH
    }

    df_balanceado = balancear_todos_los_idiomas(
        rutas, threshold=config.CLEAN_DATA_THRESHOLD
    )

    output_path = os.path.join(config.DATA_DIR,
                               f"balanced_texts_{config.N_SAMPLES}.json")
    df_balanceado.to_json(
        output_path,
        orient="records",
        force_ascii=False,
        indent=2
    )

    print(
        f"\nDataset final: {df_balanceado.shape[0]} textos balanceados "
        f"(multilingüe)"
    )
