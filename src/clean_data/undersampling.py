import pandas as pd

# Lista de columnas permitidas (ajusta según tu dataset)
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


def load_json(path):
    df = pd.read_json(path)
    cols = [col for col in COLUMNAS_PERMITIDAS if col in df.columns]
    df = df[cols]
    df = df.dropna(axis=1, how='all')
    return df


def count_positivos(df, threshold=0.5):
    return (df["originalidad"] >= threshold).sum()


def balancear_originalidad(df, n_pos, threshold=0.5):
    positivos = df[df["originalidad"] >= threshold].sample(
        n=n_pos, random_state=42
    )
    negativos = df[df["originalidad"] < threshold].sample(
        n=n_pos, random_state=42
    )
    return pd.concat([positivos, negativos]).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)


def balancear_todos_los_idiomas(json_paths: dict, threshold=0.5):
    dfs = {lang: load_json(path) for lang, path in json_paths.items()}

    positivos_por_idioma = {
        lang: count_positivos(df, threshold) for lang, df in dfs.items()
    }
    n_minimos = min(positivos_por_idioma.values())

    print("Originalidad >= {:.1f} por idioma:".format(threshold))
    for lang, count in positivos_por_idioma.items():
        print(f"  {lang}: {count}")
    print(f"\nUsaremos {n_minimos} positivos + {n_minimos} negativos por idioma.")

    balanceados = {
        lang: balancear_originalidad(df, n_pos=n_minimos, threshold=threshold)
        for lang, df in dfs.items()
    }

    df_total = pd.concat(balanceados.values()).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    return df_total


if __name__ == "__main__":
    rutas = {
        "balanceado": (
            ""
            ""
        )
    }

    df_balanceado = balancear_todos_los_idiomas(
        rutas, threshold=0.5
    )

    output_path = (
        ""
        ""
    )
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
