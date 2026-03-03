import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


def generar_caso_de_uso_rankear_importancia_features():
    """
    Genera un caso de uso aleatorio para la función rankear_importancia_features.

    Devuelve
    --------
    input_dict : dict
        Diccionario con los argumentos que se le pasarán a la función:
            - 'df'         : pd.DataFrame con columnas numéricas y NaN aleatorios
            - 'target_col' : str, nombre de la columna objetivo
    output : pd.DataFrame
        DataFrame con columnas ['feature', 'importancia'] ordenado de mayor
        a menor importancia (resultado esperado de rankear_importancia_features).

    Notas
    -----
    Cada llamada genera dimensiones, nombres de columna, porcentaje de NaN
    y número de clases distintos, garantizando variedad entre ejecuciones.
    """
    rng = np.random.default_rng()

    # ── Parámetros aleatorios ─────────────────────────────────────────────────
    n_rows  = int(rng.integers(30, 121))   # entre 30 y 120 filas
    n_feats = int(rng.integers(3, 10))     # entre 3 y 9 features

    vocabulario = [
        "edad", "ingreso", "consumo", "temperatura", "presion",
        "distancia", "velocidad", "humedad", "ph", "voltaje",
        "frecuencia", "nivel", "indice", "carga", "densidad",
        "glucosa", "peso", "altura", "latencia", "brillo",
    ]
    feat_names = list(rng.choice(vocabulario, size=n_feats, replace=False))
    target_col = str(rng.choice(["categoria", "clase", "tipo", "label", "target"]))

    # ── Datos sintéticos con NaN (~15 % por columna) ──────────────────────────
    data = rng.standard_normal((n_rows, n_feats))
    for j in range(n_feats):
        nan_mask = rng.random(n_rows) < 0.15
        data[nan_mask, j] = np.nan

    n_clases = int(rng.integers(2, 4))
    y_vals   = rng.integers(0, n_clases, size=n_rows).astype(float)

    df = pd.DataFrame(data, columns=feat_names)
    df[target_col] = y_vals

    # ── INPUT ─────────────────────────────────────────────────────────────────
    input_dict = {
        "df":         df.copy(),
        "target_col": target_col,
    }

    # ── OUTPUT ESPERADO ───────────────────────────────────────────────────────
    X_raw = df[feat_names].values

    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(X_raw)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, df[target_col].values)

    output = (
        pd.DataFrame({"feature": feat_names, "importancia": rf.feature_importances_})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )

    return input_dict, output


# ── Bloque de verificación rápida ─────────────────────────────────────────────
if __name__ == "__main__":
    inp, out = generar_caso_de_uso_rankear_importancia_features()

    print("=== INPUT ===")
    print(f"df.shape      : {inp['df'].shape}")
    print(f"df.columns    : {list(inp['df'].columns)}")
    print(f"target_col    : '{inp['target_col']}'")
    nan_count = inp['df'].drop(columns=[inp['target_col']]).isna().sum().sum()
    print(f"NaN en X      : {nan_count}")

    print("\n=== OUTPUT ESPERADO ===")
    print(out)
    print(f"\nimportancias suman 1: {abs(out['importancia'].sum() - 1.0) < 1e-6}")
    print(f"ordenado desc       : {out['importancia'].is_monotonic_decreasing}")
