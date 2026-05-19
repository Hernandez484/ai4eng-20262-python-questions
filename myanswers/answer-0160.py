import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# =============================================================================
# SOLUCIÓN — Pregunta 0004
# Residuos Estandarizados de Regresión Lineal
# Repo origen: daniel-riosr/Programacion-con-LLMs
# =============================================================================

def residuos_estandarizados(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> np.ndarray:
    """
    Entrena una regresión lineal sobre datos escalados y retorna el array
    de residuos estandarizados sobre el conjunto de prueba.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas numéricas y sin NaNs.
        Todas las columnas excepto target_col se usan como features.
    target_col : str
        Nombre de la columna objetivo (variable continua).
    test_size : float, optional
        Proporción del conjunto de prueba (default: 0.2).
    random_state : int, optional
        Semilla para reproducibilidad en train_test_split (default: 42).

    Retorna
    -------
    np.ndarray
        Array 1D de floats con los residuos estandarizados del conjunto
        de test, redondeados a 4 decimales.
        Un valor con |residuo| > 2 se considera sospechoso.
    """

    # ------------------------------------------------------------------
    # 1. Separar X e y
    # ------------------------------------------------------------------
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # ------------------------------------------------------------------
    # 2. Dividir en train / test
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ------------------------------------------------------------------
    # 3. Escalar X con StandardScaler
    #    fit solo sobre train para evitar data leakage
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 4. Entrenar LinearRegression y predecir sobre test
    # ------------------------------------------------------------------
    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    # ------------------------------------------------------------------
    # 5. Calcular residuos y estandarizar
    #    residuos = y_test - y_pred
    #    estandarizados = (residuos - mean) / std  [ddof=0 por defecto]
    # ------------------------------------------------------------------
    residuos = y_test - y_pred
    estandarizados = (residuos - residuos.mean()) / residuos.std()

    # ------------------------------------------------------------------
    # 6. Redondear a 4 decimales (requerido por el generador de casos)
    # ------------------------------------------------------------------
    return np.round(estandarizados, 4)


# =============================================================================
# BLOQUE DE VALIDACIÓN
# Ejecutar directamente: python answer-0004.py
# =============================================================================

if __name__ == "__main__":

    def generar_caso_de_uso_residuos_estandarizados() -> tuple:
        """Reproducción fiel del generador del compañero (question-0004-usecase-generator.py)."""
        rng = np.random.default_rng()
        n_samples    = int(rng.integers(60, 250))
        n_features   = int(rng.integers(2, 6))
        noise_std    = float(rng.uniform(0.5, 4.0))
        test_size    = float(rng.choice([0.2, 0.25, 0.3]))
        random_state = int(rng.integers(0, 500))

        rng2          = np.random.default_rng(random_state)
        feature_names = [f"x_{i}" for i in range(n_features)]
        X_raw = rng2.normal(size=(n_samples, n_features))
        coef  = rng2.uniform(-3, 3, size=n_features)
        y_raw = X_raw @ coef + rng2.normal(scale=noise_std, size=n_samples)

        df         = pd.DataFrame(X_raw, columns=feature_names)
        target_col = "target"
        df[target_col] = y_raw

        # Output esperado (calculado igual que en el generador original)
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        model     = LinearRegression()
        model.fit(X_train_s, y_train)
        y_pred    = model.predict(X_test_s)
        residuos  = y_test - y_pred
        output    = np.round((residuos - residuos.mean()) / residuos.std(), 4)

        input_dict = {
            "df": df,
            "target_col": target_col,
            "test_size": test_size,
            "random_state": random_state,
        }
        return input_dict, output

    # --- Prueba con múltiples seeds para robustez ---
    print("=" * 62)
    print("Validando residuos_estandarizados...")
    print("=" * 62)

    for seed in [0, 7, 42, 100, 999]:
        np.random.seed(seed)
        params, esperado = generar_caso_de_uso_residuos_estandarizados()
        resultado = residuos_estandarizados(**params)

        # Verificaciones estructurales
        assert isinstance(resultado, np.ndarray), \
            "ERROR: debe retornar np.ndarray"
        assert resultado.ndim == 1, \
            "ERROR: el array debe ser 1D"
        assert resultado.shape == esperado.shape, \
            f"ERROR seed={seed}: shape {resultado.shape} != {esperado.shape}"

        # Verificación de valores exactos (tolerancia por redondeo)
        assert np.allclose(resultado, esperado, atol=1e-4), \
            f"ERROR seed={seed}: los valores no coinciden con el esperado"

        n        = len(resultado)
        sospech  = (np.abs(resultado) > 2).sum()
        print(f"  seed={seed:>3} | n_test={n:>3} | |res|>2: {sospech} "
              f"({sospech/n*100:.1f}%) | ✅ OK")

    print("=" * 62)
    print("✅ Todas las pruebas pasaron correctamente.")
    print("=" * 62)
