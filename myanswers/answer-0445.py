import pandas as pd
import numpy as np


# =============================================================================
# SOLUCIÓN — Pregunta 0001
# Limpieza de Datos de Sensores Industriales
# Repo origen: pablocastanoj-maker/Ejercicio-LLMs-IA-UdeA
# =============================================================================

def limpiar_sensores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia un DataFrame de lecturas de sensores industriales resolviendo
    valores faltantes y filas duplicadas.

    Estrategia de limpieza (en orden):
      1. interpolate() — rellena NaNs internos por interpolación lineal,
         preservando la tendencia local de la señal.
      2. ffill()       — propaga el último valor válido hacia adelante,
         cubriendo NaNs al final del DataFrame.
      3. bfill()       — propaga el primer valor válido hacia atrás,
         cubriendo NaNs al inicio del DataFrame.
      4. drop_duplicates() — elimina filas exactamente iguales
         (registros enviados más de una vez por fallo de comunicación).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas numéricas de lecturas de sensores.
        Puede contener NaNs y filas duplicadas.

    Retorna
    -------
    pd.DataFrame
        DataFrame limpio: sin NaNs y sin filas duplicadas.
        El índice original se conserva (no se reinicia).
    """
    return df.interpolate().ffill().bfill().drop_duplicates()


# =============================================================================
# BLOQUE DE VALIDACIÓN
# Ejecutar directamente: python answer-0001.py
# =============================================================================

if __name__ == "__main__":

    def generar_caso_de_uso_limpiar_sensores():
        """Reproducción fiel del generador del compañero (question-0001-usecase-generator.py)."""
        n    = np.random.randint(5, 12)
        data = np.random.randn(n)

        for _ in range(np.random.randint(1, 4)):
            idx       = np.random.randint(0, n)
            data[idx] = np.nan

        df = pd.DataFrame({"sensor": data})
        df = pd.concat([df, df.iloc[:2]], ignore_index=True)

        df_clean   = df.interpolate().ffill().bfill().drop_duplicates()
        input_dict = {"df": df}
        output     = df_clean
        return input_dict, output

    # --- Prueba con múltiples seeds ---
    print("=" * 58)
    print("Validando limpiar_sensores...")
    print("=" * 58)

    for seed in [0, 7, 42, 99, 123]:
        np.random.seed(seed)
        params, esperado = generar_caso_de_uso_limpiar_sensores()
        resultado = limpiar_sensores(**params)

        n_in   = len(params["df"])
        n_nan  = params["df"].isna().sum().sum()
        n_out  = len(resultado)

        # Verificaciones estructurales
        assert isinstance(resultado, pd.DataFrame), \
            "ERROR: debe retornar un pd.DataFrame"
        assert resultado.isna().sum().sum() == 0, \
            f"ERROR seed={seed}: quedan {resultado.isna().sum().sum()} NaNs sin resolver"
        assert resultado.duplicated().sum() == 0, \
            f"ERROR seed={seed}: quedan {resultado.duplicated().sum()} filas duplicadas"

        # Verificación de coincidencia exacta con el output esperado
        res_reset = resultado.reset_index(drop=True)
        esp_reset = esperado.reset_index(drop=True)
        assert res_reset.equals(esp_reset), \
            f"ERROR seed={seed}: el DataFrame resultante no coincide con el esperado"

        print(f"  seed={seed:>3} | filas in={n_in:>2} → out={n_out:>2} "
              f"| NaNs={n_nan} | dups eliminados={n_in - n_out} | ✅ OK")

    print("=" * 58)
    print("✅ Todas las pruebas pasaron correctamente.")
    print("=" * 58)
