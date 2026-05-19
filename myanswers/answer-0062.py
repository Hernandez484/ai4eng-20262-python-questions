import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# =============================================================================
# SOLUCIÓN — Pregunta 0002
# Detección de Inyecciones de Datos en Sensores de IoT
# =============================================================================

def detectar_intrusiones_temporales(df, contamination=0.01):
    """
    Detecta anomalías temporales en lecturas de sensores de presión de agua.

    Una lectura puede ser numéricamente válida (dentro del rango permitido)
    pero anómala en su contexto horario (e.g., presión alta a las 3:00 AM).
    Esta función aplica ingeniería de características cíclicas + Isolation
    Forest para identificar esos patrones sospechosos.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con al menos las columnas:
        - 'hora'    : int, hora del día en rango [0, 23]
        - 'presion' : float, lectura de presión del sensor
    contamination : float, optional
        Proporción esperada de anomalías en el dataset (default=0.01).
        Se pasa directamente a IsolationForest.

    Retorna
    -------
    pd.DataFrame
        El DataFrame original con una nueva columna booleana 'es_anomalia':
        - True  → punto anómalo (IsolationForest devolvió -1)
        - False → punto normal  (IsolationForest devolvió +1)
    """

    # Trabajamos sobre una copia para no alterar el DataFrame de trabajo
    df_proc = df.copy()

    # ------------------------------------------------------------------
    # 1. Ingeniería de Características Cíclicas
    #    Codificamos la hora como punto en el círculo unitario para que
    #    el modelo comprenda que 23:00 y 00:00 están próximas entre sí.
    # ------------------------------------------------------------------
    df_proc['hora_sin'] = np.sin(2 * np.pi * df_proc['hora'] / 24)
    df_proc['hora_cos'] = np.cos(2 * np.pi * df_proc['hora'] / 24)

    # ------------------------------------------------------------------
    # 2. Preparación de Datos
    #    Eliminamos 'hora' para evitar redundancia lineal con hora_sin/cos.
    #    X contiene: [presion, hora_sin, hora_cos]
    # ------------------------------------------------------------------
    X = df_proc.drop(columns=['hora'])

    # ------------------------------------------------------------------
    # 3. Detección de Anomalías con Isolation Forest
    #    -1 → anomalía  |  +1 → normal
    # ------------------------------------------------------------------
    model = IsolationForest(contamination=contamination, random_state=42)
    predicciones = model.fit_predict(X)

    # ------------------------------------------------------------------
    # 4. Retorno: columna booleana sobre el DataFrame ORIGINAL
    # ------------------------------------------------------------------
    df['es_anomalia'] = predicciones == -1

    return df


# =============================================================================
# BLOQUE DE VALIDACIÓN
# Ejecutar directamente: python answer-0002.py
# =============================================================================

if __name__ == '__main__':
    import random
    from sklearn.ensemble import IsolationForest  # ya importado, explícito aquí

    # Reproducir el generador de casos de uso del compañero (PDF adjunto)
    def generar_caso_de_uso_detectar_intrusiones_temporales():
        n_samples  = random.randint(50, 150)
        contam_val = random.uniform(0.01, 0.05)
        horas      = [random.randint(0, 23) for _ in range(n_samples)]
        presiones  = [
            abs(10 * np.sin(np.pi * h / 24) + random.uniform(0, 2))
            for h in horas
        ]
        df_input   = pd.DataFrame({'hora': horas, 'presion': presiones})
        input_dict = {'df': df_input.copy(), 'contamination': contam_val}
        output_expected = detectar_intrusiones_temporales(
            df_input.copy(), contamination=contam_val
        )
        return input_dict, output_expected

    # --- Prueba con múltiples seeds para robustez ---
    print("=" * 60)
    print("Validando detectar_intrusiones_temporales...")
    print("=" * 60)

    for seed in [0, 1, 42, 99, 123]:
        random.seed(seed)
        params, resultado_esperado = generar_caso_de_uso_detectar_intrusiones_temporales()
        mi_resultado = detectar_intrusiones_temporales(**params)

        # Verificaciones estructurales
        assert 'es_anomalia' in mi_resultado.columns, \
            "ERROR: falta la columna 'es_anomalia'"
        assert mi_resultado['es_anomalia'].dtype == bool, \
            "ERROR: 'es_anomalia' debe ser de tipo bool"
        assert 'hora' in mi_resultado.columns, \
            "ERROR: el DataFrame retornado debe conservar la columna 'hora'"
        assert len(mi_resultado) == len(params['df']), \
            "ERROR: el número de filas no debe cambiar"

        # Verificación de coincidencia exacta con el generador
        coincide = (mi_resultado['es_anomalia'] == resultado_esperado['es_anomalia']).all()
        assert coincide, \
            f"ERROR seed={seed}: el resultado no coincide con el esperado"

        anomalias = mi_resultado['es_anomalia'].sum()
        total     = len(mi_resultado)
        print(f"  seed={seed:>3} | {total:>3} registros | "
              f"{anomalias:>2} anomalías ({anomalias/total*100:.1f}%) | ✅ OK")

    print("=" * 60)
    print("✅ Todas las pruebas pasaron correctamente.")
    print("=" * 60)
