import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss


# =============================================================================
# SOLUCIÓN — Pregunta 0003
# Clasificación de Tipos de Vidrio (Ensambles y Log-Loss)
# Repo origen: Farleybr/Python-questions-LLM
# =============================================================================

def evaluar_cristales(X_train, X_test, y_train, y_test):
    """
    Entrena un GradientBoostingClassifier y evalúa su confianza global
    mediante Log-Loss sobre el conjunto de prueba.

    Contexto forense: el juez no solo quiere la clasificación del vidrio,
    sino la probabilidad matemática de que el modelo esté en lo correcto.
    El Log-Loss penaliza fuertemente predicciones de alta confianza
    incorrectas — cuanto más cercano a 0, más fiable el modelo.

    Parámetros
    ----------
    X_train : np.ndarray
        Matriz de features de entrenamiento (n_train × n_features).
    X_test : np.ndarray
        Matriz de features de prueba (n_test × n_features).
    y_train : np.ndarray
        Etiquetas de clase del conjunto de entrenamiento.
    y_test : np.ndarray
        Etiquetas de clase del conjunto de prueba (ground truth).

    Retorna
    -------
    tuple
        (modelo, puntuacion_log_loss)
        - modelo : GradientBoostingClassifier entrenado.
        - puntuacion_log_loss : float, pérdida logarítmica sobre X_test.
    """

    # ------------------------------------------------------------------
    # 1. Instanciar el ensamble de boosting
    #    random_state=42 garantiza reproducibilidad exacta
    # ------------------------------------------------------------------
    modelo = GradientBoostingClassifier(random_state=42)

    # ------------------------------------------------------------------
    # 2. Entrenar sobre el conjunto de entrenamiento
    #    Los árboles de boosting no requieren escalado previo
    # ------------------------------------------------------------------
    modelo.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 3. Predicciones probabilísticas (matriz n_test × n_clases)
    #    Cada fila suma 1; reflect la confianza por clase
    # ------------------------------------------------------------------
    probabilidades = modelo.predict_proba(X_test)

    # ------------------------------------------------------------------
    # 4. Calcular Log-Loss
    #    L = -1/N * Σ Σ y_ij * log(p_ij)
    #    Penaliza más las predicciones confiadas que resultan erróneas
    # ------------------------------------------------------------------
    puntuacion_log_loss = log_loss(y_test, probabilidades)

    return (modelo, puntuacion_log_loss)


# =============================================================================
# BLOQUE DE VALIDACIÓN
# Ejecutar directamente: python answer-0003.py
# =============================================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import random

    def generar_caso_de_uso_evaluar_cristales():
        """Reproducción fiel del generador del compañero (question-0003-usecase-generator.py)."""
        n_samples  = random.randint(300, 600)
        n_features = random.randint(7, 10)

        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_classes=3,
            n_informative=4, random_state=random.randint(1, 1000)
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        input_data = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
        }

        # Ground truth calculado igual que en el generador original
        modelo_gt = GradientBoostingClassifier(random_state=42)
        modelo_gt.fit(input_data['X_train'], input_data['y_train'])
        proba_gt  = modelo_gt.predict_proba(input_data['X_test'])
        loss_gt   = log_loss(input_data['y_test'], proba_gt)

        return input_data, (modelo_gt, loss_gt)

    # --- Prueba con múltiples seeds ---
    print("=" * 62)
    print("Validando evaluar_cristales...")
    print("=" * 62)

    for seed in [0, 7, 42, 99, 123]:
        random.seed(seed)
        params, esperado = generar_caso_de_uso_evaluar_cristales()
        modelo_esp, loss_esp = esperado

        resultado          = evaluar_cristales(**params)
        modelo_res, loss_res = resultado

        # Verificaciones estructurales
        assert isinstance(resultado, tuple) and len(resultado) == 2, \
            "ERROR: debe retornar una tupla de 2 elementos"
        assert isinstance(modelo_res, GradientBoostingClassifier), \
            "ERROR: el primer elemento debe ser GradientBoostingClassifier"
        assert isinstance(loss_res, float), \
            "ERROR: el segundo elemento debe ser un float"

        # Verificación numérica exacta del Log-Loss
        assert abs(loss_res - loss_esp) < 1e-9, \
            f"ERROR seed={seed}: log_loss {loss_res:.6f} != {loss_esp:.6f}"

        # Verificación de probabilidades idénticas al ground truth
        proba_res = modelo_res.predict_proba(params['X_test'])
        proba_esp = modelo_esp.predict_proba(params['X_test'])
        assert np.allclose(proba_res, proba_esp), \
            f"ERROR seed={seed}: las probabilidades no coinciden"

        n_test   = len(params['y_test'])
        n_clases = proba_res.shape[1]
        print(f"  seed={seed:>3} | X_train={params['X_train'].shape} "
              f"| clases={n_clases} | log_loss={loss_res:.4f} | ✅ OK")

    print("=" * 62)
    print("✅ Todas las pruebas pasaron correctamente.")
    print("=" * 62)
