import numpy as np
from sklearn.datasets import make_classification, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score


def generar_caso_de_uso_validar_estratificado():
    """
    Genera un caso de uso aleatorio para la función validar_estratificado.

    Devuelve
    --------
    input_dict : dict
        Diccionario con los argumentos que se le pasarán a la función:
            - 'X'          : np.ndarray shape (m, n)
            - 'y'          : np.ndarray shape (m,)
            - 'n_folds'    : int entre 3 y 5
            - 'tipo_modelo': str, uno de 'logistic', 'arbol', 'knn'
    output : dict
        Diccionario con las cuatro claves esperadas:
            - 'metricas_por_fold' : np.ndarray (n_folds, 3)
            - 'media_accuracy'    : float
            - 'media_f1'          : float
            - 'media_precision'   : float

    Notas
    -----
    Cada llamada varía el tamaño del dataset, el número de clases, el número
    de folds y el tipo de modelo, garantizando diversidad entre ejecuciones.
    """
    rng  = np.random.default_rng()
    seed = int(rng.integers(0, 10000))

    # ── Parámetros aleatorios ─────────────────────────────────────────────────
    n_samples   = int(rng.integers(80, 301))     # entre 80 y 300 muestras
    n_features  = int(rng.integers(3, 11))       # entre 3 y 10 features
    n_classes   = int(rng.integers(2, 4))        # 2 o 3 clases
    n_folds     = int(rng.integers(3, 6))        # 3, 4 o 5 folds
    tipo_modelo = str(rng.choice(["logistic", "arbol", "knn"]))

    # ── Dataset sintético (alterna entre dos generadores) ─────────────────────
    if rng.random() > 0.5:
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_classes,
            cluster_std=1.5,
            random_state=seed,
        )
        y = y.astype(int)
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(n_features, n_classes + 1),
            n_redundant=0,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=seed,
        )

    # ── INPUT ─────────────────────────────────────────────────────────────────
    input_dict = {
        "X":           X.copy(),
        "y":           y.copy(),
        "n_folds":     n_folds,
        "tipo_modelo": tipo_modelo,
    }

    # ── OUTPUT ESPERADO ───────────────────────────────────────────────────────
    _modelos = {
        "logistic": LogisticRegression(max_iter=1000),
        "arbol":    DecisionTreeClassifier(random_state=42),
        "knn":      KNeighborsClassifier(n_neighbors=5),
    }
    modelo = _modelos[tipo_modelo]
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    filas = []
    for train_idx, test_idx in skf.split(X, y):
        modelo.fit(X[train_idx], y[train_idx])
        y_pred = modelo.predict(X[test_idx])
        filas.append([
            accuracy_score(y[test_idx], y_pred),
            f1_score(y[test_idx], y_pred, average="macro"),
            precision_score(y[test_idx], y_pred, average="macro", zero_division=0),
        ])

    mat = np.array(filas)   # shape (n_folds, 3)

    output = {
        "metricas_por_fold": mat,
        "media_accuracy":    round(float(mat[:, 0].mean()), 4),
        "media_f1":          round(float(mat[:, 1].mean()), 4),
        "media_precision":   round(float(mat[:, 2].mean()), 4),
    }

    return input_dict, output


# ── Bloque de verificación rápida ─────────────────────────────────────────────
if __name__ == "__main__":
    inp, out = generar_caso_de_uso_validar_estratificado()

    print("=== INPUT ===")
    print(f"X.shape      : {inp['X'].shape}")
    print(f"y.shape      : {inp['y'].shape}")
    print(f"clases       : {sorted(set(inp['y'].tolist()))}")
    print(f"n_folds      : {inp['n_folds']}")
    print(f"tipo_modelo  : '{inp['tipo_modelo']}'")

    print("\n=== OUTPUT ESPERADO ===")
    print(f"metricas_por_fold.shape : {out['metricas_por_fold'].shape}")
    print(f"metricas_por_fold       :\n{out['metricas_por_fold'].round(4)}")
    print(f"media_accuracy          : {out['media_accuracy']}")
    print(f"media_f1                : {out['media_f1']}")
    print(f"media_precision         : {out['media_precision']}")

    # Comprobación de consistencia interna
    assert abs(out["media_accuracy"] -
               round(float(out["metricas_por_fold"][:, 0].mean()), 4)) < 1e-4
    print("\n✅ Consistencia interna verificada")
