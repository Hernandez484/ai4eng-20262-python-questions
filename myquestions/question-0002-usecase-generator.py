import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso

def generar_caso_de_uso_predecir_eficiencia_reducida():
    """
    Generador de casos de uso para la función predecir_eficiencia_reducida.
    """
    n_samples = np.random.randint(100, 200)
    n_features = np.random.randint(10, 20)
    n_comp = np.random.randint(2, 6)
    
    # Generar matrices aleatorias con Numpy
    X_input = np.random.rand(n_samples, n_features)
    y_input = np.random.rand(n_samples)
    
    # --- Cálculo del Output Esperado ---
    # 1. PCA
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_input)
    
    # 2. Lasso
    model = Lasso(alpha=0.5)
    model.fit(X_pca, y_input)
    
    # 3. Varianza acumulada con Numpy
    var_acumulada = np.sum(pca.explained_variance_ratio_)
    
    input_dict = {
        "X": X_input,
        "y": y_input,
        "n_componentes": n_comp
    }
    
    output = {
        'modelo': model,
        'varianza_total': float(var_acumulada),
        'coeficientes': model.coef_
    }
    
    return input_dict, output
