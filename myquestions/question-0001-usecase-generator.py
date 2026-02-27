import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def generar_caso_de_uso_detectar_anomalias_pca():
    """
    Genera un caso de uso aleatorio para la función detectar_anomalias_pca.
    Retorna una tupla con el diccionario de inputs y el output esperado.
    """
    # 1. Generación de parámetros aleatorios
    n_train = np.random.randint(150, 300)
    n_test = np.random.randint(40, 80)
    n_features = np.random.randint(8, 15)
    threshold_percentile = np.random.randint(85, 96)
    
    # 2. Generar datos base con alta correlación (variables latentes)
    matriz_proyeccion = np.random.randn(3, n_features) 
    
    X_train_np = np.dot(np.random.randn(n_train, 3), matriz_proyeccion) + np.random.randn(n_train, n_features) * 0.05
    X_test_np = np.dot(np.random.randn(n_test, 3), matriz_proyeccion) + np.random.randn(n_test, n_features) * 0.05
    
    # Inyectar anomalías aleatorias en test para que el PCA falle al reconstruirlas
    num_anomalias = np.random.randint(2, 6)
    indices_anomalos = np.random.choice(range(n_test), size=num_anomalias, replace=False)
    X_test_np[indices_anomalos] += np.random.randn(num_anomalias, n_features) * 8
    
    columnas = [f'sensor_{i}' for i in range(n_features)]
    X_train = pd.DataFrame(X_train_np, columns=columnas)
    X_test = pd.DataFrame(X_test_np, columns=columnas)
    
    # 3. Diccionario de entrada (Input)
    input_dict = {
        'X_train': X_train,
        'X_test': X_test.copy(),
        'threshold_percentile': threshold_percentile
    }
    
    # 4. Calcular la salida esperada (Output)
    pca = PCA(n_components=0.90)
    pca.fit(X_train)
    
    X_test_reconstruido = pca.inverse_transform(pca.transform(X_test))
    
    mse_por_fila = np.mean((X_test.values - X_test_reconstruido)**2, axis=1)
    umbral = np.percentile(mse_por_fila, threshold_percentile)
    
    df_output = X_test.copy()
    df_output['alerta_integridad'] = mse_por_fila > umbral
    
    return input_dict, df_output
