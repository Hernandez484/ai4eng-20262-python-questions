import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def generar_caso_de_uso_clasificar_sensores_con_ventanas():
    """
    Generador de casos de uso para la función clasificar_sensores_con_ventanas.
    Produce un input aleatorio y el output esperado basado en la lógica del ejercicio.
    """
    n_rows = np.random.randint(150, 300)
    ventana_rand = np.random.randint(2, 5)
    
    # Generar datos aleatorios
    data = {
        'temp': np.random.normal(50, 10, n_rows),
        'presion': np.random.normal(100, 20, n_rows),
        'target': np.random.choice([0, 1], n_rows)
    }
    df_input = pd.DataFrame(data)
    
    # Inyectar algunos infinitos para probar la limpieza con Numpy
    df_input.iloc[np.random.randint(0, n_rows), 0] = np.inf

    # --- Cálculo del Output Esperado ---
    df_proc = df_input.copy()
    features = ['temp', 'presion']
    
    # 1. Rolling std
    df_proc[features] = df_proc[features].rolling(window=ventana_rand).std()
    
    # 2. Numpy clean (inf to nan) and drop
    df_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_proc.dropna(inplace=True)
    
    X = df_proc[features]
    y = df_proc['target']
    
    model = LinearSVC(random_state=42, max_iter=2000)
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    
    input_dict = {
        "df": df_input,
        "target_col": "target",
        "ventana": ventana_rand
    }
    output = (model, acc)
    
    return input_dict, output
