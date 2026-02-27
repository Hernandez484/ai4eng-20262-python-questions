import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def generar_caso_de_uso_predecir_series_tiempo():
    """
    Genera un caso de uso aleatorio para la función predecir_series_tiempo.
    Retorna una tupla con el diccionario de inputs y el output esperado.
    """
    # 1. Generación de parámetros aleatorios
    n_samples = np.random.randint(80, 200)
    n_lags = np.random.randint(1, 6)
    
    # 2. Generar una serie de tiempo sintética (tendencia senoidal + ruido)
    fechas = pd.date_range(start='2026-01-01', periods=n_samples, freq='D')
    tendencia = np.linspace(0, 15, n_samples)
    ventas = np.sin(tendencia) * 50 + np.random.randn(n_samples) * 10 + 100
    
    df_ventas = pd.DataFrame({'fecha': fechas, 'ventas': ventas})
    
    # 3. Diccionario de entrada (Input)
    input_dict = {
        'df_ventas': df_ventas.copy(),
        'n_lags': n_lags
    }
    
    # 4. Calcular la salida esperada (Output)
    df_temp = df_ventas.copy()
    columnas_lags = []
    
    for i in range(1, n_lags + 1):
        nombre_col = f'lag_{i}'
        df_temp[nombre_col] = df_temp['ventas'].shift(i)
        columnas_lags.append(nombre_col)
        
    df_temp = df_temp.dropna()
    
    X_esperado = df_temp[columnas_lags].values
    y_esperado = df_temp['ventas'].values
    
    modelo = LinearRegression()
    modelo.fit(X_esperado, y_esperado)
    predicciones = modelo.predict(X_esperado)
    
    mse_esperado = float(mean_squared_error(y_esperado, predicciones))
    
    output_esperado = (mse_esperado, predicciones)
    
    return input_dict, output_esperado
