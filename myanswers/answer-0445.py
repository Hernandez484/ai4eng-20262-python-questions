import pandas as pd

def limpiar_sensores(df):
    """
    Limpia los datos de un DataFrame de sensores aplicando interpolación lineal,
    relleno hacia adelante (ffill), relleno hacia atrás (bfill) y eliminación de duplicados.
    
    Argumentos:
    df -- pd.DataFrame original con posibles valores nulos y filas duplicadas.
    
    Devuelve:
    pd.DataFrame limpio siguiendo la secuencia exacta del generador de casos de uso.
    """
    # 1. Interpola los valores faltantes (NaN) usando interpolación lineal.
    # 2. Rellena los valores restantes con ffill y luego bfill.
    # 3. Elimina filas duplicadas.
    df_clean = df.interpolate().ffill().bfill().drop_duplicates()
    
    # 4. Devuelve el DataFrame limpio.
    return df_clean
