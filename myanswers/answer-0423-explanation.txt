import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

def evaluar_cristales(X_train, X_test, y_train, y_test):
    """
    Entrena un modelo de GradientBoostingClassifier y evalúa su desempeño
    utilizando la métrica Log-Loss.
    
    Parámetros:
    X_train (array-like): Datos de entrenamiento (características).
    X_test (array-like): Datos de prueba (características).
    y_train (array-like): Etiquetas de entrenamiento.
    y_test (array-like): Etiquetas de prueba.
    
    Devuelve:
    tuple: (modelo_entrenado, valor_log_loss)
    """
    # 1. Inicializar y entrenar el ensamble GradientBoostingClassifier con random_state=42
    modelo = GradientBoostingClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    
    # 2. Realizar predicciones probabilísticas sobre el conjunto de prueba
    probabilidades = modelo.predict_proba(X_test)
    
    # 3. Calcular la pérdida logarítmica (Log-Loss)
    puntuacion_log_loss = log_loss(y_test, probabilidades)
    
    # 4. Devolver la tupla con el modelo y la métrica
    return modelo, float(puntuacion_log_loss)
