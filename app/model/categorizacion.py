import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np

def entrenar_arbol_decision(df, columna_objetivo, columnas_usar):
    # Verificar que no falten valores en las predictoras
    predictors_with_nan = df[columnas_usar].isna().any()
    if predictors_with_nan.any():
        raise Exception("Faltan datos en las columnas predictoras. Complétalos antes de entrenar.")
    
    # Filtrar columnas necesarias
    data = df[columnas_usar + [columna_objetivo]].copy()
    
    # Eliminar filas donde el objetivo es NaN para entrenamiento
    data_clean = data.dropna(subset=[columna_objetivo])
    
    if len(data_clean) == 0:
        raise Exception("No hay datos con valores en la columna objetivo para entrenar")
    
    # Label encoding simple
    encoders = {}
    for col in data_clean.columns:
        if data_clean[col].dtype == "object":
            enc = LabelEncoder()
            data_clean[col] = enc.fit_transform(data_clean[col].astype(str))
            encoders[col] = enc
    
    # Separar datos para entrenar
    X = data_clean[columnas_usar]
    y = data_clean[columna_objetivo]
    
    # Modelo
    modelo = DecisionTreeClassifier(max_depth=4, random_state=0)
    modelo.fit(X, y)
    
    # Árbol
    arbol_raw = export_text(modelo, feature_names=list(X.columns))
    
    # Predecir valores faltantes del objetivo
    valores_rellenados = None
    filas_faltantes = data[data[columna_objetivo].isna()]
    
    if len(filas_faltantes) > 0:
        try:
            filas_cod = filas_faltantes[columnas_usar].copy()
            for col in filas_cod.columns:
                if col in encoders:
                    # Solo transformar valores que existen
                    mask = filas_cod[col].notna()
                    filas_cod.loc[mask, col] = encoders[col].transform(filas_cod.loc[mask, col].astype(str))
            
            pred = modelo.predict(filas_cod)
            if columna_objetivo in encoders:
                pred = encoders[columna_objetivo].inverse_transform(pred)
            
            filas_faltantes[columna_objetivo] = pred
            valores_rellenados = filas_faltantes
        except Exception as e:
            print(f"Error prediciendo valores faltantes: {e}")
    
    return {
        "arbol": arbol_raw,
        "columnas_usadas": columnas_usar,
        "valores_rellenados": valores_rellenados
    }