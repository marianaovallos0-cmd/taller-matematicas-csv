import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def z_score(df):
    """Normalización Z-Score"""
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return df_copy
    
    scaler = StandardScaler()
    df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
    return df_copy

def min_max(df, min_nuevo=0, max_nuevo=1):
    """Normalización Min-Max"""
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return df_copy
    
    scaler = MinMaxScaler(feature_range=(min_nuevo, max_nuevo))
    df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
    return df_copy

def log_norm(df):
    """Normalización logarítmica: log(x + 1)"""
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return df_copy
    
    # Aplicar solo a columnas positivas
    for col in numeric_cols:
        if (df_copy[col] >= 0).all():
            df_copy[col] = np.log1p(df_copy[col])
        else:
            print(f"Advertencia: Columna '{col}' tiene valores negativos, no se aplicó log-normalización")
    
    return df_copy