import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def limpiar_numericos(df):
    """
    Rellena valores faltantes en columnas numéricas para discretización.
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return df_copy
    
    # Rellenar con mediana
    for col in numeric_cols:
        if df_copy[col].isna().any():
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    return df_copy

def discretizar_ancho_igual(df, bins=4):
    """Discretización por ancho igual"""
    df_clean = limpiar_numericos(df)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return df_clean
    
    for col in numeric_cols:
        try:
            # Usar cut de pandas directamente para mejor control
            df_clean[col] = pd.cut(df_clean[col], bins=bins, labels=False, duplicates='drop')
        except Exception as e:
            print(f"Error discretizando {col}: {e}. Saltando columna.")
    
    return df_clean

def discretizar_frecuencia_igual(df, bins=4):
    """Discretización por frecuencia igual"""
    df_clean = limpiar_numericos(df)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return df_clean
    
    for col in numeric_cols:
        try:
            # Usar qcut de pandas para frecuencia igual
            df_clean[col] = pd.qcut(df_clean[col], q=bins, labels=False, duplicates='drop')
        except Exception as e:
            print(f"Error discretizando {col}: {e}. Saltando columna.")
    
    return df_clean

def chi_square(a, b):
    """Calcula estadístico Chi-cuadrado entre dos arrays"""
    total = a.sum() + b.sum()
    if total == 0:
        return 0
    
    expected_a = (a.sum() / total) * (a + b)
    expected_b = (b.sum() / total) * (a + b)
    
    # Evitar división por cero
    expected_a = np.where(expected_a == 0, 1e-9, expected_a)
    expected_b = np.where(expected_b == 0, 1e-9, expected_b)
    
    chi = ((a - expected_a) ** 2 / expected_a).sum() + \
          ((b - expected_b) ** 2 / expected_b).sum()
    
    return chi

def discretizar_chimerge(df, target_column, bins=4):
    """Discretización ChiMerge simplificada"""
    if target_column not in df.columns:
        raise Exception("Debes seleccionar la columna objetivo para ChiMerge.")
    
    df_clean = limpiar_numericos(df)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    # Excluir la columna objetivo
    numeric_cols = [col for col in numeric_cols if col != target_column]
    
    if len(numeric_cols) == 0:
        return df_clean
    
    for col in numeric_cols:
        try:
            # Discretización simplificada usando quantiles
            df_clean[col] = pd.qcut(df_clean[col], q=bins, labels=False, duplicates='drop')
        except Exception as e:
            print(f"Error en ChiMerge para {col}: {e}")
    
    return df_clean