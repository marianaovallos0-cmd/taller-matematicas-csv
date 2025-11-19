import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

def limpiar_numericos(df):
    """Rellena valores faltantes solo en columnas numéricas"""
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=["number"]).columns
    
    # Rellenar con mediana solo las numéricas
    for col in numeric_cols:
        if df_copy[col].isna().any():
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    return df_copy

def discretizar_ancho_igual(df, bins=4):
    df_clean = limpiar_numericos(df)
    numeric_cols = df_clean.select_dtypes(include=["number"]).columns
    
    if len(numeric_cols) == 0:
        return df_clean
    
    enc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    df_clean[numeric_cols] = enc.fit_transform(df_clean[numeric_cols])
    return df_clean

def discretizar_frecuencia_igual(df, bins=4):
    df_clean = limpiar_numericos(df)
    numeric_cols = df_clean.select_dtypes(include=["number"]).columns
    
    if len(numeric_cols) == 0:
        return df_clean
    
    enc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
    df_clean[numeric_cols] = enc.fit_transform(df_clean[numeric_cols])
    return df_clean

# Eliminar las funciones de ChiMerge problemáticas por ahora