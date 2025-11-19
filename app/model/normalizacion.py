from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd

def z_score(df):
    scaler = StandardScaler()
    numeric = df.select_dtypes(include=['number'])
    df[numeric.columns] = scaler.fit_transform(numeric)
    return df

def min_max(df, min_nuevo=0, max_nuevo=1):
    numeric = df.select_dtypes(include=['number'])
    min_actual = numeric.min()
    max_actual = numeric.max()
    df[numeric.columns] = ((numeric - min_actual) / (max_actual - min_actual)) * (max_nuevo - min_nuevo) + min_nuevo
    return df

def log_norm(df):
    """Normalización logarítmica: log(x + 1)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df
    df[numeric_cols] = df[numeric_cols].apply(lambda col: np.log(col + 1))
    return df
