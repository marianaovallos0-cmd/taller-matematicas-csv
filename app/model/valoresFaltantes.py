from sklearn.impute import KNNImputer
from kmodes.kmodes import KModes
import pandas as pd
import numpy as np

def imputar_knn(df, vecinos=3):
    """
    Imputa solo columnas numéricas con KNN.
    """
    imputer = KNNImputer(n_neighbors=vecinos)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df
    resultado = imputer.fit_transform(df[numeric_cols])
    df[numeric_cols] = resultado
    return df

def imputar_kmodes(df, k=3):
    """
    Imputación simple para variables categóricas:
    - si hay columnas tipo objeto, rellena con la moda (forma simple),
      porque usar KModes directamente requiere convertir todo.
    """
    object_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(object_cols) == 0:
        return df
    # estrategia simple: rellenar por la moda de cada columna
    for c in object_cols:
        moda = df[c].mode()
        if len(moda) > 0:
            df[c] = df[c].fillna(moda.iloc[0])
    return df

def imputar_media(df):
    return df.fillna(df.mean(numeric_only=True))

def imputar_mediana(df):
    return df.fillna(df.median(numeric_only=True))

def imputar_moda(df):
    # si hay columnas no numéricas, mode() devuelve fila; usamos iloc[0]
    modos = df.mode()
    if modos.shape[0] > 0:
        return df.fillna(modos.iloc[0])
    return df
