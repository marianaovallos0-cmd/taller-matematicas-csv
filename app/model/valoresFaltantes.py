# model/valoresFaltantes.py
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
# from kmodes.kmodes import KModes  # opcional, coméntalo si no está instalado
import pandas as pd
import numpy as np

def imputar_knn(df, vecinos=3):
    """
    Imputa solo columnas numéricas con KNN.
    Preserva nombres de columnas e índices.
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df_copy

    imputer = KNNImputer(n_neighbors=vecinos)
    numeric_array = imputer.fit_transform(df_copy[numeric_cols])
    numeric_df = pd.DataFrame(numeric_array, columns=numeric_cols, index=df_copy.index)
    df_copy[numeric_cols] = numeric_df
    return df_copy

def imputar_k_means(df, k=3, random_state=0):
    """
    Imputación con K-MEDIAS (KMeans).
    - Entrena KMeans en filas que NO tienen NaN en las columnas numéricas.
    - Para filas con NaN, asigna el centroide más cercano (usando dimensiones disponibles)
      y rellena los NaN con los valores medios del centroide.
    - Si no hay suficientes filas para clusterizar, cae a la media por columna.
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df_copy

    # Filas sin NaN en numéricas
    complete_mask = ~df_copy[numeric_cols].isna().any(axis=1)
    complete = df_copy.loc[complete_mask, numeric_cols]

    if complete.shape[0] < k:
        # No hay suficientes filas para KMeans: fallback a media
        means = df_copy[numeric_cols].mean()
        df_copy[numeric_cols] = df_copy[numeric_cols].fillna(means)
        return df_copy

    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(complete.values)
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)

    # Para cada fila que tenga NaN, calcular distancia a centroides usando columnas disponibles
    rows_with_nan = df_copy[df_copy[numeric_cols].isna().any(axis=1)]
    for idx, row in rows_with_nan.iterrows():
        available = row[numeric_cols].notna()
        if available.sum() == 0:
            # si no hay ninguna dimensión disponible, usar media global
            fill_vals = df_copy[numeric_cols].mean()
            df_copy.loc[idx, numeric_cols] = df_copy.loc[idx, numeric_cols].fillna(fill_vals)
            continue

        # calcular distancias solo sobre las dimensiones disponibles
        diffs = centroids.loc[:, available.index[available]].subtract(row[available.index[available]], axis=1)
        dists = (diffs**2).sum(axis=1)
        best = dists.idxmin()
        # rellenar NaN con valores del centroide elegido
        for col in numeric_cols:
            if pd.isna(df_copy.at[idx, col]):
                df_copy.at[idx, col] = centroids.at[best, col]

    return df_copy

def imputar_k_modes(df, k=3):
    """
    Imputación para variables categóricas.
    Implementación simple: rellenamos por moda por columna.
    (Si la dependencia 'kmodes' estuviera disponible, podemos usar KModes real.)
    """
    df_copy = df.copy()
    object_cols = df_copy.select_dtypes(include=['object', 'category']).columns
    if len(object_cols) == 0:
        return df_copy

    for c in object_cols:
        moda = df_copy[c].mode(dropna=True)
        if len(moda) > 0:
            df_copy[c] = df_copy[c].fillna(moda.iloc[0])
        else:
            df_copy[c] = df_copy[c].fillna("missing")
    return df_copy

def imputar_mix(df):
    """
    Función auxiliar usada por el entrenador de árbol:
    1) aplica KNN sobre numéricas (si procede)
    2) aplica K-MODAS (modo por columna) sobre categóricas
    Esta es la función que se ejecuta "por detrás" cuando el usuario pide árbol y
    hay predictoras con NaN.
    """
    df_copy = df.copy()
    df_copy = imputar_knn(df_copy, vecinos=3)
    df_copy = imputar_k_modes(df_copy)
    return df_copy
