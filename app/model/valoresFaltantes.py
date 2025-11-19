from sklearn.impute import KNNImputer
# from kmodes.kmodes import KModes  # Descomenta si quieres usar KModes real
import pandas as pd
import numpy as np

def imputar_knn(df, vecinos=3):
    """Imputa solo columnas numéricas con KNN."""
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return df_copy

    imputer = KNNImputer(n_neighbors=vecinos)

    numeric_array = imputer.fit_transform(df_copy[numeric_cols])
    numeric_df = pd.DataFrame(numeric_array, columns=numeric_cols, index=df_copy.index)

    df_copy[numeric_cols] = numeric_df
    return df_copy


def imputar_k_modes(df, k=3):
    """Imputación para variables categóricas mediante moda."""
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


def imputar_media(df):
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return df_copy

    means = df_copy[numeric_cols].mean()
    df_copy[numeric_cols] = df_copy[numeric_cols].fillna(means)
    return df_copy


def imputar_mediana(df):
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return df_copy

    medians = df_copy[numeric_cols].median()
    df_copy[numeric_cols] = df_copy[numeric_cols].fillna(medians)
    return df_copy


def imputar_moda(df):
    df_copy = df.copy()

    for col in df_copy.columns:
        modos = df_copy[col].mode(dropna=True)
        if modos.shape[0] > 0:
            df_copy[col] = df_copy[col].fillna(modos.iloc[0])

    return df_copy
