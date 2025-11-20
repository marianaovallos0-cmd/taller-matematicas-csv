# model/valoresFaltantes.py
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Solo '?' se considera inválido
_VALORES_INVALIDOS = {"?"}


def _limpiar_valores_invalidos(df: pd.DataFrame):
    """
    Reemplaza los valores inválidos por NaN y detecta si hubo alguno.
    """
    df_copy = df.copy()
    df_str = df_copy.astype(str)

    mask_invalidos = df_str.applymap(lambda x: x.strip() in _VALORES_INVALIDOS)

    hubo_error = mask_invalidos.any().any()

    df_copy = df_copy.mask(mask_invalidos, np.nan)

    return df_copy, hubo_error


def _respuesta_error():
    """
    Mensaje EXACTO que pediste, sin filas ni detalles adicionales.
    """
    return "No pongas el carácter '?'. Deja los valores faltantes vacíos."


def imputar_knn(df: pd.DataFrame, vecinos: int = 3):
    df_limpio, hubo_error = _limpiar_valores_invalidos(df)
    if hubo_error:
        return _respuesta_error()

    numeric_cols = df_limpio.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df_limpio

    imputer = KNNImputer(n_neighbors=vecinos)
    numeric_array = imputer.fit_transform(df_limpio[numeric_cols])
    df_limpio[numeric_cols] = pd.DataFrame(
        numeric_array, columns=numeric_cols, index=df_limpio.index
    )

    return df_limpio


def imputar_k_modes(df: pd.DataFrame, k: int = 3):
    df_limpio, hubo_error = _limpiar_valores_invalidos(df)
    if hubo_error:
        return _respuesta_error()

    object_cols = df_limpio.select_dtypes(include=['object', 'category']).columns
    for c in object_cols:
        moda = df_limpio[c].mode(dropna=True)
        if len(moda) > 0:
            df_limpio[c] = df_limpio[c].fillna(moda.iloc[0])
        else:
            df_limpio[c] = df_limpio[c].fillna(np.nan)

    return df_limpio


def imputar_media(df: pd.DataFrame):
    df_limpio, hubo_error = _limpiar_valores_invalidos(df)
    if hubo_error:
        return _respuesta_error()

    numeric_cols = df_limpio.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df_limpio

    means = df_limpio[numeric_cols].mean()
    df_limpio[numeric_cols] = df_limpio[numeric_cols].fillna(means)

    return df_limpio


def imputar_mediana(df: pd.DataFrame):
    df_limpio, hubo_error = _limpiar_valores_invalidos(df)
    if hubo_error:
        return _respuesta_error()

    numeric_cols = df_limpio.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df_limpio

    medians = df_limpio[numeric_cols].median()
    df_limpio[numeric_cols] = df_limpio[numeric_cols].fillna(medians)

    return df_limpio


def imputar_moda(df: pd.DataFrame):
    df_limpio, hubo_error = _limpiar_valores_invalidos(df)
    if hubo_error:
        return _respuesta_error()

    for col in df_limpio.columns:
        modos = df_limpio[col].mode(dropna=True)
        if modos.shape[0] > 0:
            df_limpio[col] = df_limpio[col].fillna(modos.iloc[0])

    return df_limpio
