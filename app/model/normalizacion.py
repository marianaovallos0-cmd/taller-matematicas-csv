from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd

def z_score(df):
    df = df.copy()  # evitar modificar el original
    numeric = df.select_dtypes(include=['number'])

    if numeric.empty:
        return df  # no hay columnas numéricas

    scaler = StandardScaler()
    df[numeric.columns] = scaler.fit_transform(numeric)
    return df


def min_max(df):
    df = df.copy()
    numeric = df.select_dtypes(include=['number'])

    if numeric.empty:
        return df

    scaler = MinMaxScaler()
    df[numeric.columns] = scaler.fit_transform(numeric)
    return df


def logaritmica(df):
    df = df.copy()
    numeric = df.select_dtypes(include=['number'])

    if numeric.empty:
        return df

    # evitar errores: log1p permite ceros y valores negativos se ignoran con reemplazo
    safe_numeric = numeric.applymap(
        lambda x: np.log1p(x) if x > -1 else np.nan
    )

    df[numeric.columns] = safe_numeric
    return df
