import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def imputar_knn(df, vecinos=3):
    """
    Imputa solo columnas numéricas con KNN.
    Versión robusta que maneja casos edge.
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return df_copy
    
    # Verificar que hay suficientes datos para KNN
    non_null_count = df_copy[numeric_cols].notna().sum().min()
    
    if non_null_count < vecinos:
        # Fallback a mediana si no hay suficientes datos
        print(f"Advertencia: No hay suficientes datos para KNN (vecinos={vecinos}). Usando mediana.")
        return imputar_mediana(df_copy)
    
    try:
        imputer = KNNImputer(n_neighbors=min(vecinos, non_null_count))
        numeric_array = imputer.fit_transform(df_copy[numeric_cols])
        numeric_df = pd.DataFrame(numeric_array, columns=numeric_cols, index=df_copy.index)
        df_copy[numeric_cols] = numeric_df
        return df_copy
    except Exception as e:
        print(f"Error en KNN: {e}. Usando mediana como fallback.")
        return imputar_mediana(df_copy)

def imputar_k_modes(df, k=3):
    """
    Imputación para variables categóricas usando moda.
    """
    df_copy = df.copy()
    object_cols = df_copy.select_dtypes(include=['object', 'category']).columns
    
    if len(object_cols) == 0:
        return df_copy
    
    for c in object_cols:
        moda_vals = df_copy[c].mode(dropna=True)
        if len(moda_vals) > 0:
            df_copy[c] = df_copy[c].fillna(moda_vals.iloc[0])
        else:
            # Si no hay moda (columna totalmente NaN), rellenamos con "Valor_Desconocido"
            df_copy[c] = df_copy[c].fillna("Valor_Desconocido")
    
    return df_copy

def imputar_media(df):
    """
    Rellena NaN SOLO en columnas numéricas usando la media de cada columna.
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return df_copy
    
    # Calcular medias ignorando NaN
    means = df_copy[numeric_cols].mean()
    df_copy[numeric_cols] = df_copy[numeric_cols].fillna(means)
    
    return df_copy

def imputar_mediana(df):
    """
    Rellena NaN SOLO en columnas numéricas usando la mediana de cada columna.
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return df_copy
    
    medians = df_copy[numeric_cols].median()
    df_copy[numeric_cols] = df_copy[numeric_cols].fillna(medians)
    
    return df_copy

def imputar_moda(df):
    """
    Rellena NaN por la moda columna por columna (aplica a numéricas y categóricas).
    """
    df_copy = df.copy()
    
    for col in df_copy.columns:
        modos = df_copy[col].mode(dropna=True)
        if modos.shape[0] > 0:
            df_copy[col] = df_copy[col].fillna(modos.iloc[0])
        else:
            # Si toda la columna es NaN, llenar con valor por defecto según tipo
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(0)
            else:
                df_copy[col] = df_copy[col].fillna("Desconocido")
    
    return df_copy