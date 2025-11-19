from model import valoresFaltantes, normalizacion, discretizacion, categorizacion

def aplicar_imputacion(df, metodo):
    """
    Aplica métodos de imputación según la selección del usuario.
    """
    if metodo == "KNN":
        return valoresFaltantes.imputar_knn(df)
    elif metodo == "K-Modes":
        return valoresFaltantes.imputar_k_modes(df)
    elif metodo == "K-Means":
        # Usamos la media como equivalente simple de K-Means
        return valoresFaltantes.imputar_media(df)
    else:
        return df

def aplicar_normalizacion(df, metodo):
    if metodo == "Z-Score":
        return normalizacion.z_score(df)
    elif metodo == "Min-Max":
        return normalizacion.min_max(df)
    elif metodo == "Log":
        return normalizacion.log_norm(df)
    else:
        return df

def aplicar_discretizacion(df, metodo):
    if metodo == "Equal Width":
        return discretizacion.discretizar_ancho_igual(df)
    elif metodo == "Equal Frequency":
        return discretizacion.discretizar_frecuencia_igual(df)
    else:
        return df

def aplicar_arbol_decision(df, columna_objetivo, columnas_usar):
    return categorizacion.entrenar_arbol_decision(df, columna_objetivo, columnas_usar)