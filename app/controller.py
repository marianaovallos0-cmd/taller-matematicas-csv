from model import valoresFaltantes, normalizacion, discretizacion, categorizacion

def aplicar_imputacion(df, metodo):
    """Aplica método de imputación seleccionado"""
    if metodo == "KNN":
        return valoresFaltantes.imputar_knn(df)
    elif metodo == "K-Modes":
        return valoresFaltantes.imputar_k_modes(df)
    elif metodo == "Mean":
        return valoresFaltantes.imputar_media(df)
    elif metodo == "Median":
        return valoresFaltantes.imputar_mediana(df)
    elif metodo == "Mode":
        return valoresFaltantes.imputar_moda(df)
    else:
        return df

def aplicar_normalizacion(df, metodo):
    """Aplica método de normalización seleccionado"""
    if metodo == "Z-Score":
        return normalizacion.z_score(df)
    elif metodo == "Min-Max":
        return normalizacion.min_max(df)
    elif metodo == "Log":
        return normalizacion.log_norm(df)
    else:
        return df

def aplicar_discretizacion(df, metodo, target_column=None, bins=4):
    """Aplica método de discretización seleccionado"""
    if metodo == "Equal Width":
        return discretizacion.discretizar_ancho_igual(df, bins=bins)
    elif metodo == "Equal Frequency":
        return discretizacion.discretizar_frecuencia_igual(df, bins=bins)
    elif metodo == "ChiMerge" and target_column:
        return discretizacion.discretizar_chimerge(df, target_column, bins=bins)
    else:
        return df

def aplicar_arbol_decision(df, columna_objetivo, columnas_usar):
    """Aplica árbol de decisión para categorización"""
    return categorizacion.entrenar_arbol_decision(df, columna_objetivo, columnas_usar)