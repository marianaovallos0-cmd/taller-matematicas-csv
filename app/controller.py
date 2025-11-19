from model import valoresFaltantes, normalizacion, discretizacion, categorizacion

def aplicar_imputacion(df, metodo):
    if metodo == "KNN":
        return valoresFaltantes.imputar_knn(df)
    if metodo == "K-Modes":
        return valoresFaltantes.imputar_k_modes(df)
    if metodo == "Mean":
        return valoresFaltantes.imputar_media(df)
    if metodo == "Median":
        return valoresFaltantes.imputar_mediana(df)
    if metodo == "Mode":
        return valoresFaltantes.imputar_moda(df)
    return df

def aplicar_normalizacion(df, metodo):
    if metodo == "Z-Score":
        return normalizacion.z_score(df)
    if metodo == "Min-Max":
        return normalizacion.min_max(df)
    if metodo == "Log":
        return normalizacion.log_norm(df)
    return df

def aplicar_discretizacion(df, metodo):
    if metodo == "Equal Width":
        return discretizacion.discretizar_ancho_igual(df)
    if metodo == "Equal Frequency":
        return discretizacion.discretizar_frecuencia_igual(df)
    return df

def aplicar_arbol_decision(df, columna_objetivo, columnas_usar):
    return categorizacion.entrenar_arbol_decision(df, columna_objetivo, columnas_usar)