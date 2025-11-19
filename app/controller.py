from model import valoresFaltantes, normalizacion, discretizacion, categorizacion

def aplicar_imputacion(df, metodo):
    # Restringimos solo a las 3 opciones permitidas
    if metodo == "KNN":
        return valoresFaltantes.imputar_knn(df)
    if metodo == "K-MEDIAS":
        return valoresFaltantes.imputar_k_means(df)
    if metodo == "K-MODAS":
        return valoresFaltantes.imputar_k_modes(df)
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
    if metodo == "ChiMerge":
        return discretizacion.discretizar_chimerge(df)
    return df

def discretizar_chimerge(df, num_bins=4):
    try:
        from chimerge import ChiMerge

        df_copy = df.copy()
        columnas_numericas = df_copy.select_dtypes(include=["int64", "float64"]).columns

        for col in columnas_numericas:
            df_copy[col] = ChiMerge(df_copy[col], max_intervals=num_bins)["bin"]

        return df_copy

    except Exception as e:
        print("Error en ChiMerge:", e)
        return df

def aplicar_arbol_decision(df, columna_objetivo, columnas_usar):
    return categorizacion.entrenar_arbol_decision(df, columna_objetivo, columnas_usar)
