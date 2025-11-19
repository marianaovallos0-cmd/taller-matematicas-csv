# categorizacion.py
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from model import valoresFaltantes

# -----------------------
# Helpers de encoding
# -----------------------
def _make_mapping(series):
    """Mapeo value(string) -> int y su inverso. Mantiene orden de aparición."""
    uniques = list(series.dropna().astype(str).unique())
    mapping = {v: i for i, v in enumerate(uniques)}
    inverse = {i: v for v, i in mapping.items()}
    return mapping, inverse

def _encode_series(series, mapping):
    """Encode una serie usando mapping; NaN -> -1; valores desconocidos -> -1."""
    return series.apply(lambda v: mapping.get(str(v), -1) if pd.notna(v) else -1).astype(int)

def _decode_value(val, inverse_map):
    """Decodifica tratando -1 como NaN/None."""
    if val is None:
        return None
    if int(val) == -1:
        return None
    return inverse_map.get(int(val), str(val))

# -----------------------
# Generación de reglas
# -----------------------
def _nice_condition_for_split(feature_name, threshold, is_categorical, inverse_map=None):
    """
    Dado un split "feature <= threshold" devuelve una condición legible.
    Para categóricas: intenta transformar en 'feature = X' si el split corresponde a una categoría única.
    Si el split separa varias categorías, lo expresa como 'feature in {A,B}'.
    Para numéricas: devuelve 'feature <= threshold' con 2 decimales.
    """
    if is_categorical and inverse_map is not None:
        # thresholds suelen ser flotantes; los códigos son ints 0..k-1
        # interpretamos que la separación es por conjuntos de códigos <= floor(threshold)
        t = float(threshold)
        upper = int(np.floor(t))
        # categorías incluidas en la rama izquierda = códigos <= upper
        cats = [inverse_map[i] for i in range(0, upper + 1) if i in inverse_map]
        if len(cats) == 0:
            return f"{feature_name} <= {threshold:.2f}"
        if len(cats) == 1:
            return f"{feature_name} = {cats[0]}"
        return f"{feature_name} in {{{', '.join(map(str, cats))}}}"
    else:
        # numérico: mostrar comparación con 2 decimales
        return f"{feature_name} <= {threshold:.2f}"

def reglas_legibles(modelo, feature_names, encoders, target_encoder):
    tree = modelo.tree_
    rules = []

    def recorrer(nodo, regla_actual):
        if tree.feature[nodo] != -2:  # Si no es hoja
            feature = feature_names[tree.feature[nodo]]
            threshold = tree.threshold[nodo]

            # Si la columna fue codificada, recuperar el encoder
            if feature in encoders:
                encoder = encoders[feature]
                categorias = encoder.classes_
                codigos = range(len(categorias))

                menores = [cat for cat, code in zip(categorias, codigos) if code <= threshold]
                mayores = [cat for cat, code in zip(categorias, codigos) if code > threshold]

                # Rama izquierda
                regla_left = regla_actual + [f"{feature} ∈ {menores}"]
                recorrer(tree.children_left[nodo], regla_left)

                # Rama derecha
                regla_right = regla_actual + [f"{feature} ∈ {mayores}"]
                recorrer(tree.children_right[nodo], regla_right)

            else:
                # Columna numérica normal
                regla_left = regla_actual + [f"{feature} <= {threshold:.2f}"]
                recorrer(tree.children_left[nodo], regla_left)

                regla_right = regla_actual + [f"{feature} > {threshold:.2f}"]
                recorrer(tree.children_right[nodo], regla_right)

        else:
            # Es hoja → obtener clase
            clase = target_encoder.inverse_transform([np.argmax(tree.value[nodo])])[0]
            regla_final = " Y ".join(regla_actual)
            rules.append(f"Si {regla_final}, entonces {clase}")

    recorrer(0, [])
    return rules


# -----------------------
# Entrenamiento y predicción
# -----------------------
def entrenar_arbol_decision(df, columna_objetivo, columnas_usar):
    """
    - df: DataFrame original
    - columna_objetivo: nombre de la columna objetivo (y)
    - columnas_usar: lista de columnas predictoras (X)
    Retorna dict con: arbol (raw), reglas (legibles), columnas_usadas, valores_rellenados (DataFrame con filas del objetivo rellenadas)
    """
    # Validaciones
    if columna_objetivo not in df.columns:
        raise Exception("La columna objetivo no existe en el DataFrame.")
    for c in columnas_usar:
        if c not in df.columns:
            raise Exception(f"Columna predictora inexistente: {c}")

    # Si hay NaN en predictoras, aplicamos imputación automática (mix)
    if df[columnas_usar].isna().any(axis=None):
        df = valoresFaltantes.imputar_mix(df)

    data = df[columnas_usar + [columna_objetivo]].copy()

    # Guardar filas donde el objetivo está vacío (para rellenar luego)
    filas_faltantes_idx = data[data[columna_objetivo].isna()].index.tolist()
    filas_faltantes = data.loc[filas_faltantes_idx].copy() if len(filas_faltantes_idx) > 0 else pd.DataFrame()

    # Preparar encoders y maps para columnas categóricas (predictoras y target)
    encoders = {}       # col -> mapping value->int
    inverse_maps = {}   # col -> inverse_map int->value
    categorical_info = {}  # col -> inverse_map (None if numeric)

    X = pd.DataFrame(index=data.index)
    for col in columnas_usar:
        if data[col].dtype == 'object' or str(data[col].dtype).startswith("category"):
            mapping, inverse = _make_mapping(data[col])
            encoders[col] = mapping
            inverse_maps[col] = inverse
            categorical_info[col] = inverse
            X[col] = _encode_series(data[col], mapping)
        else:
            # Numérico
            categorical_info[col] = None
            X[col] = data[col].fillna(data[col].mean())  # ya imputamos, pero por seguridad
    # Preparar y codificar y (target)
    y_series = data[columna_objetivo]
    target_is_categorical = False
    if y_series.dtype == 'object' or str(y_series.dtype).startswith("category"):
        target_is_categorical = True
        tmap, tinv = _make_mapping(y_series)
        encoders[columna_objetivo] = tmap
        inverse_maps[columna_objetivo] = tinv
        y = _encode_series(y_series, tmap)
    else:
        # numérico: dejar como está
        y = y_series.copy()
        # convertir NaN a np.nan (ya están así)

    # Entrenamiento con filas donde y no es NaN
    train_mask = ~y.isna()
    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    # Revisiones: si no hay suficientes filas para entrenar
    if X_train.shape[0] == 0:
        raise Exception("No hay filas con valor objetivo para entrenar.")
    # Convertir y_train a entero si target categórico
    if target_is_categorical:
        y_train = y_train.astype(int)

    # Entrenar árbol
    modelo = DecisionTreeClassifier(max_depth=4, random_state=0)
    modelo.fit(X_train, y_train)

    # Export raw tree (texto)
    arbol_raw = export_text(modelo, feature_names=list(X.columns))

    # Generar reglas legibles (utilizando inverse maps para categóricas)
    reglas = generar_reglas_legibles(modelo, list(X.columns), categorical_info)

    # Predicción de filas faltantes (si existían)
    valores_rellenados = None
    if not filas_faltantes.empty:
        # Codificamos las filas a predecir con los mismos encoders
        filas_cod = pd.DataFrame(index=filas_faltantes.index)
        for col in columnas_usar:
            if col in encoders and categorical_info[col] is not None:
                filas_cod[col] = _encode_series(filas_faltantes[col], encoders[col])
            else:
                filas_cod[col] = filas_faltantes[col].fillna(X[col].mean())

        # Predecir
        pred = modelo.predict(filas_cod)

        # Si el target era categórico, invertir codificación
        if target_is_categorical:
            inv = inverse_maps[columna_objetivo]
            pred_decoded = [_decode_value(p, inv) for p in pred]
        else:
            # numérico: dejar tal cual
            pred_decoded = pred.tolist()

        # Construir DataFrame con las filas rellenadas en su forma original
        df_rellenadas = filas_faltantes.copy()
        df_rellenadas[columna_objetivo] = pred_decoded
        valores_rellenados = df_rellenadas

    return {
        "arbol": arbol_raw,
        "reglas": reglas,
        "columnas_usadas": columnas_usar,
        "valores_rellenados": valores_rellenados
    }
