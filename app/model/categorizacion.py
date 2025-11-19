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

def generar_reglas_legibles(modelo, feature_names, categorical_info):
    """
    categorical_info: dict feature_name -> inverse_map (o None si numérica)
    Retorna cadena con reglas legibles.
    """
    from sklearn.tree import _tree
    tree_ = modelo.tree_
    reglas = []

    def recorrer(nodo, condiciones):
        if tree_.feature[nodo] != _tree.TREE_UNDEFINED:
            idx = tree_.feature[nodo]
            feature_name = feature_names[idx]
            threshold = tree_.threshold[nodo]
            is_cat = feature_name in categorical_info and categorical_info[feature_name] is not None

            # rama izquierda: <= threshold
            cond_izq = condiciones + [_nice_condition_for_split(feature_name, threshold, is_cat, categorical_info.get(feature_name))]
            recorrer(tree_.children_left[nodo], cond_izq)

            # rama derecha: > threshold
            # Para derecha convertimos la condición complementaria:
            if is_cat and categorical_info.get(feature_name) is not None:
                # si la izquierda era "feature in {A,B}", la derecha es "feature in {rest}"
                inv = categorical_info[feature_name]
                t = float(threshold)
                upper = int(np.floor(t))
                left_codes = set(range(0, upper + 1))
                all_codes = set(inv.keys())
                right_codes = sorted(all_codes - left_codes)
                if len(right_codes) == 0:
                    cond_der = [f"{feature_name} > {threshold:.2f}"]
                elif len(right_codes) == 1:
                    cond_der = [f"{feature_name} = {inv[right_codes[0]]}"]
                else:
                    cats = [inv[i] for i in right_codes]
                    cond_der = [f"{feature_name} in {{{', '.join(map(str, cats))}}}"]
            else:
                cond_der = condiciones + [f"{feature_name} > {threshold:.2f}"]

            if is_cat and categorical_info.get(feature_name) is not None:
                recorrer(tree_.children_right[nodo], condiciones + cond_der)
            else:
                recorrer(tree_.children_right[nodo], cond_der)

        else:
            # Nodo hoja: tomar la clase con mayor conteo
            values = tree_.value[nodo][0]
            class_index = int(np.argmax(values))
            predicted = modelo.classes_[class_index]
            regla = "Si " + " y ".join(condiciones) + f", entonces {predicted}"
            reglas.append(regla)

    recorrer(0, [])
    return "\n".join(reglas)

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
