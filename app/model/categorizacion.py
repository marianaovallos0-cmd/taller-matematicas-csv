# categorizacion.py
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree as _sk_tree
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
    """Decodifica tratando -1 como None (sin valor)."""
    if val is None:
        return None
    try:
        iv = int(val)
    except Exception:
        return str(val)
    if iv == -1:
        return None
    return inverse_map.get(iv, str(iv))

# -----------------------
# Generación de reglas legibles
# -----------------------
def _codes_leq_threshold(threshold):
    """Dado threshold float devuelve el upper code incluido (floor)."""
    return int(np.floor(float(threshold)))

def _format_category_set(codes, inverse_map):
    """Formato legible para un conjunto de códigos: si 1 elemento -> 'col = X', si varios -> 'col in {A,B}'"""
    cats = [inverse_map[c] for c in sorted(codes) if c in inverse_map]
    if len(cats) == 0:
        return None
    if len(cats) == 1:
        return cats[0]
    return "{" + ", ".join(map(str, cats)) + "}"

def _generate_rules(modelo, feature_names, categorical_inverse_maps, target_inverse_map, target_is_categorical):
    """
    Recorre el árbol y genera reglas legibles.
    - categorical_inverse_maps: dict feature -> inverse_map (or None)
    - target_inverse_map: map int->value if target categorical else None
    """
    tree_ = modelo.tree_
    rules = []

    def recorrer(nodo, condiciones):
        # Nodo interno
        if tree_.feature[nodo] != _sk_tree._tree.TREE_UNDEFINED:
            feat_idx = tree_.feature[nodo]
            feat_name = feature_names[feat_idx]
            thr = tree_.threshold[nodo]
            is_cat = feat_name in categorical_inverse_maps and categorical_inverse_maps[feat_name] is not None

            # Izquierda: <= threshold
            if is_cat:
                upper = _codes_leq_threshold(thr)
                left_codes = set([c for c in categorical_inverse_maps[feat_name].keys() if c <= upper])
                left_text = _format_category_set(left_codes, categorical_inverse_maps[feat_name])
                if left_text is not None:
                   cond_left = condiciones + [f"{feat_name} = {left_text}" if not left_text.startswith("{") else f"{feat_name} in {left_text}"]
                else:
                    cond_left = condiciones + [f"{feat_name} <= {thr:.2f}"]
            else:
                cond_left = condiciones + [f"{feat_name} <= {thr:.2f}"]

            recorrer(tree_.children_left[nodo], cond_left)

            # Derecha: > threshold
            if is_cat:
                upper = _codes_leq_threshold(thr)
                all_codes = set(categorical_inverse_maps[feat_name].keys())
                right_codes = sorted(all_codes - set([c for c in categorical_inverse_maps[feat_name].keys() if c <= upper]))
                right_text = _format_category_set(right_codes, categorical_inverse_maps[feat_name])
                if right_text is not None:
                    cond_right = condiciones + [f"{feat_name} = {right_text}" if not right_text.startswith("{") else f"{feat_name} in {right_text}"]
                else:
                    cond_right = condiciones + [f"{feat_name} > {thr:.2f}"]
            else:
                cond_right = condiciones + [f"{feat_name} > {thr:.2f}"]

            recorrer(tree_.children_right[nodo], cond_right)

        else:
            # Hoja: tomar clase con mayor soporte
            vals = tree_.value[nodo][0]
            class_idx = int(np.argmax(vals))
            if target_is_categorical and target_inverse_map is not None:
                predicted = target_inverse_map.get(class_idx, str(class_idx))
            else:
                # Si no es categórico, sklearn almacena clases_ igual; extraemos clase
                try:
                    predicted = modelo.classes_[class_idx]
                except Exception:
                    predicted = str(class_idx)
            cond_text = " y ".join(condiciones) if condiciones else "(sin condiciones)"
            rules.append(f"Si {cond_text}, entonces {predicted}")

    recorrer(0, [])
    return "\n".join(rules)

# -----------------------
# Entrenamiento / Función pública
# -----------------------
def entrenar_arbol_decision(df, columna_objetivo, columnas_usar, max_depth=4, random_state=0):
    """
    Entrena árbol, genera reglas legibles y rellena filas con target faltante.
    Retorna dict: arbol (raw), reglas (string), columnas_usadas, valores_rellenados (DataFrame o None)
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

    # Filas donde objetivo está vacío (para predecir luego)
    filas_faltantes_idx = data[data[columna_objetivo].isna()].index.tolist()
    filas_faltantes = data.loc[filas_faltantes_idx].copy() if len(filas_faltantes_idx) > 0 else pd.DataFrame()

    # Preparar encoders (mapping dicts) e inverse maps
    encoders = {}       # col -> mapping value->int
    inverse_maps = {}   # col -> inverse_map int->value
    categorical_inverse_maps = {}  # feature -> inverse_map or None

    X = pd.DataFrame(index=data.index)
    for col in columnas_usar:
        if data[col].dtype == 'object' or str(data[col].dtype).startswith("category"):
            mapping, inverse = _make_mapping(data[col])
            encoders[col] = mapping
            inverse_maps[col] = inverse
            categorical_inverse_maps[col] = inverse
            X[col] = _encode_series(data[col], mapping)
        else:
            categorical_inverse_maps[col] = None
            X[col] = data[col].astype(float).fillna(data[col].mean())

    # Preparar target (y)
    y_series = data[columna_objetivo]
    target_is_categorical = False
    target_inverse_map = None
    if y_series.dtype == 'object' or str(y_series.dtype).startswith("category"):
        target_is_categorical = True
        tmap, tinv = _make_mapping(y_series)
        encoders[columna_objetivo] = tmap
        inverse_maps[columna_objetivo] = tinv
        target_inverse_map = tinv
        y = _encode_series(y_series, tmap)
    else:
        y = y_series.copy()

    # Entrenar solo con filas donde y no es NaN
    train_mask = ~y.isna()
    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    if X_train.shape[0] == 0:
        raise Exception("No hay filas con valor objetivo para entrenar.")

    if target_is_categorical:
        y_train = y_train.astype(int)

    modelo = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    modelo.fit(X_train, y_train)

    arbol_raw = export_text(modelo, feature_names=list(X.columns))

    # Generar reglas legibles
    reglas = _generate_rules(modelo, list(X.columns), categorical_inverse_maps, target_inverse_map, target_is_categorical)

    # Predecir filas faltantes del objetivo (si las hubiera)
    valores_rellenados = None
    if not filas_faltantes.empty:
        filas_cod = pd.DataFrame(index=filas_faltantes.index)
        for col in columnas_usar:
            if col in encoders and categorical_inverse_maps.get(col) is not None:
                filas_cod[col] = _encode_series(filas_faltantes[col], encoders[col])
            else:
                filas_cod[col] = filas_faltantes[col].astype(float).fillna(X[col].mean())

        pred = modelo.predict(filas_cod)

        if target_is_categorical and target_inverse_map is not None:
            pred_decoded = [_decode_value(p, target_inverse_map) for p in pred]
        else:
            # Si no es categórico, devolver como vienen (pueden ser ints)
            pred_decoded = pred.tolist()

        df_rellenadas = filas_faltantes.copy()
        df_rellenadas[columna_objetivo] = pred_decoded
        valores_rellenados = df_rellenadas

    return {
        "arbol": arbol_raw,
        "reglas": reglas,
        "columnas_usadas": columnas_usar,
        "valores_rellenados": valores_rellenados
    }