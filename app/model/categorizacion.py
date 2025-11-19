import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np
from model import valoresFaltantes

def generar_reglas_legibles(modelo, feature_names, target_inverse_map=None):
    from sklearn.tree import _tree
    tree_ = modelo.tree_
    reglas = []

    def recorrer_nodo(nodo, condiciones):
        if tree_.feature[nodo] != _tree.TREE_UNDEFINED:
            col = feature_names[tree_.feature[nodo]]
            threshold = tree_.threshold[nodo]

            recorrer_nodo(
                tree_.children_left[nodo],
                condiciones + [f"{col} <= {threshold:.2f}"]
            )
            recorrer_nodo(
                tree_.children_right[nodo],
                condiciones + [f"{col} > {threshold:.2f}"]
            )

        else:
            # Obtener clase predicha (índice)
            class_index = int(np.argmax(tree_.value[nodo]))
            valor = modelo.classes_[class_index]
            # Si tenemos mapping para invertir la codificación del target, aplicarlo
            if target_inverse_map is not None:
                valor_str = target_inverse_map.get(int(valor), str(valor))
            else:
                valor_str = str(valor)
            regla = "Si " + " y ".join(condiciones) + f", entonces {valor_str}"
            reglas.append(regla)

    recorrer_nodo(0, [])
    return "\n".join(reglas)

def _fit_simple_int_encoding(series):
    """
    Crea un mapeo de valor -> entero y el inverso.
    (Más transparente que LabelEncoder y maneja NaN sin romper.)
    """
    uniques = list(series.dropna().astype(str).unique())
    mapping = {v: i for i, v in enumerate(uniques)}
    inverse = {i: v for v, i in mapping.items()}
    return mapping, inverse

def entrenar_arbol_decision(df, columna_objetivo, columnas_usar):
    # Validaciones simples
    if columna_objetivo not in df.columns:
        raise Exception("La columna objetivo no existe en el DataFrame.")
    for c in columnas_usar:
        if c not in df.columns:
            raise Exception(f"La columna predictora '{c}' no existe en el DataFrame.")

    # Si hay NaN en las columnas predictoras, aplicamos imputación automática (mix)
    predictors_with_nan = df[columnas_usar].isna().any()
    if predictors_with_nan.any():
        # aplicamos una imputación automática y documentada (KNN para num y k-modas para categóricas)
        df = valoresFaltantes.imputar_mix(df)

    # Trabajamos sobre copia
    data = df[columnas_usar + [columna_objetivo]].copy()

    # Guardar filas donde el objetivo está vacío (para luego rellenarlas)
    filas_faltantes = data[data[columna_objetivo].isna()].copy()

    # Encoding simple y reproducible usando dicts
    encoders = {}
    inverse_maps = {}
    for col in data.columns:
        if data[col].dtype == "object" or data[col].dtype.name == "category":
            mapping, inverse = _fit_simple_int_encoding(data[col])
            encoders[col] = mapping
            inverse_maps[col] = inverse
            # transformar (NaN -> -1)
            data[col] = data[col].apply(lambda v: mapping.get(str(v), -1) if pd.notna(v) else -1)
        else:
            # si es numérica, dejamos tal cual y no la codificamos
            pass

    # Separar X e y
    X = data[columnas_usar]
    y = data[columna_objetivo]

    # Definir mask de entrenamiento: si target es numérico y no usamos -1, preservamos filas no-nulas
    mask_train = ~y.isna()
    X_train = X[mask_train]
    y_train = y[mask_train]

    # Si target es categórico codificado (mapeo), transformar y_train
    if columna_objetivo in encoders:
        y_train = y_train.apply(lambda v: encoders[columna_objetivo].get(str(v), -1) if pd.notna(v) else -1)

    # Modelo (mantengo simple y explicable)
    modelo = DecisionTreeClassifier(max_depth=4, random_state=0)
    modelo.fit(X_train, y_train)

    # Árbol raw
    arbol_raw = export_text(modelo, feature_names=list(X.columns))

    # Preparamos target_inverse_map para reglas legibles (si existe)
    target_inverse_map = inverse_maps.get(columna_objetivo) if columna_objetivo in inverse_maps else None

    reglas = generar_reglas_legibles(
        modelo,
        list(X.columns),
        target_inverse_map=target_inverse_map
    )

    # Predecir valores faltantes del objetivo (si los hay)
    valores_rellenados = None
    if len(filas_faltantes) > 0:
        filas_cod = filas_faltantes[columnas_usar].copy()
        # transformar columnas categóricas en filas_cod con los mismos mapeos
        for col in filas_cod.columns:
            if col in encoders:
                filas_cod[col] = filas_cod[col].apply(lambda v: encoders[col].get(str(v), -1) if pd.notna(v) else -1)

        pred = modelo.predict(filas_cod)

        # invertir codificación si target era categórico
        if columna_objetivo in inverse_maps:
            inv = inverse_maps[columna_objetivo]
            pred = [inv.get(int(p), str(p)) for p in pred]

        filas_faltantes[columna_objetivo] = pred
        valores_rellenados = filas_faltantes

    return {
        "arbol": arbol_raw,
        "reglas": reglas,
        "columnas_usadas": columnas_usar,
        "valores_rellenados": valores_rellenados
    }
