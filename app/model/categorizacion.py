import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split


def generar_reglas_legibles(modelo, feature_names, target_encoder=None):
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
            valor = modelo.classes_[tree_.value[nodo].argmax()] \
                if hasattr(modelo, "classes_") else tree_.value[nodo][0][0]

            if target_encoder is not None:
                valor = target_encoder.inverse_transform([int(valor)])[0]

            regla = "Si " + " y ".join(condiciones) + f", entonces {valor}"
            reglas.append(regla)

    recorrer_nodo(0, [])
    return "\n".join(reglas)



def entrenar_arbol_decision(df, columna_objetivo, columnas_usar):

    # --- Verificar que SOLO la columna objetivo tenga faltantes ---
    columnas_predictoras = columnas_usar
    predictors_with_nan = df[columnas_predictoras].isna().any()

    if predictors_with_nan.any():
        raise Exception(
            "Faltan datos para poder predecir o categorizar. "
            "Primero completa los valores faltantes en las columnas predictoras."
        )

    # --- Filtrar SOLO las columnas indicadas ---
    data = df[columnas_usar + [columna_objetivo]].copy()

    # --- Guardar filas con NaN en objetivo ---
    filas_faltantes = data[data[columna_objetivo].isna()].copy()

    # --- Label Encoding ---
    encoders = {}
    for col in data.columns:
        if data[col].dtype == "object":
            enc = LabelEncoder()
            no_nulos = data[col].dropna().astype(str)
            enc.fit(no_nulos)

            data[col] = data[col].apply(
                lambda v: enc.transform([str(v)])[0] if pd.notna(v) else None
            )
            encoders[col] = enc

    X = data[columnas_usar]
    y = data[columna_objetivo]

    # --- Separar filas completas para entrenar ---
    X_train = X[~y.isna()]
    y_train = y[~y.isna()]

    # --- Modelo ---
    modelo = DecisionTreeClassifier(max_depth=4, random_state=0)

    # --- Entrenar ---
    modelo.fit(X_train, y_train)

    # --- Árbol raw ---
    arbol_raw = export_text(modelo, feature_names=list(X.columns))

    # --- Reglas legibles ---
    reglas = generar_reglas_legibles(
        modelo,
        list(X.columns),
        target_encoder=encoders.get(columna_objetivo)
    )

    # --- Predecir valores faltantes del objetivo ---
    valores_rellenados = None
    if len(filas_faltantes) > 0:
        filas_cod = filas_faltantes[columnas_usar].copy()

        # codificar predictoras si es necesario
        for col in filas_cod.columns:
            if col in encoders:
                filas_cod[col] = encoders[col].transform(filas_cod[col].astype(str))

        pred = modelo.predict(filas_cod)

        if columna_objetivo in encoders:
            pred = encoders[columna_objetivo].inverse_transform(pred)

        filas_faltantes[columna_objetivo] = pred
        valores_rellenados = filas_faltantes

    return {
        "arbol": arbol_raw,
        "reglas": reglas,
        "columnas_usadas": columnas_usar,
        "valores_rellenados": valores_rellenados
    }
