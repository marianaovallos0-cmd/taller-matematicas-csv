import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
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
            # Hoja → obtener clase objetivo
            valor = modelo.classes_[tree_.value[nodo].argmax()] \
                    if hasattr(modelo, "classes_") else tree_.value[nodo][0][0]

            if target_encoder is not None:
                valor = target_encoder.inverse_transform([int(valor)])[0]

            regla = "Si " + " y ".join(condiciones) + f", entonces {valor}"
            reglas.append(regla)

    recorrer_nodo(0, [])
    return "\n".join(reglas)



def entrenar_arbol_decision(df, columna_objetivo, columnas_usar):

    # --- Filtrar SOLO las columnas indicadas ---
    data = df[columnas_usar + [columna_objetivo]].copy()

    # --- Label Encoding ---
    encoders = {}
    for col in data.columns:
        if data[col].dtype == "object":
            enc = LabelEncoder()
            data[col] = enc.fit_transform(data[col].astype(str))
            encoders[col] = enc

    X = data[columnas_usar]
    y = data[columna_objetivo]

    # --- Elegir modelo ---
    # Si el objetivo fue codificado → es categórico → clasificador
    if columna_objetivo in encoders:
        modelo = DecisionTreeClassifier(max_depth=4, random_state=0)
    else:
        modelo = DecisionTreeRegressor(max_depth=4, random_state=0)

    # --- Entrenar ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    modelo.fit(X_train, y_train)

    # Árbol raw (estético)
    arbol_raw = export_text(modelo, feature_names=list(X.columns))

    # Reglas legibles reales
    reglas = generar_reglas_legibles(
        modelo,
        list(X.columns),
        target_encoder=encoders.get(columna_objetivo)
    )

    return {
        "arbol": arbol_raw,
        "reglas": reglas,
        "columnas_usadas": columnas_usar
    }
