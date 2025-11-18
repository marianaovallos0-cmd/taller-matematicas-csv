import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split


def generar_reglas_legibles(arbol_texto, encoders, columna_objetivo, encoder_objetivo):
    reglas = []
    regla_actual = ""

    for linea in arbol_texto.split("\n"):
        l = linea.strip()
        if l == "":
            continue

        # --- Nodo de condición ---
        if "<=" in l or ">" in l:
            partes = l.split()
            col = partes[0]
            op = partes[1]
            val = partes[2]

            # Decodificar categorías
            if col in encoders:
                val = encoders[col].inverse_transform([int(float(val))])[0]

            regla_actual = f"Si {col} {op} {val}"

        # --- Nodo hoja / salida ---
        if "value:" in l:
            # extraer clase predicha
            clase_raw = l.split("[")[1].split("]")[0]

            # decodificar clase objetivo si es categórica
            try:
                clase = encoder_objetivo.inverse_transform([int(float(clase_raw))])[0]
            except:
                clase = clase_raw

            reglas.append(f"{regla_actual} → {columna_objetivo} = {clase}")

    return "\n".join(reglas)


def entrenar_arbol_decision(df, columna_objetivo, columnas_usar):

    # --- Filtrar SOLO las columnas indicadas ---
    data = df[columnas_usar + [columna_objetivo]].copy()

    # --- Codificación ---
    encoders = {}
    for col in data.columns:
        if data[col].dtype == "object":
            enc = LabelEncoder()
            data[col] = enc.fit_transform(data[col].astype(str))
            encoders[col] = enc

    X = data[columnas_usar]
    y = data[columna_objetivo]

    # Guardar encoder del objetivo para reglas legibles
    encoder_objetivo = encoders.get(columna_objetivo, None)

    # --- Elegir modelo ---
    if y.dtype in ("int64", "float64"):
        modelo = DecisionTreeRegressor(max_depth=4, random_state=0)
    else:
        modelo = DecisionTreeClassifier(max_depth=4, random_state=0)

    # --- Entrenar ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    modelo.fit(X_train, y_train)

    # --- Árbol en texto ---
    arbol_raw = export_text(modelo, feature_names=list(X.columns))

    # --- Reglas legibles ---
    reglas = generar_reglas_legibles(
        arbol_raw,
        encoders,
        columna_objetivo,
        encoder_objetivo
    )

    return {
        "arbol": arbol_raw,
        "reglas": reglas,
        "score": modelo.score(X_test, y_test),
        "columnas_usadas": columnas_usar
    }
