import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split


def generar_reglas_legibles(arbol_texto, encoders):
    reglas = []
    for linea in arbol_texto.split("\n"):
        l = linea.strip()
        if l == "" or l.startswith("class"):
            continue
        if "<=" in l or ">" in l:
            col, op, val = l.split()[:3]

            if col in encoders:
                val = encoders[col].inverse_transform([int(float(val))])[0]

            reglas.append(f"Si {col} {op} {val}")

        if "value" in l:
            reglas[-1] += f" → {l}"
    return "\n".join(reglas)


def entrenar_arbol_decision(df, columna_objetivo, columnas_usar):

    data = df[columnas_usar + [columna_objetivo]].copy()

    encoders = {}
    for col in data.columns:
        if data[col].dtype == "object":
            enc = LabelEncoder()
            data[col] = enc.fit_transform(data[col].astype(str))
            encoders[col] = enc

    X = data[columnas_usar]
    y = data[columna_objetivo]

    if y.dtype in ["int64", "float64"]:
        modelo = DecisionTreeRegressor(max_depth=4, random_state=0)
    else:
        modelo = DecisionTreeClassifier(max_depth=4, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    modelo.fit(X_train, y_train)

    arbol_raw = export_text(modelo, feature_names=list(X.columns))
    reglas = generar_reglas_legibles(arbol_raw, encoders)

    return {
        "arbol": arbol_raw,
        "reglas": reglas,
        "score": modelo.score(X_test, y_test),
        "columnas_usadas": columnas_usar
    }
