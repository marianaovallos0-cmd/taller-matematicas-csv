import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split


def generar_reglas_legibles(arbol_texto, decodificadores):
    reglas = []
    lineas = arbol_texto.split("\n")

    # Convertir cada línea en una regla clara
    for linea in lineas:
        l = linea.strip()
        if "class:" in l:
            continue
        if "<=" in l or ">" in l:
            partes = l.split()
            col = partes[0]
            operador = partes[1]
            valor = partes[2]

            if col in decodificadores:
                valor = decodificadores[col].inverse_transform([int(float(valor))])[0]

            reglas.append(f"{col} {operador} {valor}")

    return "\n".join(reglas)


def entrenar_arbol_decision(df, columna_objetivo, columnas_usar):

    data = df.copy()

    # Reducir solo a las columnas deseadas + objetivo
    data = data[columnas_usar + [columna_objetivo]]

    # ---- Codificar categóricas con decodificación guardada ----
    encoders = {}
    for col in data.columns:
        if data[col].dtype == "object":
            enc = LabelEncoder()
            data[col] = enc.fit_transform(data[col].astype(str))
            encoders[col] = enc

    # ---- Separar X e y ----
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

    # ---- Estructura del árbol ----
    arbol_raw = export_text(modelo, feature_names=list(X.columns))

    # ---- Generar reglas legibles ----
    reglas = generar_reglas_legibles(arbol_raw, encoders)

    return {
        "arbol": arbol_raw,
        "reglas": reglas,
        "score": modelo.score(X_test, y_test),
        "columnas_usadas": columnas_usar
    }
