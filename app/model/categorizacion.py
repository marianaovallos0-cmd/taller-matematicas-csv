import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def entrenar_arbol_decision(df, nombre_objetivo):

    if nombre_objetivo not in df.columns:
        raise ValueError(f"La columna objetivo '{nombre_objetivo}' no existe en el dataset.")

    y = df[nombre_objetivo]
    X = df.drop(columns=[nombre_objetivo])

    if y.isna().any():
        modo = y.mode()
        if len(modo) > 0:
            y = y.fillna(modo.iloc[0])
        else:
            # Si no hay moda se rellena con 0
            y = y.fillna(0)

    X = X.fillna(0)

    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))
        model = DecisionTreeClassifier()
    else:
        model = DecisionTreeRegressor()


    if X.shape[1] == 0:
        raise ValueError("No hay columnas suficientes para entrenar el árbol de decisión.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    except ValueError as e:
        raise ValueError("No se puede dividir el dataset. Verifica NaN o columnas vacías.\n" + str(e))

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        raise ValueError("Error entrenando el árbol de decisión:\n" + str(e))

    try:
        precision = model.score(X_test, y_test)
    except Exception:
        precision = 0


    try:
        reglas = export_text(model, feature_names=list(X.columns))
    except Exception:
        reglas = "No se pudieron generar reglas."

    return precision, model, reglas
