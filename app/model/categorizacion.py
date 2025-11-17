import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def entrenar_arbol_decision(df, nombre_objetivo):
    # Separamos X e y
    y = df[nombre_objetivo]
    X = df.drop(columns=[nombre_objetivo])

    # Codificar columnas categóricas
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # Determinar si es regresión o clasificación
    if y.dtype == "object":
        y_encoded = LabelEncoder().fit_transform(y.astype(str))
        model = DecisionTreeClassifier()
    else:
        y_encoded = y
        model = DecisionTreeRegressor()

    # División datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Entrenar
    model.fit(X_train, y_train)

    # Calcular precisión
    precision = model.score(X_test, y_test)

    # Exportar reglas
    reglas = export_text(model, feature_names=list(X.columns))

    return precision, model, reglas
