import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split

def entrenar_arbol_decision(df, nombre_objetivo, tipo):
    df_original = df.copy()     # Guardamos versión con palabras para mostrarla al final
    df = df.copy()

    columnas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- 1. Imputación ---
    for col in columnas_numericas:
        df[col] = df[col].fillna(df[col].mean())

    for col in columnas_categoricas:
        df[col] = df[col].fillna(df[col].mode()[0])

    # --- 2. Codificación ---
    label_encoders = {}

    for col in columnas_categoricas:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # --- 3. Dividir datos ---
    X = df.drop(columns=[nombre_objetivo])
    y = df[nombre_objetivo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # --- 4. Entrenar árbol ---
    if tipo == "clasificacion":
        modelo = DecisionTreeClassifier(random_state=0)
    else:
        modelo = DecisionTreeRegressor(random_state=0)

    modelo.fit(X_train, y_train)

    # --- 5. Reglas ---
    reglas = export_text(modelo, feature_names=list(X.columns))

    # --- 6. Construir tabla FINAL con palabras ---
    df_resultado = df_original.copy()   # regresamos a palabras

    # Si es regresión, agregamos columna de predicción
    if tipo == "regresion":
        df_resultado["Predicción"] = modelo.predict(X)

    precision = None  # en clasificación no estás pidiendo accuracy

    return precision, modelo, reglas, df_resultado, {}
