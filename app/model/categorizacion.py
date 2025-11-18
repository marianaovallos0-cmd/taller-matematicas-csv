import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split

def entrenar_arbol_decision(df, nombre_objetivo, tipo):
    df = df.copy()

    columnas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()

   
    columnas_numericas_imputadas = []
    columnas_categoricas_imputadas = []

    for col in columnas_numericas:
        df[col] = df[col].fillna(df[col].mean())
        columnas_numericas_imputadas.append(col)

    for col in columnas_categoricas:
        df[col] = df[col].fillna(df[col].mode()[0])
        columnas_categoricas_imputadas.append(col)

  
    label_encoders = {}
    columnas_codificadas = []

    for col in columnas_categoricas:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        columnas_codificadas.append(col)


    X = df.drop(columns=[nombre_objetivo])
    y = df[nombre_objetivo]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    if tipo == "clasificacion":
        modelo = DecisionTreeClassifier(random_state=0)
    else:
        modelo = DecisionTreeRegressor(random_state=0)

    modelo.fit(X_train, y_train)

    reglas = export_text(modelo, feature_names=list(X.columns))

   
    df_resultado = df.copy()

    if tipo == "regresion":
        df_resultado["Predicción"] = modelo.predict(X)

  
    info_preprocesamiento = {
        "numericos": columnas_numericas_imputadas,
        "categoricos": columnas_categoricas_imputadas,
        "codificados": columnas_codificadas
    }

    precision = None

    return precision, modelo, reglas, df_resultado, info_preprocesamiento
