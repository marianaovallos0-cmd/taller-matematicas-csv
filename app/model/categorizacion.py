import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split

def entrenar_arbol_decision(df, columna_objetivo):
    # 1. Copia para no modificar el original
    data = df.copy()

    # 2. Convertir todas las variables categóricas a números
    label_encoders = {}
    for col in data.columns:
        if data[col].dtype == "object":
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

    # 3. Separar X e y
    X = data.drop(columns=[columna_objetivo])
    y = data[columna_objetivo]

    # 4. Seleccionar árbol de clasificación o regresión
    if y.dtype in ["int64", "float64"]:
        modelo = DecisionTreeRegressor(random_state=0)
    else:
        modelo = DecisionTreeClassifier(random_state=0)

    # 5. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # 6. Entrenar
    modelo.fit(X_train, y_train)

    # 7. Generar árbol como texto
    arbol_texto = export_text(modelo, feature_names=list(X.columns))

    # 8. Devolver el árbol en texto + predicciones de prueba
    resultados = {
        "arbol": arbol_texto,
        "score": modelo.score(X_test, y_test)
    }

    return resultados
