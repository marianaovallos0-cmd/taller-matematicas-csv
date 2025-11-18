import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split

def entrenar_arbol_decision(df, columna_objetivo):

    data = df.copy()
    columnas_eliminar = []

    # --- 1. Identificar columnas tipo ID automáticamente ---
    for col in data.columns:
        if col == columna_objetivo:
            continue
        
        # Solo columnas numéricas pueden ser ID
        if pd.api.types.is_numeric_dtype(data[col]):
            # % de valores únicos
            propor_unique = data[col].nunique() / len(data)

            if propor_unique > 0.8:
                columnas_eliminar.append(col)

    # Eliminar columnas detectadas como ID
    data = data.drop(columns=columnas_eliminar)

    # --- 2. Codificar categóricas ---
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    # --- 3. Separar X e y ---
    X = data.drop(columns=[columna_objetivo])
    y = data[columna_objetivo]

    # --- 4. Modelo pequeño, árbol legible ---
    if y.dtype in ["int64", "float64"]:
        modelo = DecisionTreeRegressor(max_depth=3, min_samples_split=4, random_state=0)
    else:
        modelo = DecisionTreeClassifier(max_depth=3, min_samples_split=4, random_state=0)

    # --- 5. Entrenar ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    modelo.fit(X_train, y_train)

    # --- 6. Generar árbol corto ---
    arbol_texto = export_text(
        modelo,
        feature_names=list(X.columns)
    )

    return {
        "arbol": arbol_texto,
        "score": modelo.score(X_test, y_test),
        "columnas_usadas": list(X.columns),
        "columnas_eliminadas": columnas_eliminar
    }
