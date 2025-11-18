import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def entrenar_arbol_decision(df, nombre_objetivo):
    """
    Entrena un árbol de decisión y predice valores faltantes en la columna objetivo
    """
    if nombre_objetivo not in df.columns:
        raise ValueError(f"La columna objetivo '{nombre_objetivo}' no existe en el dataset.")

    # Hacer una copia para no modificar el original
    df_work = df.copy()
    
    # Excluir columnas que probablemente son IDs (como "Código", "Profesor")
    columnas_a_excluir = []
    for columna in df_work.columns:
        if columna == nombre_objetivo:
            continue
        
        # Si la columna tiene muchos valores únicos, probablemente es un ID
        if df_work[columna].nunique() / len(df_work) > 0.8:
            columnas_a_excluir.append(columna)
    
    if columnas_a_excluir:
        df_work = df_work.drop(columns=columnas_a_excluir)

    # Separar datos para entrenamiento y predicción
    df_entrenamiento = df_work[df_work[nombre_objetivo].notna()]
    df_prediccion = df_work[df_work[nombre_objetivo].isna()]

    # Si no hay datos para entrenar, no podemos predecir
    if len(df_entrenamiento) == 0:
        raise ValueError(f"No hay valores conocidos en '{nombre_objetivo}' para entrenar el modelo.")

    # Preparar datos de entrenamiento
    y_entrenamiento = df_entrenamiento[nombre_objetivo]
    X_entrenamiento = df_entrenamiento.drop(columns=[nombre_objetivo])

    # Si hay datos para predecir, prepararlos
    if len(df_prediccion) > 0:
        X_prediccion = df_prediccion.drop(columns=[nombre_objetivo])
    else:
        X_prediccion = pd.DataFrame()

    # Limpieza de datos
    X_entrenamiento = X_entrenamiento.fillna('Desconocido')
    if len(X_prediccion) > 0:
        X_prediccion = X_prediccion.fillna('Desconocido')

    # Codificar variables categóricas
    label_encoders = {}
    for col in X_entrenamiento.columns:
        if X_entrenamiento[col].dtype == "object":
            le = LabelEncoder()
            X_entrenamiento[col] = le.fit_transform(X_entrenamiento[col].astype(str))
            if len(X_prediccion) > 0:
                # Manejar valores no vistos en predicción
                try:
                    X_prediccion[col] = le.transform(X_prediccion[col].astype(str))
                except:
                    # Si hay valores nuevos, usar el valor más común
                    X_prediccion[col] = 0
            label_encoders[col] = le

    # Codificar target y crear modelo
    if y_entrenamiento.dtype == "object":
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y_entrenamiento.astype(str))
        model = DecisionTreeClassifier(random_state=42)
        es_clasificacion = True
    else:
        y_encoded = y_entrenamiento
        model = DecisionTreeRegressor(random_state=42)
        es_clasificacion = False

    # Verificar que hay datos suficientes
    if X_entrenamiento.shape[1] == 0:
        raise ValueError("No hay columnas suficientes para entrenar el árbol de decisión.")

    # Entrenar modelo
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_entrenamiento, y_encoded, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        precision = model.score(X_test, y_test)
    except Exception:
        # Si falla la división, entrenar con todos los datos
        model.fit(X_entrenamiento, y_encoded)
        precision = model.score(X_entrenamiento, y_encoded)

    # Generar reglas en texto
    try:
        reglas = export_text(model, feature_names=list(X_entrenamiento.columns))
    except Exception:
        reglas = "No se pudieron generar reglas."

    # Predecir valores faltantes
    df_completo = df.copy()
    if len(df_prediccion) > 0:
        try:
            predicciones = model.predict(X_prediccion)
            if es_clasificacion:
                predicciones = le_target.inverse_transform(predicciones)
            
            # Actualizar el DataFrame original con las predicciones
            indices_prediccion = df_prediccion.index
            df_completo.loc[indices_prediccion, nombre_objetivo] = predicciones
        except Exception as e:
            raise ValueError(f"Error prediciendo valores: {e}")

    return precision, model, reglas, df_completo