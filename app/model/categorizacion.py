import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def entrenar_arbol_decision(df: pd.DataFrame, nombre_objetivo: str):
    """
    Entrena un Árbol de Decisión (Clasificación o Regresión) con el objetivo dado.
    Si el objetivo tiene valores faltantes (NaN), los imputa con las predicciones
    del modelo entrenado con los datos conocidos.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        nombre_objetivo (str): Nombre de la columna objetivo a predecir/categorizar.
        
    Returns:
        tuple: (precision, model, reglas, df_completo, hubo_prediccion)
    """
    if nombre_objetivo not in df.columns:
        raise ValueError(f"La columna objetivo '{nombre_objetivo}' no existe en el dataset.")

    df_work = df.copy()
    
    # --- Definir la estrategia de imputación de características (Media vs. Mediana) ---
    def imputar_features(X_data, is_training=False, numerical_imputer=None):
        X = X_data.copy()
        
        # Si es el conjunto de entrenamiento, calculamos los imputadores
        if is_training:
            numerical_imputer = {}
        
        for col in X.columns:
            if X[col].isna().any():
                if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col]):
                    # Imputación categórica: 'Desconocido'
                    X[col] = X[col].fillna('Desconocido')
                else:
                    # Imputación numérica: Usamos la MEDIANA (más robusta que la media)
                    if is_training:
                        # Guardamos la mediana del entrenamiento para aplicarla en la predicción
                        median_val = X[col].median()
                        numerical_imputer[col] = median_val
                    else:
                        # Usamos la mediana calculada en el entrenamiento
                        median_val = numerical_imputer.get(col)
                        if median_val is not None:
                             X[col] = X[col].fillna(median_val)
                        else:
                            # Esto es una protección si una columna no tenía faltantes en train
                            X[col] = X[col].fillna(X[col].median()) 
                            
        return X, numerical_imputer

    # --- CASO 1: Columna objetivo TIENE valores faltantes (IMPUTAR) ---
    if df_work[nombre_objetivo].isna().any():
        df_entrenamiento = df_work[df_work[nombre_objetivo].notna()]
        df_prediccion = df_work[df_work[nombre_objetivo].isna()]

        if len(df_entrenamiento) == 0:
            raise ValueError(f"No hay valores conocidos en '{nombre_objetivo}' para entrenar el modelo.")

        y_entrenamiento = df_entrenamiento[nombre_objetivo]
        X_entrenamiento = df_entrenamiento.drop(columns=[nombre_objetivo])
        X_prediccion = df_prediccion.drop(columns=[nombre_objetivo])

        # 1. Limpiar e Imputar FEATURES
        X_entrenamiento, imputadores_num = imputar_features(X_entrenamiento, is_training=True)
        X_prediccion, _ = imputar_features(X_prediccion, numerical_imputer=imputadores_num)

        # 2. Codificar variables categóricas
        label_encoders = {}
        for col in X_entrenamiento.columns:
            if X_entrenamiento[col].dtype == "object" or pd.api.types.is_categorical_dtype(X_entrenamiento[col]):
                le = LabelEncoder()
                # Aseguramos que solo haya valores conocidos en el entrenamiento para ajustarlo
                X_entrenamiento[col] = le.fit_transform(X_entrenamiento[col].astype(str))
                
                # Para la predicción, manejamos posibles valores desconocidos (aunque la imputación previa ayuda)
                # Creamos un mapeo seguro para evitar errores en transform.
                mapping = {label: index for index, label in enumerate(le.classes_)}
                
                # Transformamos la columna de predicción
                X_prediccion[col] = X_prediccion[col].astype(str).map(mapping).fillna(len(le.classes_)).astype(int)
                label_encoders[col] = le
        
        # 3. Determinar el tipo de modelo y codificar TARGET
        es_clasificacion = y_entrenamiento.dtype == "object" or pd.api.types.is_categorical_dtype(y_entrenamiento)
        
        if es_clasificacion:
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y_entrenamiento.astype(str))
            model = DecisionTreeClassifier(random_state=42)
            class_names = le_target.classes_.tolist() # Se guarda para la exportación
        else:
            y_encoded = y_entrenamiento
            model = DecisionTreeRegressor(random_state=42)
            class_names = None

        # 4. Entrenar y evaluar
        X_train, X_test, y_train, y_test = train_test_split(
            X_entrenamiento, y_encoded, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        precision = model.score(X_test, y_test)

        # 5. Generar reglas (mejorado con class_names)
        reglas_params = {'feature_names': list(X_entrenamiento.columns)}
        if class_names is not None:
            reglas_params['class_names'] = class_names
            
        reglas = export_text(model, **reglas_params)

        # 6. Predecir valores faltantes e imputar
        df_completo = df.copy()
        predicciones_encoded = model.predict(X_prediccion)
        
        if es_clasificacion:
            # Transformar las predicciones de vuelta a su valor original de cadena
            predicciones = le_target.inverse_transform(predicciones_encoded.astype(int))
        else:
            predicciones = predicciones_encoded
            
        indices_prediccion = df_prediccion.index
        df_completo.loc[indices_prediccion, nombre_objetivo] = predicciones

        return precision, model, reglas, df_completo, True  # True = hubo predicción

    # --- CASO 2: Columna objetivo NO tiene faltantes (SOLO CATEGORIZAR/MODELAR) ---
    else:
        # Preparar datos
        y = df_work[nombre_objetivo]
        X = df_work.drop(columns=[nombre_objetivo])

        # 1. Limpiar e Imputar FEATURES (usando mediana del conjunto completo)
        X, _ = imputar_features(X, is_training=True)

        # 2. Codificar variables categóricas
        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == "object" or pd.api.types.is_categorical_dtype(X[col]):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le

        # 3. Codificar target y crear modelo
        es_clasificacion = y.dtype == "object" or pd.api.types.is_categorical_dtype(y)

        if es_clasificacion:
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y.astype(str))
            model = DecisionTreeClassifier(random_state=42)
            class_names = le_target.classes_.tolist()
        else:
            y_encoded = y
            model = DecisionTreeRegressor(random_state=42)
            class_names = None

        # 4. Entrenar y evaluar
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        precision = model.score(X_test, y_test)

        # 5. Generar reglas (mejorado con class_names)
        reglas_params = {'feature_names': list(X.columns)}
        if class_names is not None:
            reglas_params['class_names'] = class_names
            
        reglas = export_text(model, **reglas_params)

        return precision, model, reglas, df, False # False = no hubo predicción