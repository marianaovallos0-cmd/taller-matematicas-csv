import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import _tree

def generar_reglas_legibles(modelo, feature_names, target_encoder=None):
    """Genera reglas legibles a partir del árbol de decisión"""
    tree_ = modelo.tree_
    feature_names = [f"{name}" for i, name in enumerate(feature_names)]
    
    reglas = []

    def recursivo(nodo, regla_actual):
        if tree_.feature[nodo] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[nodo]]
            threshold = tree_.threshold[nodo]
            
            # Rama izquierda
            izquierda_regla = regla_actual + [f"{name} <= {threshold:.2f}"]
            recursivo(tree_.children_left[nodo], izquierda_regla)
            
            # Rama derecha
            derecha_regla = regla_actual + [f"{name} > {threshold:.2f}"]
            recursivo(tree_.children_right[nodo], derecha_regla)
        else:
            # Nodo hoja
            class_id = np.argmax(tree_.value[nodo])
            class_name = class_id
            if target_encoder is not None:
                try:
                    class_name = target_encoder.inverse_transform([class_id])[0]
                except:
                    class_name = class_id
            
            if regla_actual:
                regla_completa = "SI " + " Y ".join(regla_actual) + f" ENTONCES → {class_name}"
            else:
                regla_completa = f"PREDICCIÓN POR DEFECTO: {class_name}"
            
            reglas.append(regla_completa)

    recursivo(0, [])
    return "\n".join(reglas)

def entrenar_arbol_decision(df, columna_objetivo, columnas_usar):
    """Entrena árbol de decisión de forma robusta"""
    
    # Validaciones iniciales
    if columna_objetivo not in df.columns:
        raise Exception(f"La columna objetivo '{columna_objetivo}' no existe")
    
    for col in columnas_usar:
        if col not in df.columns:
            raise Exception(f"La columna '{col}' no existe")
    
    if len(columnas_usar) == 0:
        raise Exception("Debes seleccionar al menos una columna predictora")
    
    # Preparar datos
    data = df[columnas_usar + [columna_objetivo]].copy()
    
    # Separar datos completos vs datos con objetivo faltante
    datos_entrenamiento = data.dropna(subset=[columna_objetivo])
    datos_prediccion = data[data[columna_objetivo].isna()]
    
    if len(datos_entrenamiento) == 0:
        raise Exception("No hay datos con valores en la columna objetivo para entrenar")
    
    # Encoding de variables categóricas
    encoders = {}
    datos_encoded = datos_entrenamiento.copy()
    
    for col in columnas_usar + [columna_objetivo]:
        if datos_encoded[col].dtype == 'object':
            encoder = LabelEncoder()
            # Solo encoding si hay valores no nulos
            non_null_mask = datos_encoded[col].notna()
            if non_null_mask.any():
                datos_encoded.loc[non_null_mask, col] = encoder.fit_transform(
                    datos_encoded.loc[non_null_mask, col].astype(str)
                )
                encoders[col] = encoder
    
    # Preparar datos de entrenamiento
    X = datos_encoded[columnas_usar]
    y = datos_encoded[columna_objetivo]
    
    # Entrenar modelo
    modelo = DecisionTreeClassifier(
        max_depth=4, 
        min_samples_split=2, 
        random_state=42
    )
    modelo.fit(X, y)
    
    # Generar resultados
    arbol_texto = export_text(modelo, feature_names=columnas_usar)
    reglas_legibles = generar_reglas_legibles(
        modelo, columnas_usar, encoders.get(columna_objetivo)
    )
    
    # Predecir valores faltantes si hay
    valores_rellenados = None
    if len(datos_prediccion) > 0:
        try:
            # Preparar datos para predicción
            datos_pred_encoded = datos_prediccion[columnas_usar].copy()
            
            for col in columnas_usar:
                if col in encoders:
                    non_null_mask = datos_pred_encoded[col].notna()
                    if non_null_mask.any():
                        # Solo transformar valores que existen en el encoder
                        valid_values = []
                        for val in datos_pred_encoded.loc[non_null_mask, col]:
                            try:
                                encoded_val = encoders[col].transform([str(val)])[0]
                                valid_values.append(encoded_val)
                            except:
                                valid_values.append(-1)  # Valor fuera del entrenamiento
                        
                        datos_pred_encoded.loc[non_null_mask, col] = valid_values
            
            # Hacer predicciones
            predicciones = modelo.predict(datos_pred_encoded[columnas_usar])
            
            # Decodificar si es necesario
            if columna_objetivo in encoders:
                predicciones = encoders[columna_objetivo].inverse_transform(predicciones)
            
            # Crear DataFrame con resultados
            datos_prediccion = datos_prediccion.copy()
            datos_prediccion[columna_objetivo] = predicciones
            valores_rellenados = datos_prediccion
            
        except Exception as e:
            print(f"Advertencia: No se pudieron predecir valores faltantes: {e}")
    
    return {
        "arbol": arbol_texto,
        "reglas": reglas_legibles,
        "columnas_usadas": columnas_usar,
        "valores_rellenados": valores_rellenados,
        "muestras_entrenamiento": len(datos_entrenamiento)
    }