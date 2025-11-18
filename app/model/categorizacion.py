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
    
    # IDENTIFICAR Y EXCLUIR COLUMNAS NO APROPIADAS para el modelo
    columnas_a_excluir = []
    
    for columna in df_work.columns:
        # Excluir columnas con valores únicos (como IDs) o casi únicos
        if columna == nombre_objetivo:
            continue  # No excluir la columna objetivo
        
        # Si la columna tiene más del 90% de valores únicos, probablemente es un ID
        valores_unicos = df_work[columna].nunique()
        total_valores = len(df_work[columna])
        
        if valores_unicos / total_valores > 0.9:  # Más del 90% únicos
            columnas_a_excluir.append(columna)
            print(f"Excluyendo columna '{columna}' (demasiados valores únicos: {valores_unicos}/{total_valores})")
    
    # Excluir las columnas identificadas (excepto la objetivo)
    if columnas_a_excluir:
        df_work = df_work.drop(columns=columnas_a_excluir)
        st.info(f"Columnas excluidas del modelo: {columnas_a_excluir}")

    # Separar datos para entrenamiento y predicción
    df_entrenamiento = df_work[df_work[nombre_objetivo].notna()]  # Filas con target conocido
    df_prediccion = df_work[df_work[nombre_objetivo].isna()]      # Filas con target faltante

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

    # Limpieza de datos - llenar NaN con valores apropiados
    for col in X_entrenamiento.columns:
        if X_entrenamiento[col].dtype == 'object':
            X_entrenamiento[col] = X_entrenamiento[col].fillna('Desconocido')
        else:
            X_entrenamiento[col] = X_entrenamiento[col].fillna(0)
    
    if len(X_prediccion) > 0:
        for col in X_prediccion.columns:
            if X_prediccion[col].dtype == 'object':
                X_prediccion[col] = X_prediccion[col].fillna('Desconocido')
            else:
                X_prediccion[col] = X_prediccion[col].fillna(0)

    # Codificar variables categóricas SOLO para columnas con pocos valores únicos
    label_encoders = {}
    columnas_a_codificar = []
    
    for col in X_entrenamiento.columns:
        # Solo codificar columnas con menos del 50% de valores únicos
        if (X_entrenamiento[col].dtype == "object" and 
            X_entrenamiento[col].nunique() / len(X_entrenamiento[col]) < 0.5):
            columnas_a_codificar.append(col)
    
    for col in columnas_a_codificar:
        try:
            le = LabelEncoder()
            X_entrenamiento[col] = le.fit_transform(X_entrenamiento[col].astype(str))
            if len(X_prediccion) > 0:
                # Para datos de predicción, usar transform y manejar valores nuevos
                mascara_valores_conocidos = X_prediccion[col].isin(le.classes_)
                X_prediccion.loc[mascara_valores_conocidos, col] = le.transform(
                    X_prediccion.loc[mascara_valores_conocidos, col].astype(str)
                )
                # Para valores no vistos, asignar un valor por defecto (moda)
                if not mascara_valores_conocidos.all():
                    moda = X_entrenamiento[col].mode()
                    valor_por_defecto = moda.iloc[0] if len(moda) > 0 else 0
                    X_prediccion.loc[~mascara_valores_conocidos, col] = valor_por_defecto
            label_encoders[col] = le
        except Exception as e:
            print(f"Error codificando columna {col}: {e}")

    # Codificar target y crear modelo
    if y_entrenamiento.dtype == "object":
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y_entrenamiento.astype(str))
        model = DecisionTreeClassifier(random_state=42, max_depth=5)  # Limitar profundidad
        es_clasificacion = True
    else:
        y_encoded = y_entrenamiento
        model = DecisionTreeRegressor(random_state=42, max_depth=5)
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
    except Exception as e:
        # Si falla la división, entrenar con todos los datos
        model.fit(X_entrenamiento, y_encoded)
        precision = model.score(X_entrenamiento, y_encoded) if hasattr(model, 'score') else 0
        print(f"Usando todos los datos para entrenamiento: {e}")

    # Generar reglas en texto
    try:
        reglas = export_text(model, feature_names=list(X_entrenamiento.columns))
    except Exception:
        reglas = "No se pudieron generar reglas."

    # Generar reglas en formato tabla
    try:
        reglas_tabla = extraer_reglas_tabla(model, X_entrenamiento.columns, nombre_objetivo, label_encoders, le_target if es_clasificacion else None)
    except Exception as e:
        reglas_tabla = []
        print(f"Error extrayendo reglas en tabla: {e}")

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
            
            st.success(f"✅ Se predijeron {len(predicciones)} valores faltantes en '{nombre_objetivo}'")
            
        except Exception as e:
            st.warning(f"⚠️ No se pudieron predecir algunos valores: {e}")

    return precision, model, reglas, reglas_tabla, df_completo

def extraer_reglas_tabla(model, features, target_name, label_encoders, le_target=None):
    """
    Extrae las reglas del árbol en formato de tabla
    """
    from sklearn.tree import _tree
    
    tree_ = model.tree_
    feature_names = features
    reglas = []
    
    def recorrer_arbol(nodo=0, regla_actual=None, profundidad=0):
        if regla_actual is None:
            regla_actual = []
            
        # Si es nodo hoja
        if tree_.feature[nodo] == _tree.TREE_UNDEFINED:
            # Obtener la clase predicha
            if hasattr(model, 'classes_'):
                clase_codificada = np.argmax(tree_.value[nodo])
                # Convertir a valor original si es clasificación
                if le_target is not None:
                    try:
                        clase = le_target.inverse_transform([clase_codificada])[0]
                    except:
                        clase = clase_codificada
                else:
                    clase = clase_codificada
            else:
                # Para regresión
                clase = tree_.value[nodo][0][0]
            
            # Construir la regla
            condiciones = " Y ".join(regla_actual)
            if condiciones:  # Solo agregar reglas con condiciones
                reglas.append({
                    "Número": len(reglas) + 1,
                    "Condiciones": condiciones,
                    f"Entonces {target_name}": f"= {clase}"
                })
            return
        
        # Si es muy profundo, detener la recursión
        if profundidad > 10:
            return
            
        # Obtener características del nodo
        feature = feature_names[tree_.feature[nodo]]
        threshold = tree_.threshold[nodo]
        
        # Para características categóricas codificadas
        if feature in label_encoders:
            try:
                valor_original = label_encoders[feature].inverse_transform([int(threshold)])[0]
                cond_izq = f"{feature} = {valor_original}"
                cond_der = f"{feature} ≠ {valor_original}"
            except:
                cond_izq = f"{feature} <= {threshold:.2f}"
                cond_der = f"{feature} > {threshold:.2f}"
        else:
            # Para características numéricas
            cond_izq = f"{feature} <= {threshold:.2f}"
            cond_der = f"{feature} > {threshold:.2f}"
        
        # Recorrer ramas
        nueva_regla = regla_actual + [cond_izq]
        recorrer_arbol(tree_.children_left[nodo], nueva_regla, profundidad + 1)
        
        nueva_regla = regla_actual + [cond_der]
        recorrer_arbol(tree_.children_right[nodo], nueva_regla, profundidad + 1)
    
    recorrer_arbol()
    return reglas[:20]  # Limitar a 20 reglas para no saturar