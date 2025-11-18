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

    # Guardar los valores originales del target
    y_original = df[nombre_objetivo].copy()

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
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))
        model = DecisionTreeClassifier()
        # Guardar el LabelEncoder del target para poder decodificar después
        model._le = le_target
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

    # NUEVO: REGLAS EN FORMATO TABLA CON VALORES ORIGINALES
    try:
        reglas_tabla = extraer_reglas_tabla(model, X.columns, nombre_objetivo, label_encoders, y_original)
    except Exception as e:
        reglas_tabla = []
        print(f"Error extrayendo reglas en tabla: {e}")

    return precision, model, reglas, reglas_tabla

def extraer_reglas_tabla(model, features, target_name, label_encoders, y_original):
    """
    Extrae las reglas del árbol en formato de tabla usando valores originales
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
            if hasattr(model, 'classes_'):
                # Para clasificación - obtener la clase original
                clase_codificada = np.argmax(tree_.value[nodo])
                # Convertir de vuelta al valor original si es posible
                if hasattr(model, 'classes_') and len(model.classes_) == len(np.unique(y_original)):
                    try:
                        clase = model.classes_[clase_codificada]
                        # Si el target fue codificado, obtener valor original
                        if isinstance(clase, (int, np.integer)) and hasattr(model, '_le'):
                            clase = model._le.inverse_transform([clase])[0]
                    except:
                        clase = clase_codificada
                else:
                    clase = clase_codificada
            else:
                # Para regresión
                clase = tree_.value[nodo][0][0]
            
            # Construir la regla completa con valores originales
            condiciones = " Y ".join(regla_actual)
            reglas.append({
                "Número": len(reglas) + 1,
                "Condiciones": condiciones if condiciones else "Todos los casos",
                f"Entonces {target_name}": f"= {clase}"
            })
            return
        
        # Obtener características del nodo
        feature = feature_names[tree_.feature[nodo]]
        threshold = tree_.threshold[nodo]
        
        # Intentar decodificar el valor si esta característica fue codificada
        if feature in label_encoders:
            try:
                # Para características categóricas, mostrar el valor original
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
        
        # Rama izquierda (<=)
        nueva_regla = regla_actual + [cond_izq]
        recorrer_arbol(tree_.children_left[nodo], nueva_regla, profundidad + 1)
        
        # Rama derecha (>)
        nueva_regla = regla_actual + [cond_der]
        recorrer_arbol(tree_.children_right[nodo], nueva_regla, profundidad + 1)
    
    recorrer_arbol()
    return reglas