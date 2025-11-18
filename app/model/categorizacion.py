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
        y = LabelEncoder().fit_transform(y.astype(str))
        model = DecisionTreeClassifier()
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

    # NUEVO: REGLAS EN FORMATO TABLA
    try:
        reglas_tabla = extraer_reglas_tabla(model, X.columns, nombre_objetivo)
    except Exception as e:
        reglas_tabla = []
        print(f"Error extrayendo reglas en tabla: {e}")

    return precision, model, reglas, reglas_tabla

def extraer_reglas_tabla(model, features, target_name):
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
            if hasattr(model, 'classes_'):
                clase = model.classes_[np.argmax(tree_.value[nodo])]
            else:
                clase = tree_.value[nodo][0][0]
            
            # Construir la regla completa
            condiciones = " Y ".join(regla_actual)
            reglas.append({
                "Condiciones": condiciones if condiciones else "Todos los casos",
                f"Categoría {target_name}": clase
            })
            return
        
        # Obtener características del nodo
        feature = feature_names[tree_.feature[nodo]]
        threshold = tree_.threshold[nodo]
        
        # Rama izquierda (<=)
        nueva_regla = regla_actual + [f"{feature} <= {threshold:.2f}"]
        recorrer_arbol(tree_.children_left[nodo], nueva_regla, profundidad + 1)
        
        # Rama derecha (>)
        nueva_regla = regla_actual + [f"{feature} > {threshold:.2f}"]
        recorrer_arbol(tree_.children_right[nodo], nueva_regla, profundidad + 1)
    
    recorrer_arbol()
    return reglas