from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def entrenar_arbol_decision(df, objetivo):
    """
    Entrena un árbol de decisión simple.
    Devuelve: (precision, modelo)
    """
    X = df.drop(columns=[objetivo])
    y = df[objetivo]

    # convertir variables categóricas a dummies simples
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    prediccion = modelo.predict(X_test)
    precision = accuracy_score(y_test, prediccion)

    return precision, modelo
