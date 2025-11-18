import streamlit as st
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# ----------------------------------------------
# Función para entrenar el modelo de decisión
# ----------------------------------------------
def entrenar_arbol_decision(df, target_column, use_test_data=False, use_tree_settings=False,
                            max_depth=None, min_samples_split=2, min_samples_leaf=1):

    st.subheader("🔍 Preprocesamiento: Imputación y Codificación")

    # Separar Y
    y = df[target_column]

    # Crear X sin la columna objetivo
    X = df.drop(columns=[target_column])

    #############################################
    # 1. Imputación simple de numéricos y categóricos
    #############################################
    st.write("✔ Aplicando imputación simple (media para numéricos y moda para categóricos)...")

    numeric_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    imputer_num = SimpleImputer(strategy='mean')
    imputer_cat = SimpleImputer(strategy='most_frequent')

    if len(numeric_cols) > 0:
        X[numeric_cols] = imputer_num.fit_transform(X[numeric_cols])
        st.write("Numericos imputados:", list(numeric_cols))

    if len(categorical_cols) > 0:
        X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])
        st.write("Categóricos imputados:", list(categorical_cols))

    #############################################
    # 2. Codificación categórica
    #############################################
    st.write("✔ Codificando categorías con LabelEncoder...")

    label_encoders = {}
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        st.write(f"Columna {col} → codificada en números")

    #############################################
    # 3. Determinar si es clasificación o regresión
    #############################################
    es_clasificacion = y.dtype == 'object' or y.nunique() < 20

    if es_clasificacion:
        y = LabelEncoder().fit_transform(y)
        modelo = DecisionTreeClassifier(
            max_depth=max_depth if use_tree_settings else None,
            min_samples_split=min_samples_split if use_tree_settings else 2,
            min_samples_leaf=min_samples_leaf if use_tree_settings else 1
        )
        st.info("📘 Clasificación detectada")
    else:
        modelo = DecisionTreeRegressor(
            max_depth=max_depth if use_tree_settings else None,
            min_samples_split=min_samples_split if use_tree_settings else 2,
            min_samples_leaf=min_samples_leaf if use_tree_settings else 1
        )
        st.info("📗 Regresión detectada")

    #############################################
    # 4. Entrenamiento con o sin split
    #############################################
    if use_test_data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        st.subheader("📊 Resultados del Modelo")

        if es_clasificacion:
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"✔ Accuracy: **{accuracy:.3f}**")
            st.write("📄 Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.write("📌 Matriz de Confusión:")
            st.write(confusion_matrix(y_test, y_pred))
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"✔ MSE: **{mse:.3f}**")
            st.write(f"✔ R²: **{r2:.3f}**")

    else:
        modelo.fit(X, y)

    #############################################
    # 5. Mostrar reglas del árbol
    #############################################
    st.subheader("📝 Reglas del Árbol de Decisión (texto)")

    reglas = export_text(modelo, feature_names=list(X.columns))
    st.text_area("Reglas del modelo:", reglas, height=300)

    #############################################
    # 6. Mostrar el árbol graficado (mucho más bonito)
    #############################################
    st.subheader("🌳 Visualización del Árbol de Decisión")

    fig = plt.figure(figsize=(18, 10))
    plot_tree(
        modelo,
        feature_names=X.columns,
        filled=True,
        rounded=True
    )
    st.pyplot(fig)

    #############################################
    # 7. Retornar el modelo entrenado
    #############################################
    return modelo
