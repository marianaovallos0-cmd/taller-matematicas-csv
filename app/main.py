import streamlit as st
import pandas as pd
from helpers.cargarArchivo import cargar_csv
from controller import (
    apply_imputation, apply_normalization,
    apply_discretization, apply_decision_tree
)

st.set_page_config(page_title="Taller Matemáticas Aplicadas - CSV", layout="centered")

st.title("Taller Matemáticas Aplicadas — CSV")

archivo_subido = st.file_uploader("Sube un archivo CSV", type=["csv"])

if archivo_subido is not None:
    df, error = cargar_csv(archivo_subido)

    if error:
        st.error(f"No se pudo cargar el archivo: {error}")
    else:
        st.subheader("Datos originales")
        st.dataframe(df)

        st.subheader("Selecciona la operación")
        opcion = st.selectbox(
            "¿Qué quieres hacer?",
            ["Relleno de valores faltantes",
             "Normalización",
             "Discretización",
             "Árbol de decisión"]
        )

        if opcion == "Relleno de valores faltantes":
            metodo = st.selectbox("Método:", ["KNN", "K-Modes", "Mean", "Median", "Mode"])
            if st.button("Aplicar imputación"):
                resultado = apply_imputation(df.copy(), metodo)
                st.subheader("Resultado")
                st.dataframe(resultado)

        elif opcion == "Normalización":
            metodo = st.selectbox("Método:", ["Z-Score", "Min-Max", "Log"])
            if st.button("Aplicar normalización"):
                resultado = apply_normalization(df.copy(), metodo)
                st.subheader("Resultado")
                st.dataframe(resultado)

        elif opcion == "Discretización":
            metodo = st.selectbox("Método:", ["Equal Width", "Equal Frequency"])
            if st.button("Aplicar discretización"):
                resultado = apply_discretizacion(df.copy(), metodo)
                st.subheader("Resultado")
                st.dataframe(resultado)

        elif opcion == "Árbol de decisión":
            objetivo = st.selectbox("Selecciona la columna objetivo:", df.columns)
            if st.button("Entrenar árbol"):
                precision, _ = apply_decision_tree(df.copy(), objetivo)
                st.success(f"Precisión del modelo: {precision:.2f}")
