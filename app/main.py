import streamlit as st
import pandas as pd
from helpers.cargarArchivo import cargar_csv
from controller import (
    aplicar_imputacion, aplicar_normalizacion,
    aplicar_discretizacion, aplicar_categorizacion
)

st.set_page_config(page_title="Taller Matemáticas Aplicadas - CSV", layout="centered")

st.title(st.title("PRUEBA — SI VES ESTO, SE ACTUALIZÓ")
)
st.write("Sube un archivo CSV y aplica una operación: relleno, normalización, discretización o árbol de decisión.")

archivo = st.file_uploader("Sube un archivo CSV", type=["csv"])

if archivo is not None:
    df, error = cargar_csv(archivo)
    if error:
        st.error(f"No se pudo cargar el archivo: {error}")
    else:
        st.subheader("Datos originales")
        st.dataframe(df)

        st.subheader("Seleccione la operación")
        opcion = st.selectbox(
            "¿Qué quieres hacer?",
            ["Relleno de valores faltantes", "Normalización", "Discretización", "Árbol de decisión"]
        )

        if opcion == "Relleno de valores faltantes":
            metodo = st.selectbox("Método:", ["KNN", "K-Modes", "Mean", "Median", "Mode"])
            if st.button("Aplicar relleno"):
                resultado = aplicar_imputacion(df.copy(), metodo)
                st.subheader("Resultado")
                st.dataframe(resultado)
                csv = resultado.to_csv(index=False).encode('utf-8')
                st.download_button("Descargar resultado (CSV)", data=csv, file_name="resultado_imputacion.csv", mime="text/csv")

        elif opcion == "Normalización":
            metodo = st.selectbox("Método:", ["Z-Score", "Min-Max", "Log"])
            if st.button("Aplicar normalización"):
                resultado = aplicar_normalizacion(df.copy(), metodo)
                st.subheader("Resultado")
                st.dataframe(resultado)
                csv = resultado.to_csv(index=False).encode('utf-8')
                st.download_button("Descargar resultado (CSV)", data=csv, file_name="resultado_normalizacion.csv", mime="text/csv")

        elif opcion == "Discretización":
            metodo = st.selectbox("Método:", ["Equal Width", "Equal Frequency"])
            if st.button("Aplicar discretización"):
                resultado = aplicar_discretizacion(df.copy(), metodo)
                st.subheader("Resultado")
                st.dataframe(resultado)
                csv = resultado.to_csv(index=False).encode('utf-8')
                st.download_button("Descargar resultado (CSV)", data=csv, file_name="resultado_discretizacion.csv", mime="text/csv")

        elif opcion == "Árbol de decisión":
            columnas = list(df.columns)
            nombre_objetivo = st.selectbox("Selecciona la columna objetivo (target):", columnas)

            if st.button("Entrenar árbol"):
                try:
                    precision, modelo, reglas = aplicar_categorizacion(df.copy(), nombre_objetivo)
                    st.success(f"Precisión en el conjunto de prueba: {precision:.2f}")

                    st.subheader("Reglas del árbol (texto)")
                    st.text(reglas)

                except Exception as e:
                    st.error(f"Ocurrió un error al entrenar el árbol: {e}")

