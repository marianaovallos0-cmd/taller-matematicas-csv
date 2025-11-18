import streamlit as st
import pandas as pd
from helpers.cargarArchivo import cargar_csv
from controller import (
    aplicar_imputacion, aplicar_normalizacion,
    aplicar_discretizacion, aplicar_arbol_decision
)

st.set_page_config(
    page_title="Taller Matemáticas Aplicadas - CSV", layout="centered"
)

st.title("Taller Matemáticas Aplicadas - CSV")
st.write("Sube un archivo CSV y aplica una operación: relleno, normalización, discretización o árbol de decisión.")

# ---- 1. Subir archivo ----
archivo = st.file_uploader("Sube un archivo CSV", type=["csv"])

if archivo is not None:
    df, error = cargar_csv(archivo)

    # Si hubo error → mostrar y detener
    if error:
        st.error(error)
        st.stop()

    # Mostrar datos cargados
    st.subheader("Datos originales")
    st.dataframe(df)

    # ---- 2. Elegir operación ----
    st.subheader("Seleccione la operación")
    opcion = st.selectbox(
        "¿Qué quieres hacer?",
        ["Relleno de valores faltantes", "Normalización",
         "Discretización", "Árbol de decisión"]
    )

    # ---- 3. Relleno de valores faltantes ----
    if opcion == "Relleno de valores faltantes":
        metodo = st.selectbox("Método:", ["KNN", "K-Modes", "Mean", "Median", "Mode"])

        if st.button("Aplicar relleno"):
            try:
                resultado = aplicar_imputacion(df.copy(), metodo)
                st.subheader("Resultado")
                st.dataframe(resultado)

                csv = resultado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Descargar resultado (CSV)",
                    data=csv,
                    file_name="resultado_imputacion.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error durante la imputación: {e}")

    # ---- 4. Normalización ----
    elif opcion == "Normalización":
        metodo = st.selectbox("Método:", ["Z-Score", "Min-Max", "Log"])

        if st.button("Aplicar normalización"):
            try:
                resultado = aplicar_normalizacion(df.copy(), metodo)
                st.subheader("Resultado")
                st.dataframe(resultado)

                csv = resultado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Descargar resultado (CSV)",
                    data=csv,
                    file_name="resultado_normalizacion.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error durante la normalización: {e}")

        # ---- 5. Discretización ----
        elif opcion == "Discretización":
            metodo = st.selectbox("Método:", ["Equal Width", "Equal Frequency"])

            if st.button("Aplicar discretización"):
                try:
                    resultado = aplicar_discretizacion(df.copy(), metodo)
                    st.subheader("Resultado")
                    st.dataframe(resultado)

                    csv = resultado.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Descargar resultado (CSV)",
                        data=csv,
                        file_name="resultado_discretizacion.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error durante la discretización: {e}")

        # ---- 6. Árbol de decisión ----
        elif opcion == "Árbol de decisión":
            columnas = list(df.columns)

            st.subheader("Tipo de modelo de Árbol")
            tipo_arbol = st.radio(
                "Selecciona el tipo:",
                ["Categorización (Clasificación)", "Predicción (Regresión)"]
            )

            nombre_objetivo = st.selectbox("Selecciona la columna objetivo:", columnas)

            if st.button("Entrenar árbol"):
                try:
                    precision, modelo, reglas, df_resultado, _ = aplicar_arbol_decision(
                        df.copy(),
                        nombre_objetivo,
                        tipo="clasificacion" if "Categorización" in tipo_arbol else "regresion"
                    )

                    st.subheader("Resultado")
                    st.dataframe(df_resultado)

                    csv = df_resultado.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Descargar resultado (CSV)",
                        data=csv,
                        file_name="resultado_arbol_decision.csv",
                        mime="text/csv"
                    )

                    st.subheader("Reglas del Árbol de Decisión")
                    st.text_area("Reglas:", reglas, height=380)

                except Exception as e:
                    st.error(f"Error durante el árbol de decisión: {e}")
