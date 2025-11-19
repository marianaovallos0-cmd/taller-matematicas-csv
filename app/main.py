import streamlit as st
import pandas as pd
import numpy as np
from helpers.cargarArchivo import cargar_csv
from controller.controller import (
    aplicar_imputacion,
    aplicar_normalizacion,
    aplicar_discretizacion,
    aplicar_arbol_decision
)

# Configuración de la página
st.set_page_config(
    page_title="Taller Matemáticas Aplicadas - CSV",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("📊 Taller Matemáticas Aplicadas - Procesamiento de CSV")
st.markdown("---")

# Sidebar para información
with st.sidebar:
    st.header("ℹ️ Información")
    st.markdown("""
    **Operaciones disponibles:**
    - 🔍 **Relleno de valores faltantes**
    - 📐 **Normalización de datos**
    - 📊 **Discretización**
    - 🌳 **Árboles de decisión**
    """)
    
    st.header("📝 Instrucciones")
    st.markdown("""
    1. Sube tu archivo CSV
    2. Selecciona la operación
    3. Configura los parámetros
    4. Aplica y descarga resultados
    """)

# ---- 1. SUBIR ARCHIVO ----
st.header("1. Subir Archivo CSV")
archivo = st.file_uploader("Selecciona un archivo CSV", type=["csv"], help="El archivo debe tener estructura de tabla con columnas y filas")

df = None
if archivo is not None:
    with st.spinner("Cargando y validando archivo..."):
        df, error = cargar_csv(archivo)
    
    if error:
        st.error(f"❌ Error al cargar el archivo: {error}")
        st.stop()
    
    # Mostrar información del dataset
    st.success(f"✅ Archivo cargado correctamente: {df.shape[0]} filas × {df.shape[1]} columnas")
    
    # Mostrar pestañas para explorar datos
    tab1, tab2, tab3 = st.tabs(["📋 Datos", "📊 Estadísticas", "⚠️ Valores Faltantes"])
    
    with tab1:
        st.subheader("Vista previa de los datos")
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.subheader("Estadísticas descriptivas")
        st.dataframe(df.describe(include='all').fillna(''), use_container_width=True)
    
    with tab3:
        st.subheader("Análisis de valores faltantes")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Conteo de valores faltantes:**")
            st.write(missing_data[missing_data > 0])
        
        with col2:
            st.write("**Porcentaje de valores faltantes:**")
            st.write(missing_percent[missing_percent > 0])
        
        # Alertas específicas
        columnas_vacias = df.columns[df.isnull().all()].tolist()
        if columnas_vacias:
            st.warning(f"🚨 **Columnas completamente vacías:** {columnas_vacias}")
        
        columnas_muchos_nulos = missing_percent[missing_percent > 50].index.tolist()
        if columnas_muchos_nulos:
            st.warning(f"⚠️ **Columnas con más del 50% de valores faltantes:** {columnas_muchos_nulos}")

    st.markdown("---")

    # ---- 2. SELECCIÓN DE OPERACIÓN ----
    st.header("2. Seleccionar Operación")
    
    operacion = st.selectbox(
        "Elige la operación a realizar:",
        ["Relleno de valores faltantes", "Normalización", "Discretización", "Árbol de decisión"],
        help="Selecciona el tipo de procesamiento que deseas aplicar"
    )

    # ---- 3. OPERACIONES ----
    
    # 🔍 RELLENO DE VALORES FALTANTES
    if operacion == "Relleno de valores faltantes":
        st.subheader("🔍 Relleno de Valores Faltantes")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            metodo = st.selectbox(
                "Método de imputación:",
                ["KNN", "K-Modes", "Mean", "Median", "Mode"],
                help="KNN: Para datos numéricos | K-Modes: Para datos categóricos"
            )
        
        with col2:
            if metodo == "KNN":
                vecinos = st.number_input("Número de vecinos", min_value=1, max_value=10, value=3)
        
        if st.button("🔄 Aplicar Relleno de Valores", type="primary"):
            with st.spinner("Aplicando imputación..."):
                try:
                    if metodo == "KNN":
                        resultado = aplicar_imputacion(df.copy(), metodo)
                    else:
                        resultado = aplicar_imputacion(df.copy(), metodo)
                    
                    # Mostrar resultados
                    st.success("✅ Imputación completada exitosamente")
                    
                    # Comparación antes/después
                    col_before, col_after = st.columns(2)
                    
                    with col_before:
                        st.subheader("Antes (Valores Faltantes)")
                        missing_before = df.isnull().sum().sum()
                        st.metric("Valores faltantes", missing_before)
                    
                    with col_after:
                        st.subheader("Después (Valores Faltantes)")
                        missing_after = resultado.isnull().sum().sum()
                        st.metric("Valores faltantes", missing_after, delta=-missing_before)
                    
                    # Mostrar datos resultantes
                    st.subheader("Datos con Valores Rellenados")
                    st.dataframe(resultado, use_container_width=True)
                    
                    # Botón de descarga
                    csv = resultado.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "💾 Descargar Resultado (CSV)",
                        data=csv,
                        file_name=f"relleno_{metodo.lower()}.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error durante la imputación: {str(e)}")

    # 📐 NORMALIZACIÓN
    elif operacion == "Normalización":
        st.subheader("📐 Normalización de Datos")
        
        metodo = st.selectbox(
            "Método de normalización:",
            ["Z-Score", "Min-Max", "Log"],
            help="Z-Score: Media=0, Desv=1 | Min-Max: Escala [0,1] | Log: Transformación logarítmica"
        )
        
        if st.button("🔄 Aplicar Normalización", type="primary"):
            with st.spinner("Aplicando normalización..."):
                try:
                    resultado = aplicar_normalizacion(df.copy(), metodo)
                    
                    st.success("✅ Normalización completada exitosamente")
                    
                    # Mostrar datos normalizados
                    st.subheader("Datos Normalizados")
                    st.dataframe(resultado, use_container_width=True)
                    
                    # Estadísticas después de normalización
                    st.subheader("Estadísticas después de Normalización")
                    st.dataframe(resultado.describe(), use_container_width=True)
                    
                    # Botón de descarga
                    csv = resultado.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "💾 Descargar Resultado (CSV)",
                        data=csv,
                        file_name=f"normalizacion_{metodo.lower()}.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error durante la normalización: {str(e)}")

    # 📊 DISCRETIZACIÓN
    elif operacion == "Discretización":
        st.subheader("📊 Discretización de Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            metodo = st.selectbox(
                "Método de discretización:",
                ["Equal Width", "Equal Frequency"],
                help="Equal Width: Intervalos iguales | Equal Frequency: Misma cantidad de datos por intervalo"
            )
        
        with col2:
            bins = st.number_input("Número de bins", min_value=2, max_value=10, value=4)
        
        if st.button("🔄 Aplicar Discretización", type="primary"):
            with st.spinner("Aplicando discretización..."):
                try:
                    resultado = aplicar_discretizacion(df.copy(), metodo, bins=bins)
                    
                    st.success("✅ Discretización completada exitosamente")
                    
                    # Mostrar datos discretizados
                    st.subheader("Datos Discretizados")
                    st.dataframe(resultado, use_container_width=True)
                    
                    # Explicación de los bins
                    st.subheader("📝 Explicación")
                    st.info(f"Los datos numéricos han sido convertidos a {bins} categorías discretas usando el método {metodo}.")
                    
                    # Botón de descarga
                    csv = resultado.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "💾 Descargar Resultado (CSV)",
                        data=csv,
                        file_name=f"discretizacion_{metodo.lower()}.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error durante la discretización: {str(e)}")

    # 🌳 ÁRBOL DE DECISIÓN
    elif operacion == "Árbol de decisión":
        st.subheader("🌳 Árbol de Decisión para Categorización")
        
        col1, col2 = st.columns(2)
        
        with col1:
            columna_objetivo = st.selectbox(
                "Variable objetivo (y):",
                df.columns,
                help="La columna que quieres predecir o categorizar"
            )
        
        with col2:
            st.write("**Columnas disponibles para predictores:**")
            columnas_disponibles = [c for c in df.columns if c != columna_objetivo]
            columnas_seleccionadas = st.multiselect(
                "Variables predictoras (X):",
                columnas_disponibles,
                default=columnas_disponibles,
                help="Selecciona las columnas que usarás para predecir la variable objetivo"
            )
        
        if st.button("🌳 Entrenar Árbol de Decisión", type="primary"):
            if not columnas_seleccionadas:
                st.error("❌ Debes seleccionar al menos una columna predictora")
            else:
                with st.spinner("Entrenando árbol de decisión..."):
                    try:
                        resultado = aplicar_arbol_decision(
                            df.copy(), 
                            columna_objetivo, 
                            columnas_seleccionadas
                        )
                        
                        st.success("✅ Árbol de decisión entrenado exitosamente")
                        
                        # Mostrar información del entrenamiento
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Muestras de entrenamiento", resultado["muestras_entrenamiento"])
                        with col2:
                            st.metric("Variables usadas", len(columnas_seleccionadas))
                        
                        # Mostrar árbol en formato texto
                        st.subheader("🌳 Estructura del Árbol")
                        st.code(resultado["arbol"], language="text")
                        
                        # Mostrar reglas legibles
                        st.subheader("📋 Reglas de Decisión")
                        st.code(resultado["reglas"], language="text")
                        
                        # Mostrar valores rellenados si los hay
                        if resultado["valores_rellenados"] is not None:
                            st.subheader("🎯 Valores Faltantes Rellenados")
                            st.dataframe(resultado["valores_rellenados"], use_container_width=True)
                            
                            # Botón para descargar valores rellenados
                            csv_rellenado = resultado["valores_rellenados"].to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "💾 Descargar Valores Rellenados",
                                data=csv_rellenado,
                                file_name="valores_arbol_rellenados.csv",
                                mime="text/csv"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Error al entrenar el árbol: {str(e)}")

else:
    # Estado cuando no hay archivo cargado
    st.info("👆 Por favor, sube un archivo CSV para comenzar el procesamiento.")
    
    # Ejemplo de datos
    with st.expander("📋 ¿Qué formato debe tener el CSV?"):
        st.markdown("""
        **Estructura esperada:**
        - Primera fila: Nombres de columnas
        - Filas siguientes: Datos
        - Separadores: coma (,) o punto y coma (;)
        - Codificación: UTF-8
        
        **Ejemplo:**
        ```
        edad,ingreso,departamento,categoria
        25,50000,Ventas,A
        30,60000,IT,B
        35,70000,Ventas,A
        ```
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Taller de Matemáticas Aplicadas - Procesamiento de Datos CSV"
    "</div>",
    unsafe_allow_html=True
)