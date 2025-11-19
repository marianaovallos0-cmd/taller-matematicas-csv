from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def imputar_knn(df, vecinos=3):
   
    df_copy = df.copy()
    
    # Seleccionar solo columnas numéricas
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        st.warning("⚠️ No hay columnas numéricas para aplicar KNN")
        return df_copy
    
    # Crear y aplicar el imputador KNN
    imputer = KNNImputer(n_neighbors=vecinos)
    numeric_array = imputer.fit_transform(df_copy[numeric_cols])
    
    # Reconstruir el DataFrame
    numeric_df = pd.DataFrame(numeric_array, columns=numeric_cols, index=df_copy.index)
    df_copy[numeric_cols] = numeric_df
    
    return df_copy


def imputar_k_modes(df, k=3):
    
    df_copy = df.copy()
    
    # Seleccionar solo columnas categóricas
    object_cols = df_copy.select_dtypes(include=['object', 'category']).columns
    
    if len(object_cols) == 0:
        st.warning("⚠️ No hay columnas categóricas para aplicar K-Modes")
        return df_copy
    
    # Para cada columna categórica, rellenar con la moda
    for columna in object_cols:
        moda = df_copy[columna].mode(dropna=True)
        if len(moda) > 0:
            df_copy[columna] = df_copy[columna].fillna(moda.iloc[0])
        else:
            # Si no hay moda (columna vacía), usar "missing"
            df_copy[columna] = df_copy[columna].fillna("missing")
    
    return df_copy


def imputar_k_means(df, n_clusters=3):
   
    df_copy = df.copy()
    
    # Seleccionar solo columnas numéricas
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        st.warning("⚠️ No hay columnas numéricas para aplicar K-Means")
        return df_copy
    
    # Para cada columna con valores faltantes
    for columna in numeric_cols:
        if df_copy[columna].isnull().sum() > 0:
            
            # Separar datos completos e incompletos
            datos_completos = df_copy.dropna(subset=[columna])
            datos_incompletos = df_copy[df_copy[columna].isnull()]
            
            if len(datos_completos) == 0:
                continue  # Si no hay datos completos, saltar esta columna
            
            # Preparar características para clustering (excluir la columna objetivo)
            caracteristicas = [c for c in numeric_cols if c != columna]
            
            if len(caracteristicas) > 0:
                # Entrenar K-Means con datos completos
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(datos_completos[caracteristicas])
                
                # Para cada dato incompleto, encontrar el cluster más cercano
                for idx in datos_incompletos.index:
                    punto = df_copy.loc[idx, caracteristicas].values.reshape(1, -1)
                    cluster = kmeans.predict(punto)[0]
                    
                    # Calcular la media de la columna faltante en ese cluster
                    datos_cluster = datos_completos[kmeans.labels_ == cluster]
                    valor_imputado = datos_cluster[columna].mean()
                    
                    # Rellenar el valor faltante
                    df_copy.loc[idx, columna] = valor_imputado
            else:
                # Si no hay otras características, usar la media global
                media = datos_completos[columna].mean()
                df_copy[columna] = df_copy[columna].fillna(media)
    
    return df_copy