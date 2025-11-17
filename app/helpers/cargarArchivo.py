import pandas as pd

def cargar_csv(file):
    """
    Carga un archivo CSV subido por el usuario.
    Devuelve (dataframe, None) si salió bien, o (None, mensaje_error).
    """
    try:
        # streamlit pasa un _io.BytesIO; pandas lo maneja bien
        df = pd.read_csv(file)
        return df, None
    except Exception as e:
        return None, str(e)
