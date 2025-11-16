import pandas as pd

def cargar_csv(archivo):
    """
    Carga un archivo CSV y devuelve (dataframe, error).
    Si no hay error, error es None.
    """
    try:
        df = pd.read_csv(archivo)
        return df, None
    except Exception as e:
        return None, str(e)
