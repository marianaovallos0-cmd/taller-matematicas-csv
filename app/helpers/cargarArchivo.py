import pandas as pd
from model.exceptions import ArchivoInvalidoException, TablaInvalidaException

def cargar_csv(file):
    """
    Carga un archivo CSV subido por el usuario.
    Devuelve (dataframe, None) si salió bien, o (None, mensaje_error) si falla.
    """

    # 1. Validar que el archivo sea CSV estrictamente
    if not file.name.lower().endswith(".csv"):
        return None, "❌ El archivo debe ser formato CSV estrictamente."

    try:
        # 2. Intentar leer
        df = pd.read_csv(file)

        # 3. Validar que tenga columnas
        if df.empty or len(df.columns) == 0:
            raise TablaInvalidaException("❌ El archivo no contiene una tabla válida.")

        # 4. Validar que tenga filas
        if df.shape[0] == 0:
            raise TablaInvalidaException("❌ El archivo está vacío, no hay datos para procesar.")

        return df, None

    except TablaInvalidaException as e:
        return None, str(e)

    except Exception:
        return None, "❌ El archivo no tiene estructura de tabla válida (debe tener columnas separadas por comas)."
