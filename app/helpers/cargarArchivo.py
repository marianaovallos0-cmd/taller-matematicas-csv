import pandas as pd
from model.exceptions import ArchivoInvalidoException, TablaInvalidaException

def cargar_csv(file):
    """
    Carga un archivo CSV con validaciones estrictas.
    Retorna (df, None) si funciona.
    Retorna (None, mensaje_error) si falla.
    """
    # 1. Validación estricta de extensión
    if not file.name.lower().endswith(".csv"):
        return None, "Extensión no válida. Solo se aceptan archivos .csv"

    # 2. Leer contenido para verificar si tiene estructura de tabla
    contenido = file.getvalue().decode("utf-8", errors="ignore")
    
    # Validar que tenga separadores
    if "," not in contenido and ";" not in contenido and "\t" not in contenido:
        return None, "El archivo NO contiene separadores. Parece texto plano"

    try:
        # Intentar leer CSV separando por coma o punto y coma
        try:
            file.seek(0)
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            try:
                df = pd.read_csv(file, sep=";")
            except Exception:
                raise ArchivoInvalidoException(
                    "No se pudo leer el archivo. Asegúrate de que sea un CSV válido y no un TXT renombrado."
                )

        # 3. Validar columnas
        if df.empty or len(df.columns) == 0:
            raise TablaInvalidaException(
                "El archivo no contiene columnas. El CSV debe tener una estructura de tabla."
            )

        # 4. Validar filas (mínimo 1 fila de datos)
        if df.shape[0] == 0:
            raise TablaInvalidaException(
                "El archivo está vacío. Debe contener datos."
            )

        # 5. Validar que no sea texto plano disfrazado
        if len(df.columns) == 1:
            col = df.columns[0].lower()
            if df[col].dtype == object:
                # Contar filas que NO parecen valores tabulares
                if df[col].str.contains(r"[a-zA-Z]").sum() > 3:
                    raise ArchivoInvalidoException(
                        "El archivo cargado parece ser texto plano y NO una tabla."
                    )

        # 6. Detectar y manejar columnas completamente vacías
        columnas_vacias = df.columns[df.isnull().all()].tolist()
        if columnas_vacias:
            print(f"Advertencia: Columnas completamente vacías detectadas: {columnas_vacias}")
            # No las eliminamos automáticamente, solo informamos

        return df, None

    except (ArchivoInvalidoException, TablaInvalidaException) as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error desconocido: {str(e)}"