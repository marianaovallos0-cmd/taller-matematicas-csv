class ArchivoInvalidoException(Exception):
    """Se lanza cuando el archivo no es CSV o no tiene formato válido."""
    pass


class TablaInvalidaException(Exception):
    """Se lanza cuando el archivo está vacío o no contiene tabla (columnas/datos)."""
    pass
