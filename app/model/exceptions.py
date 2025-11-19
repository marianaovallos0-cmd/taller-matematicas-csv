class ArchivoInvalidoException(Exception):
    """Se lanza cuando el archivo no es CSV válido."""
    pass

class TablaInvalidaException(Exception):
    """Se lanza cuando el archivo no contiene una tabla con columnas."""
    pass