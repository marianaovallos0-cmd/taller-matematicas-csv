# Archivo vacío o con utilidades pequeñas.
# Por ahora lo dejamos preparado por si quieres añadir funciones de ayuda.
def resumen_dataframe(df, n=5):
    """
    Devuelve un resumen sencillo: primeras n filas y descripción.
    """
    return df.head(n), df.describe(include='all')
