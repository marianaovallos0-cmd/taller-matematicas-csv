import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

# ============================================================
#  FUNCIÓN PARA LIMPIAR NA ANTES DE DISCRETIZAR
# ============================================================

def limpiar_numericos(df):
    """
    Rellena valores faltantes solo en columnas numéricas
    para evitar errores en KBinsDiscretizer.
    """
    numeric = df.select_dtypes(include=["number"]).columns
    df[numeric] = df[numeric].fillna(df[numeric].median())
    return df

def discretizar_ancho_igual(df, bins=4):
    df = limpiar_numericos(df)   # <--- SOLUCIÓN AL ERROR
    numeric = df.select_dtypes(include=["number"])
    enc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    df[numeric.columns] = enc.fit_transform(numeric)
    return df


def discretizar_frecuencia_igual(df, bins=4):
    df = limpiar_numericos(df)   # <--- SOLUCIÓN AL ERROR
    numeric = df.select_dtypes(include=["number"])
    enc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
    df[numeric.columns] = enc.fit_transform(numeric)
    return df



def chi_square(a, b):
    total = a.sum() + b.sum()
    expected_a = (a.sum() / total) * (a + b)
    expected_b = (b.sum() / total) * (a + b)

    expected_a[expected_a == 0] = 1e-9
    expected_b[expected_b == 0] = 1e-9

    chi = ((a - expected_a) ** 2 / expected_a).sum() + \
          ((b - expected_b) ** 2 / expected_b).sum()
    return chi


def chimerge_column(col, target, max_bins=4):
    df = pd.DataFrame({"X": col, "Y": target})
    df = df.sort_values("X")

    intervals = []
    for value in df["X"].unique():
        subset = df[df["X"] == value]["Y"]
        counts = subset.value_counts().reindex(df["Y"].unique(), fill_value=0)
        intervals.append(counts.values)

    intervals = [np.array(i) for i in intervals]

    while len(intervals) > max_bins:
        chi_values = [
            chi_square(intervals[i], intervals[i+1])
            for i in range(len(intervals) - 1)
        ]
        min_index = np.argmin(chi_values)

        merged = intervals[min_index] + intervals[min_index+1]
        intervals[min_index:min_index+2] = [merged]

    return pd.cut(col, bins=len(intervals), labels=list(range(len(intervals))))


def discretizar_chimerge(df, target_column, bins=4):
    if target_column not in df.columns:
        raise Exception("Debes seleccionar la columna objetivo para ChiMerge.")

    df = limpiar_numericos(df)  # También limpiamos antes

    numeric = df.select_dtypes(include=["number"]).columns

    for col in numeric:
        if col != target_column:
            df[col] = chimerge_column(df[col], df[target_column], max_bins=bins)

    return df
