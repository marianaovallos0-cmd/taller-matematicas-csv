from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd

def ancho_igual(df, bins=4):
    numeric = df.select_dtypes(include=['number'])
    enc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    df[numeric.columns] = enc.fit_transform(numeric)
    return df

def frecuencia_igual(df, bins=4):
    numeric = df.select_dtypes(include=['number'])
    enc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
    df[numeric.columns] = enc.fit_transform(numeric)
    return df
