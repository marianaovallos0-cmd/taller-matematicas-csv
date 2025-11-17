from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd

def z_score(df):
    scaler = StandardScaler()
    numeric = df.select_dtypes(include=['number'])
    df[numeric.columns] = scaler.fit_transform(numeric)
    return df

def min_max(df, min_nuevo=0, max_nuevo=1):
    numeric = df.select_dtypes(include=['number'])

    min_actual = numeric.min()
    max_actual = numeric.max()

    df[numeric.columns] = ((numeric - min_actual) / (max_actual - min_actual)) * (max_nuevo - min_nuevo) + min_nuevo

    return df

def logaritmica(df):
    numeric = df.select_dtypes(include=['number'])
    df[numeric.columns] = numeric.apply(lambda x: np.log1p(x))
    return df
