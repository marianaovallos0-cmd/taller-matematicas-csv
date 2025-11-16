from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd

def z_score(df):
    scaler = StandardScaler()
    numeric = df.select_dtypes(include=['number'])
    df[numeric.columns] = scaler.fit_transform(numeric)
    return df

def min_max(df):
    scaler = MinMaxScaler()
    numeric = df.select_dtypes(include=['number'])
    df[numeric.columns] = scaler.fit_transform(numeric)
    return df

def logaritmica(df):
    numeric = df.select_dtypes(include=['number'])
    # usar log1p para no fallar con ceros
    df[numeric.columns] = numeric.apply(lambda x: np.log1p(x))
    return df
