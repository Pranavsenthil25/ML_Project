import numpy as np
import pandas as pd

def load_hour_data(path='Data/hour.csv'):
    """Load and return hour.csv dataset as pandas DataFrame."""
    return pd.read_csv(path)

def prepare_features(df):
    """Perform feature selection and preprocessing."""
    df = df.drop(columns=['instant', 'dteday', 'casual', 'registered'])
    df = pd.get_dummies(df, columns=['season', 'weathersit', 'mnth', 'hr', 'weekday'], drop_first=True)
    return df
