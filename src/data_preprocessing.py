import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.dropna()
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
