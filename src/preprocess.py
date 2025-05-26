"""
Data loading and preprocessing utilities.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RNG = 42


def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """
    Reads the CSV dataset and returns a DataFrame.
    """
    return pd.read_csv(path)


def train_test_split_scaled(df: pd.DataFrame, test_size: float = 0.2):
    """
    Splits features/labels, scales features, and returns X_train, X_test, y_train, y_test.
    Assumes a `Class` column where 1 = fraud.
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RNG, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
