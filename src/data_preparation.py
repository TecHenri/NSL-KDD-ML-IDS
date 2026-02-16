import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(filepath):

    # Charger dataset
    df = pd.read_csv(filepath)

    # Supprimer colonne level (éviter data leakage)
    if 'level' in df.columns:
        df = df.drop(columns=['level'])

    # Supprimer doublons
    df.drop_duplicates(inplace=True)

    # Encodage des variables catégorielles
    categorical_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=categorical_cols)

    # Séparation X / y
    X = df.drop('outcome', axis=1)
    y = df['outcome'].apply(lambda x: 0 if x == 'normal' else 1)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    # Standardisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
