import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def split_train_test(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"División de datos:")
    print(f"  • Entrenamiento: {X_train.shape[0]} muestras")
    print(f"  • Prueba: {X_test.shape[0]} muestras")
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    print("\n=== Entrenando Regresión Logística ===")
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    print(f"  • Modelo entrenado exitosamente")
    print(f"  • Coeficientes: {len(model.coef_[0])}")
    
    return model

def train_random_forest(X_train, y_train, n_estimators=100):
    print("\n=== Entrenando Random Forest ===")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print(f"  • Random Forest entrenado con {n_estimators} árboles")
    return model

def save_model(model, model_name, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model, filepath)
    print(f"  • Modelo guardado en: {filepath}")
    return filepath

if __name__ == "__main__":
    print("=== Módulo de Entrenamiento ===")
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.randint(0, 2, 100)
    
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    print("\n--- Entrenando Regresión Logística ---")
    lr_model = train_logistic_regression(X_train, y_train)
    save_model(lr_model, "logistic_regression_model")
    
    print("\n--- Entrenando Random Forest ---")
    rf_model = train_random_forest(X_train, y_train, n_estimators=50)
    save_model(rf_model, "random_forest_model")
