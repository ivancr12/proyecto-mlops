import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_data(data, strategy='drop'):
    if data is None or data.empty:
        print("Error: Dataset vacío o None")
        return data
    
    original_rows = data.shape[0]
    
    if strategy == 'drop':
        cleaned_data = data.dropna()
        removed_rows = original_rows - cleaned_data.shape[0]
        if removed_rows > 0:
            print(f"Se eliminaron {removed_rows} filas con valores nulos")
    elif strategy == 'mean':
        cleaned_data = data.copy()
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_data[col].isnull().any():
                mean_val = cleaned_data[col].mean()
                cleaned_data[col].fillna(mean_val, inplace=True)
                print(f"Columna '{col}': valores nulos reemplazados con media {mean_val:.2f}")
    else:
        print(f"Estrategia '{strategy}' no reconocida.")
        cleaned_data = data
    
    return cleaned_data

def encode_categorical(data, columns=None):
    encoded_data = data.copy()
    encoders = {}
    
    if columns is None:
        categorical_cols = encoded_data.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        categorical_cols = [col for col in columns if col in encoded_data.columns]
    
    for column in categorical_cols:
        le = LabelEncoder()
        encoded_data[column] = le.fit_transform(encoded_data[column].astype(str))
        encoders[column] = le
        print(f"Columna '{column}' codificada ({len(le.classes_)} categorías)")
    
    return encoded_data, encoders

def scale_features(data, columns=None):
    scaled_data = data.copy()
    
    if columns is None:
        numeric_cols = scaled_data.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if col in scaled_data.columns]
    
    if numeric_cols:
        scaler = StandardScaler()
        scaled_data[numeric_cols] = scaler.fit_transform(scaled_data[numeric_cols])
        print(f"Características escaladas: {numeric_cols}")
        return scaled_data, scaler
    else:
        print("No hay columnas numéricas para escalar")
        return scaled_data, None

if __name__ == "__main__":
    print("=== Módulo de Preprocesamiento ===")
    test_data = pd.DataFrame({
        'age': [25, 30, None, 40, 45],
        'income': [30000, None, 40000, 45000, 50000],
        'department': ['IT', 'HR', 'IT', 'Sales', 'HR'],
        'target': [0, 1, 0, 1, 0]
    })
    
    print("\n1. Datos originales:")
    print(test_data)
    
    print("\n2. Limpieza (eliminar nulos):")
    cleaned = clean_data(test_data, strategy='drop')
    print(cleaned)
