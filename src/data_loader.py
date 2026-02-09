import pandas as pd
import numpy as np
import os

def load_dataset(filepath):
    try:
        data = pd.read_csv(filepath)
        print(f"Dataset cargado: {data.shape[0]} filas, {data.shape[1]} columnas")
        return data
    except FileNotFoundError:
        print(f"Error: No se encontró {filepath}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def validate_dataset(data):
    if data is None:
        return False, "Dataset es None"
    
    if data.empty:
        return False, "Dataset está vacío"
    
    missing_values = data.isnull().sum().sum()
    if missing_values > 0:
        print(f"Advertencia: {missing_values} valores faltantes")
    
    return True, "Dataset válido"

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [30000, 35000, 40000, 45000, 50000],
        'department': ['IT', 'HR', 'IT', 'Sales', 'HR']
    })
    
    os.makedirs('data', exist_ok=True)
    sample_data.to_csv('data/sample_data.csv', index=False)
    
    data = load_dataset('data/sample_data.csv')
    is_valid, message = validate_dataset(data)
    print(f"Validación: {message}")

def show_basic_stats(data):
    """Muestra estadísticas básicas del dataset"""
    if data is not None and not data.empty:
        print("
Estadísticas básicas:")
        print(f"• Columnas: {list(data.columns)}")
        print(f"• Forma: {data.shape[0]} filas x {data.shape[1]} columnas")
        print(f"• Tipos de datos:")
        for col in data.columns:
            print(f"  - {col}: {data[col].dtype}")
        print(f"• Valores nulos: {data.isnull().sum().sum()}")
        print(f"
Primeras 3 filas:")
        print(data.head(3))
    else:
        print("Dataset vacío o no válido")

# Actualizar prueba
if __name__ == "__main__":
    # ... código anterior ...
    print("
--- Mostrando estadísticas ---")
    show_basic_stats(data)
