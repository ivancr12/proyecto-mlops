import pandas as pd
import numpy as np

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

# Función adicional para mostrar estadísticas básicas
def show_basic_stats(data):
    """Muestra estadísticas básicas del dataset"""
    if data is not None and not data.empty:
        print("\nEstadísticas básicas del dataset:")
        print(f"• Columnas: {list(data.columns)}")
        print(f"• Forma: {data.shape[0]} filas x {data.shape[1]} columnas")
        print(f"• Tipos de datos:")
        for col in data.columns:
            print(f"  - {col}: {data[col].dtype}")
        print(f"• Valores nulos totales: {data.isnull().sum().sum()}")
        
        # Mostrar algunas filas de ejemplo
        print(f"\nPrimeras 3 filas:")
        print(data.head(3))
    else:
        print("Dataset vacío o no válido")

if __name__ == "__main__":
    # Crear datos de ejemplo más completos
    sample_data = pd.DataFrame({
        'age': [25, 30, None, 40, 45, 50, 35],
        'income': [30000, 35000, 40000, 45000, None, 55000, 60000],
        'department': ['IT', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales'],
        'experience_years': [2, 5, 8, 3, 10, 12, 6]
    })
    
    # Asegurar que existe el directorio data
    import os
    os.makedirs('data', exist_ok=True)
    
    # Guardar dataset de ejemplo
    sample_data.to_csv('data/sample_data.csv', index=False)
    print("Dataset de ejemplo creado en 'data/sample_data.csv'")
    
    # Probar funciones
    data = load_dataset('data/sample_data.csv')
    is_valid, message = validate_dataset(data)
    print(f"\nValidación: {message}")
    
    # Mostrar estadísticas
    if is_valid:
        show_basic_stats(data)
