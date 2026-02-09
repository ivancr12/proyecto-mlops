import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_metrics(y_true, y_pred, model_name="Modelo"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print(f"\n=== Métricas de {model_name} ===")
    print(f"  • Accuracy:  {accuracy:.4f}")
    print(f"  • Precision: {precision:.4f}")
    print(f"  • Recall:    {recall:.4f}")
    print(f"  • F1-Score:  {f1:.4f}")
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name="Modelo", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"  • Matriz guardada en: {save_path}")
    
    plt.show()
    plt.close()

if __name__ == "__main__":
    print("=== Módulo de Evaluación ===")
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    
    metrics = calculate_metrics(y_true, y_pred, "Modelo de Prueba")
    
    print("\n--- Generando matriz de confusión ---")
    plot_confusion_matrix(y_true, y_pred, "Test Model", "reports/test_confusion_matrix.png")

def generate_evaluation_report(y_true, y_pred, model_name="Modelo", report_dir="reports"):
    """Genera un reporte completo de evaluación"""
    os.makedirs(report_dir, exist_ok=True)
    
    # Calcular métricas
    metrics = calculate_metrics(y_true, y_pred, model_name)
    
    # Guardar métricas en CSV
    metrics_df = pd.DataFrame([metrics])
    csv_path = os.path.join(report_dir, f"{model_name.lower()}_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"  • Métricas guardadas en: {csv_path}")
    
    # Generar matriz de confusión
    img_path = os.path.join(report_dir, f"{model_name.lower()}_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, model_name, img_path)
    
    return metrics_df

# Actualizar prueba
if __name__ == "__main__":
    # ... código anterior ...
    print("
--- Generando reporte completo ---")
    report = generate_evaluation_report(y_true, y_pred, "Modelo Completo")
    print("
Reporte generado exitosamente")
