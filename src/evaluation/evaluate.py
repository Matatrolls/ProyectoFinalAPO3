import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.data.preprocess import FruitDatasetBuilder, FRUIT_CLASSES, QUALITY_CLASSES
from src.models.cnn_model import MultiHeadCNN

def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    """Genera y guarda una matriz de confusión formateada con seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    # Evitar división por cero en clases vacías
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, cbar=True)
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[evaluate] Matriz de confusión guardada: {save_path}")

def plot_learning_curves(log_csv_path, save_path):
    """Lee el log de entrenamiento de la CNN y genera curvas de pérdida y precisión."""
    if not os.path.exists(log_csv_path):
        print(f"[evaluate] No se encontró el log CSV de la CNN en: {log_csv_path}")
        return
        
    epochs = []
    train_loss, val_loss = [], []
    train_acc_f, val_acc_f = [], []
    train_acc_q, val_acc_q = [], []
    
    with open(log_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            train_acc_f.append(float(row['train_acc_fruit']))
            val_acc_f.append(float(row['val_acc_fruit']))
            train_acc_q.append(float(row['train_acc_quality']))
            val_acc_q.append(float(row['val_acc_quality']))
            
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Pérdida (Loss)
    axes[0].plot(epochs, train_loss, label='Train Loss', color='blue', marker='o')
    axes[0].plot(epochs, val_loss, label='Val Loss', color='orange', marker='s')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Pérdida')
    axes[0].set_title('Curva de Pérdida (Cross-Entropy)')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Precisión Tipo de Fruta
    axes[1].plot(epochs, [a*100 for a in train_acc_f], label='Train Acc', color='green', marker='o')
    axes[1].plot(epochs, [a*100 for a in val_acc_f], label='Val Acc', color='red', marker='s')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Precisión (%)')
    axes[1].set_title('Precisión - Tipo de Fruta')
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. Precisión Calidad
    axes[2].plot(epochs, [a*100 for a in train_acc_q], label='Train Acc', color='purple', marker='o')
    axes[2].plot(epochs, [a*100 for a in val_acc_q], label='Val Acc', color='brown', marker='s')
    axes[2].set_xlabel('Época')
    axes[2].set_ylabel('Precisión (%)')
    axes[2].set_title('Precisión - Calidad')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.suptitle('Curvas de Aprendizaje de la CNN Multi-Head', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[evaluate] Curvas de aprendizaje guardadas: {save_path}")

def plot_comparison_chart(summary, save_path):
    """Genera un gráfico de barras comparativo de precisión entre los modelos."""
    if not summary:
        print("[evaluate] Sin datos en summary para generar gráfico comparativo.")
        return
        
    models = list(summary.keys())
    fruit_accs = [summary[m].get('accuracy_fruit', 0.0) * 100 for m in models]
    quality_accs = [summary[m].get('accuracy_quality', 0.0) * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, fruit_accs, width, label='Exactitud Fruta', color='#6366f1')
    rects2 = ax.bar(x + width/2, quality_accs, width, label='Exactitud Calidad', color='#10b981')
    
    ax.set_ylabel('Exactitud (%)')
    ax.set_title('Comparativa de Modelos: Fruta vs Calidad')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').upper() for m in models])
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Añadir etiquetas de porcentaje arriba de cada barra
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[evaluate] Gráfico comparativo de modelos guardado en: {save_path}")

def main():
    print("\n" + "="*50)
    print("PIPELINE DE EVALUACIÓN GLOBAL")
    print("="*50)
    
    # 1. Cargar el dataset de prueba
    builder = FruitDatasetBuilder(balance=True)
    builder.build()
    
    # 2. Cargar características de prueba desde caché
    cache_path = "data/features_tabular.npz"
    if not os.path.exists(cache_path):
        print(f"[evaluate] ERROR: No se encontró el caché de características en {cache_path}. Corre el entrenamiento tradicional primero.")
        return
        
    data = np.load(cache_path)
    X_test, y_test_combined = data['X_test'], data['y_test']
    
    # Decodificar etiquetas combinadas a fruta y calidad individuales
    # combinado = fruit_idx * 3 + quality_idx
    y_test_fruit = y_test_combined // len(QUALITY_CLASSES)
    y_test_quality = y_test_combined % len(QUALITY_CLASSES)
    
    evaluation_summary = {}
    
    # ─────────────────────────────────────────────
    # EVALUACIÓN: RANDOM FOREST
    # ─────────────────────────────────────────────
    rf_path = "experiments/checkpoints/rf_best.joblib"
    if os.path.exists(rf_path):
        print("\n--- Evaluando Random Forest ---")
        rf_model = joblib.load(rf_path)
        
        # Predicción combinada
        rf_preds_comb = rf_model.predict(X_test)
        rf_probs = rf_model.predict_proba(X_test)
        
        rf_preds_fruit = rf_preds_comb // len(QUALITY_CLASSES)
        rf_preds_quality = rf_preds_comb % len(QUALITY_CLASSES)
        
        # Filtro OOD (Umbral de confianza)
        rf_max_probs = np.max(rf_probs, axis=1)
        rf_known_mask = rf_max_probs >= 0.70
        rf_rejection_rate = (1.0 - np.mean(rf_known_mask)) * 100
        
        # Evaluar
        rf_acc_fruit = accuracy_score(y_test_fruit, rf_preds_fruit)
        rf_acc_quality = accuracy_score(y_test_quality, rf_preds_quality)
        
        print(f"Accuracy Fruta  : {rf_acc_fruit*100:.2f}%")
        print(f"Accuracy Calidad: {rf_acc_quality*100:.2f}%")
        print(f"Muestras Rechazadas (Confianza < 70%): {rf_rejection_rate:.2f}%")
        
        # Generar Matrices de Confusión
        plot_confusion_matrix(
            y_test_fruit, rf_preds_fruit, FRUIT_CLASSES,
            "Matriz de Confusión - Random Forest (Fruta)",
            "experiments/results/confusion_matrix_rf_fruit.png"
        )
        plot_confusion_matrix(
            y_test_quality, rf_preds_quality, QUALITY_CLASSES,
            "Matriz de Confusión - Random Forest (Calidad)",
            "experiments/results/confusion_matrix_rf_quality.png"
        )
        
        evaluation_summary['random_forest'] = {
            'accuracy_fruit': float(rf_acc_fruit),
            'accuracy_quality': float(rf_acc_quality),
            'rejection_rate': float(rf_rejection_rate)
        }
    else:
        print(f"[evaluate] No se encontró el modelo Random Forest en {rf_path}")
        
    # ─────────────────────────────────────────────
    # EVALUACIÓN: SVM
    # ─────────────────────────────────────────────
    svm_path = "experiments/checkpoints/svm_best.joblib"
    if os.path.exists(svm_path):
        print("\n--- Evaluando SVM ---")
        svm_model = joblib.load(svm_path)
        
        svm_preds_comb = svm_model.predict(X_test)
        svm_probs = svm_model.predict_proba(X_test)
        
        svm_preds_fruit = svm_preds_comb // len(QUALITY_CLASSES)
        svm_preds_quality = svm_preds_comb % len(QUALITY_CLASSES)
        
        # Filtro OOD
        svm_max_probs = np.max(svm_probs, axis=1)
        svm_known_mask = svm_max_probs >= 0.70
        svm_rejection_rate = (1.0 - np.mean(svm_known_mask)) * 100
        
        svm_acc_fruit = accuracy_score(y_test_fruit, svm_preds_fruit)
        svm_acc_quality = accuracy_score(y_test_quality, svm_preds_quality)
        
        print(f"Accuracy Fruta  : {svm_acc_fruit*100:.2f}%")
        print(f"Accuracy Calidad: {svm_acc_quality*100:.2f}%")
        print(f"Muestras Rechazadas (Confianza < 70%): {svm_rejection_rate:.2f}%")
        
        plot_confusion_matrix(
            y_test_fruit, svm_preds_fruit, FRUIT_CLASSES,
            "Matriz de Confusión - SVM (Fruta)",
            "experiments/results/confusion_matrix_svm_fruit.png"
        )
        plot_confusion_matrix(
            y_test_quality, svm_preds_quality, QUALITY_CLASSES,
            "Matriz de Confusión - SVM (Calidad)",
            "experiments/results/confusion_matrix_svm_quality.png"
        )
        
        evaluation_summary['svm'] = {
            'accuracy_fruit': float(svm_acc_fruit),
            'accuracy_quality': float(svm_acc_quality),
            'rejection_rate': float(svm_rejection_rate)
        }
    else:
        print(f"[evaluate] No se encontró el modelo SVM en {svm_path}")
        
    # ─────────────────────────────────────────────
    # EVALUACIÓN: CNN PyTorch
    # ─────────────────────────────────────────────
    cnn_path = "experiments/checkpoints/best_cnn_model.pth"
    if os.path.exists(cnn_path):
        print("\n--- Evaluando CNN Multi-Head ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Re-construir dataloader de test
        _, _, test_loader = builder.get_dataloaders(batch_size=32)
        
        # Cargar modelo
        model = MultiHeadCNN(num_fruits=len(FRUIT_CLASSES), num_qualities=len(QUALITY_CLASSES)).to(device)
        model.load_state_dict(torch.load(cnn_path, map_location=device))
        model.eval()
        
        cnn_preds_f, cnn_preds_q = [], []
        cnn_probs_f, cnn_probs_q = [], []
        cnn_true_f, cnn_true_q = [], []
        
        with torch.no_grad():
            for imgs, fruit_labels, quality_labels in test_loader:
                imgs = imgs.to(device)
                out_fruit, out_quality = model(imgs)
                
                # Obtener probabilidades
                prob_f = F.softmax(out_fruit, dim=1)
                prob_q = F.softmax(out_quality, dim=1)
                
                max_prob_f, pred_f = torch.max(prob_f, dim=1)
                max_prob_q, pred_q = torch.max(prob_q, dim=1)
                
                cnn_preds_f.extend(pred_f.cpu().numpy())
                cnn_preds_q.extend(pred_q.cpu().numpy())
                cnn_probs_f.extend(max_prob_f.cpu().numpy())
                cnn_probs_q.extend(max_prob_q.cpu().numpy())
                
                cnn_true_f.extend(fruit_labels.numpy())
                cnn_true_q.extend(quality_labels.numpy())
                
        cnn_preds_f = np.array(cnn_preds_f)
        cnn_preds_q = np.array(cnn_preds_q)
        cnn_probs_f = np.array(cnn_probs_f)
        cnn_probs_q = np.array(cnn_probs_q)
        cnn_true_f = np.array(cnn_true_f)
        cnn_true_q = np.array(cnn_true_q)
        
        # Filtro OOD en CNN
        # Si la confianza de fruta < 70%, clasificamos como fuera de distribución (Desconocido)
        cnn_known_mask = cnn_probs_f >= 0.70
        cnn_rejection_rate = (1.0 - np.mean(cnn_known_mask)) * 100
        
        cnn_acc_fruit = accuracy_score(cnn_true_f, cnn_preds_f)
        cnn_acc_quality = accuracy_score(cnn_true_q, cnn_preds_q)
        
        print(f"Accuracy Fruta  : {cnn_acc_fruit*100:.2f}%")
        print(f"Accuracy Calidad: {cnn_acc_quality*100:.2f}%")
        print(f"Muestras Rechazadas (Confianza < 70%): {cnn_rejection_rate:.2f}%")
        
        plot_confusion_matrix(
            cnn_true_f, cnn_preds_f, FRUIT_CLASSES,
            "Matriz de Confusión - CNN Multi-Head (Fruta)",
            "experiments/results/confusion_matrix_cnn_fruit.png"
        )
        plot_confusion_matrix(
            cnn_true_q, cnn_preds_q, QUALITY_CLASSES,
            "Matriz de Confusión - CNN Multi-Head (Calidad)",
            "experiments/results/confusion_matrix_cnn_quality.png"
        )
        
        # Graficar curvas de aprendizaje
        plot_learning_curves(
            "experiments/logs/cnn_training_log.csv",
            "experiments/results/cnn_learning_curves.png"
        )
        
        evaluation_summary['cnn'] = {
            'accuracy_fruit': float(cnn_acc_fruit),
            'accuracy_quality': float(cnn_acc_quality),
            'rejection_rate': float(cnn_rejection_rate)
        }
    else:
        print(f"[evaluate] No se encontró el modelo CNN en {cnn_path}")
        
    # Generar gráfico comparativo de modelos
    if len(evaluation_summary) > 0:
        plot_comparison_chart(evaluation_summary, "experiments/results/model_comparison.png")

    # Guardar reporte consolidado
    summary_path = "experiments/results/evaluation_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, indent=4)
    print(f"\n[evaluate] Resumen de evaluación guardado en: {summary_path}")

if __name__ == "__main__":
    main()
