import os
import json
import time
import joblib
import numpy as np

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score

from src.data.preprocess import FruitDatasetBuilder
from src.data.size_estimator import SizeEstimator
from src.models.traditional_models import create_random_forest, create_svm

def main():
    print("[train_traditional] Iniciando pipeline de entrenamiento tradicional...")
    
    # 1. Cargar el dataset y construir el inventario
    builder = FruitDatasetBuilder(balance=True)
    builder.build()
    
    # 2. Ajustar el Estimador de Tamaño Geométrico con el conjunto de entrenamiento (usando subconjunto del 25% para agilizar)
    print("[train_traditional] Ajustando estimador de tamaño geométrico (usando subconjunto del 25% para agilizar)...")
    train_labels_for_size = [item["fruit"] for item in builder.train_inv]
    train_inv_sub, _ = train_test_split(
        builder.train_inv,
        test_size=0.75,
        stratify=train_labels_for_size,
        random_state=42
    )
    size_estimator = SizeEstimator()
    size_estimator.fit(train_inv_sub)
    
    # 3. Extraer o cargar características tabulares (HSV + HOG)
    cache_dir = "data"
    cache_path = os.path.join(cache_dir, "features_tabular.npz")
    
    cache_valid = False
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            X_train, y_train = data['X_train'], data['y_train']
            X_val, y_val = data['X_val'], data['y_val']
            X_test, y_test = data['X_test'], data['y_test']
            
            total_cache_samples = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
            total_current_samples = len(builder.inventory_full)
            
            if total_cache_samples == total_current_samples:
                print(f"[train_traditional] Cargando características tabulares desde caché: {cache_path}")
                cache_valid = True
            else:
                print(f"[train_traditional] La caché ({total_cache_samples} muestras) no coincide con el inventario actual ({total_current_samples} muestras). Regenerando...")
        except Exception as e:
            print(f"[train_traditional] Error leyendo caché ({e}). Regenerando...")
            
    if not cache_valid:
        print("[train_traditional] Extrayendo características tabulares (HSV + HOG) para todo el dataset...")
        t_start = time.time()
        # label="combined" para clasificación conjunta de (fruta + calidad)
        X_train, X_val, X_test, y_train, y_val, y_test = builder.get_tabular_splits(label="combined")
        t_end = time.time()
        print(f"[train_traditional] Extracción completada en {t_end - t_start:.2f} segundos.")
        
        # Guardar en caché
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache_path, X_train=X_train, y_train=y_train, 
                            X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
        print(f"[train_traditional] Características guardadas en caché: {cache_path}")
        
    print(f"Dimensiones originales:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}, y_val  : {y_val.shape}")
    
    # Submuestrear al 25% para agilizar entrenamiento tradicional
    print("[train_traditional] Reduciendo conjunto de entrenamiento y validación al 25% para acelerar el ajuste en CPU...")
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train,
        test_size=0.75,
        stratify=y_train,
        random_state=42
    )
    X_val, _, y_val, _ = train_test_split(
        X_val, y_val,
        test_size=0.75,
        stratify=y_val,
        random_state=42
    )
    
    print(f"Nuevas dimensiones de entrenamiento:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}, y_val  : {y_val.shape}")
    print(f"  X_test : {X_test.shape}, y_test : {y_test.shape}")
    
    # Asegurar que existan directorios de salida
    os.makedirs("experiments/checkpoints", exist_ok=True)
    os.makedirs("experiments/results", exist_ok=True)
    
    results = {}
    
    # ─────────────────────────────────────────────
    # 4. Entrenamiento: Random Forest
    # ─────────────────────────────────────────────
    print("\n" + "="*50)
    print("ENTRENAMIENTO: RANDOM FOREST (Parámetros óptimos)")
    print("="*50)
    
    best_rf = create_random_forest(n_estimators=200, max_depth=None, random_state=42)
    
    t_start = time.time()
    best_rf.fit(X_train, y_train)
    t_end = time.time()
    
    print(f"Random Forest entrenado en {t_end - t_start:.2f} segundos.")
    
    # Guardar modelo RF
    rf_checkpoint_path = "experiments/checkpoints/rf_best.joblib"
    joblib.dump(best_rf, rf_checkpoint_path)
    print(f"Modelo Random Forest guardado en: {rf_checkpoint_path}")
    
    # Evaluar RF en conjunto de validación
    rf_val_preds = best_rf.predict(X_val)
    rf_val_acc = accuracy_score(y_val, rf_val_preds)
    print(f"Accuracy de Random Forest en Validación: {rf_val_acc:.4f}")
    
    results['random_forest'] = {
        'best_params': {'n_estimators': 200, 'max_depth': None},
        'val_accuracy': float(rf_val_acc),
        'training_time_sec': t_end - t_start
    }
    
    # ─────────────────────────────────────────────
    # 5. Entrenamiento: SVM
    # ─────────────────────────────────────────────
    print("\n" + "="*50)
    print("ENTRENAMIENTO: SVM (Parámetros óptimos)")
    print("="*50)
    
    best_svm = create_svm(C=10, gamma='scale', random_state=42)
    
    t_start = time.time()
    best_svm.fit(X_train, y_train)
    t_end = time.time()
    
    print(f"SVM entrenado en {t_end - t_start:.2f} segundos.")
    
    # Guardar modelo SVM
    svm_checkpoint_path = "experiments/checkpoints/svm_best.joblib"
    joblib.dump(best_svm, svm_checkpoint_path)
    print(f"Modelo SVM guardado en: {svm_checkpoint_path}")
    
    # Evaluar SVM en conjunto de validación
    svm_val_preds = best_svm.predict(X_val)
    svm_val_acc = accuracy_score(y_val, svm_val_preds)
    print(f"Accuracy de SVM en Validación: {svm_val_acc:.4f}")
    
    results['svm'] = {
        'best_params': {'C': 10, 'gamma': 'scale'},
        'val_accuracy': float(svm_val_acc),
        'training_time_sec': t_end - t_start
    }
    
    # Guardar reporte de resultados
    results_path = "experiments/results/traditional_tuning.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"\nResultados de optimización guardados en: {results_path}")
    print("\n[train_traditional] Pipeline de Machine Learning Tradicional completado con éxito.")

if __name__ == "__main__":
    main()
