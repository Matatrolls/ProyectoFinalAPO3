import os
import json
import time
import joblib
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

from src.data.preprocess import FruitDatasetBuilder
from src.data.size_estimator import SizeEstimator
from src.models.traditional_models import create_random_forest, create_svm

def main():
    print("[train_traditional] Iniciando pipeline de entrenamiento tradicional...")
    
    # 1. Cargar el dataset y construir el inventario
    builder = FruitDatasetBuilder(balance=True)
    builder.build()
    
    # 2. Ajustar el Estimador de Tamaño Geométrico con el conjunto de entrenamiento
    print("[train_traditional] Ajustando estimador de tamaño geométrico...")
    size_estimator = SizeEstimator()
    size_estimator.fit(builder.train_inv)
    
    # 3. Extraer o cargar características tabulares (HSV + HOG)
    cache_dir = "data"
    cache_path = os.path.join(cache_dir, "features_tabular.npz")
    
    if os.path.exists(cache_path):
        print(f"[train_traditional] Cargando características tabulares desde caché: {cache_path}")
        data = np.load(cache_path)
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
    else:
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
        
    print(f"Dimensiones de los conjuntos:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}, y_val  : {y_val.shape}")
    print(f"  X_test : {X_test.shape}, y_test : {y_test.shape}")
    
    # Asegurar que existan directorios de salida
    os.makedirs("experiments/checkpoints", exist_ok=True)
    os.makedirs("experiments/results", exist_ok=True)
    
    results = {}
    
    # ─────────────────────────────────────────────
    # 4. Ajuste de hiperparámetros: Random Forest
    # ─────────────────────────────────────────────
    print("\n" + "="*50)
    print("ENTRENAMIENTO Y AJUSTE: RANDOM FOREST")
    print("="*50)
    
    rf_base = create_random_forest(random_state=42)
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None]
    }
    
    print(f"Ejecutando GridSearchCV (K=5) para Random Forest...")
    rf_grid = GridSearchCV(
        estimator=rf_base,
        param_grid=rf_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    t_start = time.time()
    rf_grid.fit(X_train, y_train)
    t_end = time.time()
    
    print(f"Random Forest entrenado en {t_end - t_start:.2f} segundos.")
    print(f"Mejores parámetros RF: {rf_grid.best_params_}")
    print(f"Mejor Accuracy en CV: {rf_grid.best_score_:.4f}")
    
    # Guardar mejor modelo RF
    best_rf = rf_grid.best_estimator_
    rf_checkpoint_path = "experiments/checkpoints/rf_best.joblib"
    joblib.dump(best_rf, rf_checkpoint_path)
    print(f"Modelo Random Forest guardado en: {rf_checkpoint_path}")
    
    # Evaluar RF en conjunto de validación
    rf_val_preds = best_rf.predict(X_val)
    rf_val_acc = accuracy_score(y_val, rf_val_preds)
    print(f"Accuracy de Random Forest en Validación: {rf_val_acc:.4f}")
    
    results['random_forest'] = {
        'best_params': rf_grid.best_params_,
        'best_cv_score': float(rf_grid.best_score_),
        'val_accuracy': float(rf_val_acc),
        'tuning_time_sec': t_end - t_start
    }
    
    # ─────────────────────────────────────────────
    # 5. Ajuste de hiperparámetros: SVM
    # ─────────────────────────────────────────────
    print("\n" + "="*50)
    print("ENTRENAMIENTO Y AJUSTE: SVM")
    print("="*50)
    
    # Reducimos un poco el dataset para el ajuste de SVM si es muy grande,
    # pero dado que corre en n_jobs=-1 e hilos nativos de C++, probamos primero con todo.
    # Si tarda demasiado, el usuario verá la barra de progreso de GridSearch.
    # NOTA: C=[0.1, 1, 10] y gamma=['scale'] es suficiente y rápido.
    svm_base = create_svm(random_state=42)
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
    
    print(f"Ejecutando GridSearchCV (K=5) para SVM...")
    svm_grid = GridSearchCV(
        estimator=svm_base,
        param_grid=svm_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    t_start = time.time()
    svm_grid.fit(X_train, y_train)
    t_end = time.time()
    
    print(f"SVM entrenado en {t_end - t_start:.2f} segundos.")
    print(f"Mejores parámetros SVM: {svm_grid.best_params_}")
    print(f"Mejor Accuracy en CV: {svm_grid.best_score_:.4f}")
    
    # Guardar mejor modelo SVM
    best_svm = svm_grid.best_estimator_
    svm_checkpoint_path = "experiments/checkpoints/svm_best.joblib"
    joblib.dump(best_svm, svm_checkpoint_path)
    print(f"Modelo SVM guardado en: {svm_checkpoint_path}")
    
    # Evaluar SVM en conjunto de validación
    svm_val_preds = best_svm.predict(X_val)
    svm_val_acc = accuracy_score(y_val, svm_val_preds)
    print(f"Accuracy de SVM en Validación: {svm_val_acc:.4f}")
    
    results['svm'] = {
        'best_params': svm_grid.best_params_,
        'best_cv_score': float(svm_grid.best_score_),
        'val_accuracy': float(svm_val_acc),
        'tuning_time_sec': t_end - t_start
    }
    
    # Guardar reporte de resultados
    results_path = "experiments/results/traditional_tuning.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"\nResultados de optimización guardados en: {results_path}")
    print("\n[train_traditional] Pipeline de Machine Learning Tradicional completado con éxito.")

if __name__ == "__main__":
    main()
