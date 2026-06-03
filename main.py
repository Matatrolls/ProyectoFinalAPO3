"""
main.py — Punto de Entrada del Pipeline de Clasificación de Calidad de Frutas
==============================================================================
Universidad Icesi · APO III · 2026-1

Este script actúa como interfaz unificada para ejecutar los tres modos del proyecto:

  1. Entrenamiento (train):
       Entrena los modelos de Machine Learning Tradicional (Random Forest, SVM)
       y la Red Neuronal Convolucional (CNN Multi-Head en PyTorch).

  2. Evaluación (evaluate):
       Evalúa los tres modelos sobre el conjunto de prueba independiente y genera:
         - Matrices de confusión (por fruta y por calidad)
         - Curvas de aprendizaje de la CNN
         - Resumen comparativo en JSON

  3. Despliegue (deploy):
       Lanza la aplicación web Flask + Socket.IO en http://localhost:5000
       que incluye el simulador de faja transportadora y diagnóstico en tiempo real.

Uso:
    python main.py --mode train
    python main.py --mode train --cnn_only       (solo CNN, si RF/SVM ya están entrenados)
    python main.py --mode evaluate
    python main.py --mode deploy
    python main.py --mode deploy --port 8080
"""

import argparse
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasificación de Calidad de Frutas — Pipeline Principal (APO III, Icesi 2026-1)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "deploy"],
        required=True,
        help=(
            "train    => Entrenamiento de modelos (RF + SVM + CNN)\n"
            "evaluate => Generacion de metricas y graficas\n"
            "deploy   => Lanzar aplicacion web Flask"
        )
    )
    parser.add_argument(
        "--cnn_only",
        action="store_true",
        default=False,
        help="En modo 'train': omitir RF/SVM y entrenar solo la CNN (útil si los checkpoints ya existen)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Número de épocas para el entrenamiento de la CNN (default: 10)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Tamaño de batch para la CNN (default: 32)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Puerto para la aplicación web en modo 'deploy' (default: 5000)."
    )
    return parser.parse_args()


def mode_train(args):
    """Ejecuta el pipeline de entrenamiento completo."""
    print("\n" + "=" * 60)
    print("MODO: ENTRENAMIENTO DE MODELOS")
    print("=" * 60)

    if not args.cnn_only:
        # Verificar si los modelos tradicionales ya existen
        rf_path = "experiments/checkpoints/rf_best.joblib"
        svm_path = "experiments/checkpoints/svm_best.joblib"

        if os.path.exists(rf_path) and os.path.exists(svm_path):
            print(f"[main] [OK] Checkpoints de RF y SVM ya encontrados. Saltando entrenamiento tradicional.")
            print(f"[main]    RF:  {rf_path}")
            print(f"[main]    SVM: {svm_path}")
            print("[main]    (Usa '--cnn_only' explícitamente para confirmar este comportamiento)")
        else:
            print("[main] 1/2 Entrenando modelos tradicionales (Random Forest, SVM)...")
            from src.training.train_traditional import main as train_trad
            train_trad()

    print("\n[main] Entrenando Red Neuronal Convolucional (CNN Multi-Head PyTorch)...")
    # Inyectar argumentos de la CNN al namespace de sys.argv para train_cnn
    sys.argv = [
        "train_cnn",
        f"--epochs={args.epochs}",
        f"--batch_size={args.batch_size}"
    ]
    from src.training.train_cnn import main as train_cnn
    train_cnn()

    print("\n[main] [OK] Pipeline de entrenamiento completado.")
    print("[main]    Checkpoints guardados en: experiments/checkpoints/")
    print("[main]    Logs de CNN en:           experiments/logs/cnn_training_log.csv")


def mode_evaluate():
    """Ejecuta la evaluación sobre el conjunto de prueba y genera gráficas."""
    print("\n" + "=" * 60)
    print("MODO: EVALUACIÓN DE MODELOS")
    print("=" * 60)

    from src.evaluation.evaluate import main as run_evaluation
    run_evaluation()

    print("\n[main] [OK] Evaluación completada.")
    print("[main]    Resultados en: experiments/results/")


def mode_deploy(port: int = 5000):
    """Lanza la aplicación web Flask con Socket.IO."""
    print("\n" + "=" * 60)
    print("MODO: DESPLIEGUE DE APLICACIÓN WEB")
    print("=" * 60)
    print(f"[main] Iniciando servidor Flask en http://localhost:{port}")
    print("[main] Presiona Ctrl+C para detener el servidor.\n")

    from src.deployment.app import app, socketio, load_models
    load_models()
    socketio.run(app, debug=False, host="0.0.0.0", port=port)


def main():
    args = parse_args()

    if args.mode == "train":
        mode_train(args)
    elif args.mode == "evaluate":
        mode_evaluate()
    elif args.mode == "deploy":
        mode_deploy(port=args.port)


if __name__ == "__main__":
    main()
