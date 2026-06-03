import os
import time
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.preprocess import FruitDatasetBuilder
from src.models.cnn_model import MultiHeadCNN

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento de la CNN Multi-Head en PyTorch")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de batch para el entrenamiento")
    parser.add_argument("--lr", type=float, default=0.001, help="Tasa de aprendizaje (learning rate)")
    parser.add_argument("--patience", type=int, default=3, help="Paciencia para ReduceLROnPlateau y Early Stopping")
    return parser.parse_args()

def train_one_epoch(model, dataloader, criterion_fruit, criterion_quality, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_fruit = 0
    correct_quality = 0
    total = 0
    
    for batch_idx, (imgs, fruit_labels, quality_labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        fruit_labels = fruit_labels.to(device)
        quality_labels = quality_labels.to(device)
        
        optimizer.zero_grad()
        out_fruit, out_quality = model(imgs)
        
        loss_fruit = criterion_fruit(out_fruit, fruit_labels)
        loss_quality = criterion_quality(out_quality, quality_labels)
        loss = loss_fruit + loss_quality
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        
        _, pred_fruit = torch.max(out_fruit, 1)
        _, pred_quality = torch.max(out_quality, 1)
        
        correct_fruit += (pred_fruit == fruit_labels).sum().item()
        correct_quality += (pred_quality == quality_labels).sum().item()
        total += imgs.size(0)
        
    epoch_loss = running_loss / total
    acc_fruit = correct_fruit / total
    acc_quality = correct_quality / total
    
    return epoch_loss, acc_fruit, acc_quality

def validate(model, dataloader, criterion_fruit, criterion_quality, device):
    model.eval()
    running_loss = 0.0
    correct_fruit = 0
    correct_quality = 0
    total = 0
    
    with torch.no_grad():
        for imgs, fruit_labels, quality_labels in dataloader:
            imgs = imgs.to(device)
            fruit_labels = fruit_labels.to(device)
            quality_labels = quality_labels.to(device)
            
            out_fruit, out_quality = model(imgs)
            
            loss_fruit = criterion_fruit(out_fruit, fruit_labels)
            loss_quality = criterion_quality(out_quality, quality_labels)
            loss = loss_fruit + loss_quality
            
            running_loss += loss.item() * imgs.size(0)
            
            _, pred_fruit = torch.max(out_fruit, 1)
            _, pred_quality = torch.max(out_quality, 1)
            
            correct_fruit += (pred_fruit == fruit_labels).sum().item()
            correct_quality += (pred_quality == quality_labels).sum().item()
            total += imgs.size(0)
            
    epoch_loss = running_loss / total
    acc_fruit = correct_fruit / total
    acc_quality = correct_quality / total
    
    return epoch_loss, acc_fruit, acc_quality

def main():
    args = parse_args()
    
    print("\n" + "="*50)
    print("PIPELINE DE ENTRENAMIENTO: CNN MULTI-HEAD")
    print("="*50)
    
    # 1. Configurar dispositivo (GPU si está disponible, si no CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_cnn] Dispositivo de entrenamiento: {device}")
    
    # 2. Cargar el dataset y obtener los Dataloaders
    builder = FruitDatasetBuilder(balance=True)
    builder.build()
    
    if device.type == "cpu":
        print("[train_cnn] Detectado entorno CPU. Reduciendo conjunto de entrenamiento y validación al 25% para agilizar.")
        from sklearn.model_selection import train_test_split
        # Submuestrear entrenamiento
        train_labels = [item["quality_idx"] for item in builder.train_inv]
        train_sub, _ = train_test_split(
            builder.train_inv,
            test_size=0.75,
            stratify=train_labels,
            random_state=42
        )
        builder.train_inv = train_sub
        
        # Submuestrear validación
        val_labels = [item["quality_idx"] for item in builder.val_inv]
        val_sub, _ = train_test_split(
            builder.val_inv,
            test_size=0.75,
            stratify=val_labels,
            random_state=42
        )
        builder.val_inv = val_sub
        print(f"[train_cnn] Nuevo tamaño - Entrenamiento: {len(builder.train_inv)} imágenes | Validación: {len(builder.val_inv)} imágenes")
    
    class_info = builder.get_class_info()
    num_fruits = class_info["n_fruits"]
    num_qualities = class_info["n_qualities"]
    
    train_loader, val_loader, _ = builder.get_dataloaders(batch_size=args.batch_size)
    
    # 3. Inicializar el modelo
    model = MultiHeadCNN(num_fruits=num_fruits, num_qualities=num_qualities).to(device)
    
    # 4. Definir Pérdidas, Optimizador y Scheduler
    criterion_fruit = nn.CrossEntropyLoss()
    criterion_quality = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.patience-1, factor=0.5
    )
    
    # Preparar directorios de salida
    os.makedirs("experiments/logs", exist_ok=True)
    os.makedirs("experiments/checkpoints", exist_ok=True)
    
    log_csv_path = "experiments/logs/cnn_training_log.csv"
    best_model_path = "experiments/checkpoints/best_cnn_model.pth"
    
    # Inicializar archivo de logs CSV
    with open(log_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc_fruit", "train_acc_quality",
            "val_loss", "val_acc_fruit", "val_acc_quality", "lr"
        ])
        
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience_limit = args.patience + 2 # Margen para early stopping
    
    # 5. Ciclo de Entrenamiento
    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        
        # Entrenamiento
        train_loss, train_acc_f, train_acc_q = train_one_epoch(
            model, train_loader, criterion_fruit, criterion_quality, optimizer, device
        )
        
        # Validación
        val_loss, val_acc_f, val_acc_q = validate(
            model, val_loader, criterion_fruit, criterion_quality, device
        )
        
        t_end = time.time()
        epoch_time = t_end - t_start
        
        # Obtener tasa de aprendizaje actual
        current_lr = optimizer.param_groups[0]['lr']
        
        # Actualizar scheduler
        scheduler.step(val_loss)
        
        # Mostrar métricas
        print(f"\nÉpoca {epoch:02d}/{args.epochs:02d} ({epoch_time:.1f}s) | LR: {current_lr:.6f}")
        print(f"  [Entrenamiento] Pérdida: {train_loss:.4f} | Acc Fruta: {train_acc_f*100:.2f}% | Acc Calidad: {train_acc_q*100:.2f}%")
        print(f"  [Validación]    Pérdida: {val_loss:.4f} | Acc Fruta: {val_acc_f*100:.2f}% | Acc Calidad: {val_acc_q*100:.2f}%")
        
        # Registrar en CSV
        with open(log_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, train_acc_f, train_acc_q,
                val_loss, val_acc_f, val_acc_q, current_lr
            ])
            
        # Guardar el mejor modelo y verificar Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  * [NUEVO MEJOR] ¡Nuevo mejor modelo guardado en {best_model_path}!")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience_limit:
                print(f"\n[train_cnn] Early Stopping activado. No hay mejora tras {patience_limit} épocas.")
                break
                
    print("\n[train_cnn] Pipeline de entrenamiento CNN finalizado con éxito.")

if __name__ == "__main__":
    main()
