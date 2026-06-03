"""
cnn_model.py — Arquitectura CNN Multi-Head con PyTorch
======================================================
Universidad Icesi · APO III · 2026-1

Define una CNN personalizada en PyTorch que procesa imágenes de fruta (3×128×128) y
predice simultáneamente:
  - Tipo de fruta (head_fruit):   6 clases (Apple, Banana, Guava, Lime, Orange, Pomegranate)
  - Calidad de fruta (head_quality): 3 clases (Bueno, Malo, Normal)

Arquitectura:
  Entrada (3×128×128)
    → Bloque Conv 1: Conv2d(3, 32, 3×3) → BN → ReLU → MaxPool2d(2×2)   → 32×64×64
    → Bloque Conv 2: Conv2d(32, 64, 3×3) → BN → ReLU → MaxPool2d(2×2)  → 64×32×32
    → Bloque Conv 3: Conv2d(64, 128, 3×3) → BN → ReLU → MaxPool2d(2×2) → 128×16×16
    → GlobalAveragePooling2D                                          → 128
    → Linear(128, 256) → ReLU → Dropout(0.3)
    → head_fruit:   Linear(256, 6)
    → head_quality: Linear(256, 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCNN(nn.Module):
    def __init__(self, num_fruits: int = 6, num_qualities: int = 3):
        super(MultiHeadCNN, self).__init__()
        
        # Bloque Convolucional 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Bloque Convolucional 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bloque Convolucional 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Capa densa compartida
        self.fc_shared = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.3)
        
        # Cabezas de Clasificación (Multi-Head)
        # Nota: Retornan logits crudos. CrossEntropyLoss de PyTorch aplica Softmax internamente.
        self.head_fruit = nn.Linear(256, num_fruits)
        self.head_quality = nn.Linear(256, num_qualities)
        
    def forward(self, x):
        # x shape: (batch_size, 3, 128, 128)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.gap(x)
        x = torch.flatten(x, 1) # shape: (batch_size, 128)
        
        x = F.relu(self.fc_shared(x))
        x = self.dropout(x)
        
        out_fruit = self.head_fruit(x)
        out_quality = self.head_quality(x)
        
        return out_fruit, out_quality

if __name__ == "__main__":
    # Prueba rápida de la arquitectura
    model = MultiHeadCNN(num_fruits=6, num_qualities=3)
    dummy_input = torch.randn(2, 3, 128, 128) # batch_size=2, 3 channels, 128x128
    out_f, out_q = model(dummy_input)
    print("MultiHeadCNN (PyTorch) construida correctamente.")
    print(f"Entrada: {dummy_input.shape}")
    print(f"Salida Fruta (logits): {out_f.shape} (Esperado: [2, 6])")
    print(f"Salida Calidad (logits): {out_q.shape} (Esperado: [2, 3])")
