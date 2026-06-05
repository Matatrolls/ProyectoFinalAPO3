"""
preprocess.py — Pipeline de Carga y Preprocesamiento de Datos
=============================================================
Universidad Icesi · APO III · 2026-1

Este módulo es responsable de:
  1. Localizar el dataset de Kaggle en caché local y el custom_dataset del grupo.
  2. Construir un inventario de imágenes con etiquetas duales:
       - tipo_fruta : Apple, Banana, Guava, Lime, Orange, Pomegranate
       - calidad    : Bueno, Malo, Normal
  3. Balancear las clases (submuestreo al mínimo de la clase mayoritaria).
  4. Dividir en conjuntos de entrenamiento / validación / prueba (70/15/15).
  5. Extraer características tabulares (HSV + HOG) para modelos tradicionales.
  6. Construir DataLoaders de PyTorch para la CNN con Data Augmentation.

Uso básico:
    from src.data.preprocess import FruitDatasetBuilder
    builder = FruitDatasetBuilder()
    builder.build()
    X_train, X_val, X_test, y_train, y_val, y_test = builder.get_tabular_splits()
    train_loader, val_loader, test_loader = builder.get_dataloaders(batch_size=32)
"""

import os
import random
import warnings
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Constantes globales del proyecto
# ─────────────────────────────────────────────

# Tamaño estándar de imagen para la CNN
IMG_SIZE = 128

# Tamaño reducido para extracción de características HOG
HOG_IMG_SIZE = 64

# Número de bins del histograma HSV por canal
HSV_BINS = 32

# Semilla aleatoria para reproducibilidad
RANDOM_SEED = 42

# Umbral mínimo de confianza del clasificador (OOD)
OOD_CONFIDENCE_THRESHOLD = 0.70

# Estadísticas de normalización (se calculan del dataset, aquí usamos ImageNet como base)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# Mapeo de nombres de carpetas del dataset Kaggle a etiquetas legibles
KAGGLE_QUALITY_MAP = {
    "Good Quality_Fruits": "Bueno",
    "Bad Quality_Fruits":  "Malo",
}

# Frutas soportadas (en orden alfabético para índices consistentes)
FRUIT_CLASSES = sorted([
    "Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate"
])

# Clases de calidad soportadas
QUALITY_CLASSES = ["Bueno", "Malo", "Normal"]


# ─────────────────────────────────────────────
# Rutas del proyecto
# ─────────────────────────────────────────────

def get_project_root() -> Path:
    """Devuelve la raíz del repositorio del proyecto."""
    return Path(__file__).resolve().parent.parent.parent


def find_kaggle_dataset_path(dataset_handle: str) -> Path | None:
    """
    Localiza automáticamente el dataset de Kaggle en la caché local de kagglehub.
    Si no se encuentra, intenta descargarlo.

    Args:
        dataset_handle: El handle del dataset en Kaggle (ej. "ryandpark/fruit-quality-classification").

    Returns:
        Path al directorio raíz del dataset (que contiene las carpetas de calidad).
        None si no puede encontrarlo ni descargarlo.
    """
    try:
        parts = dataset_handle.split('/')
        versions_path = (
            Path.home() / ".cache" / "kagglehub" / "datasets"
            / parts[0] / parts[1] / "versions"
        )

        if versions_path.exists():
            version_dirs = sorted(
                [p for p in versions_path.iterdir() if p.is_dir()],
                key=lambda p: p.name
            )
            if version_dirs:
                dataset_root = version_dirs[-1]
                print(f"[preprocess] Dataset de Kaggle '{dataset_handle}' encontrado en: {dataset_root}")
                return dataset_root
    except Exception as e:
        print(f"[preprocess] Error buscando en caché para {dataset_handle}: {e}")

    print(f"[preprocess] Dataset '{dataset_handle}' no encontrado en caché. Intentando descargar...")
    try:
        import kagglehub
        path_str = kagglehub.dataset_download(dataset_handle)
        return Path(path_str)
    except Exception as e:
        print(f"[preprocess] ERROR al descargar el dataset {dataset_handle}: {e}")
        return None


def get_custom_dataset_path() -> Path:
    """Devuelve la ruta al dataset personalizado del grupo."""
    return get_project_root() / "data" / "custom_dataset"


# ─────────────────────────────────────────────
# Extracción de características de imagen
# ─────────────────────────────────────────────

def extract_hsv_histogram(img_bgr: np.ndarray, bins: int = HSV_BINS) -> np.ndarray:
    """
    Extrae un histograma de color en espacio HSV.

    La formulación matemática es:
        h_c[k] = (1/N) * Σ 1[I_c(x,y) ∈ bin_k]   para cada canal c ∈ {H, S, V}

    Donde N es el número total de píxeles e I_c es el valor del canal c.

    Args:
        img_bgr: Imagen en formato BGR de OpenCV (numpy array uint8).
        bins:    Número de bins por canal del histograma.

    Returns:
        Vector 1D de longitud 3 * bins con el histograma normalizado concatenado.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    features = []
    # H: rango [0, 180], S: [0, 256], V: [0, 256]
    ranges = [(0, 180), (0, 256), (0, 256)]
    for ch, (lo, hi) in enumerate(ranges):
        hist = cv2.calcHist([img_hsv], [ch], None, [bins], [lo, hi])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)  # Normalización L1
        features.append(hist)
    return np.concatenate(features)  # Longitud: 3 * bins = 96


def extract_hog_features(img_bgr: np.ndarray, size: int = HOG_IMG_SIZE) -> np.ndarray:
    """
    Extrae características HOG (Histogram of Oriented Gradients) usando OpenCV.

    HOG cuantifica la distribución local de gradientes de intensidad,
    capturando la forma estructural de la fruta:
        G_x = ∂I/∂x,  G_y = ∂I/∂y
        magnitude   = sqrt(G_x² + G_y²)
        orientation = arctan(G_y / G_x)

    Los gradientes se acumulan en histogramas de celdas de 8×8 píxeles,
    agrupados en bloques de 2×2 celdas y normalizados (L2-Hys).

    Args:
        img_bgr: Imagen en formato BGR de OpenCV.
        size:    Tamaño al que redimensionar antes de extraer HOG (debe ser múltiplo de 8).

    Returns:
        Vector 1D de características HOG (longitud fija para size=64: 1764 dims).
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (size, size))

    # Configuración estándar del HOGDescriptor de OpenCV
    win_size    = (size, size)
    block_size  = (16, 16)   # Bloque de 2×2 celdas (cada celda = 8×8 px)
    block_stride = (8, 8)
    cell_size   = (8, 8)
    n_bins      = 9           # 9 orientaciones

    hog_desc = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
    features = hog_desc.compute(img_resized)
    return features.flatten()


def extract_tabular_features(image_path: str) -> np.ndarray | None:
    """
    Extrae el vector de características tabulares completo para modelos de ML tradicional.
    Combina histograma HSV (96 dims) + HOG (≈ 1764 dims).

    Args:
        image_path: Ruta al archivo de imagen.

    Returns:
        Vector 1D concatenado [HSV | HOG], o None si la imagen no puede leerse.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Redimensionar a tamaño estándar para consistencia
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    hsv_feats = extract_hsv_histogram(img)
    hog_feats = extract_hog_features(img)

    return np.concatenate([hsv_feats, hog_feats])


# ─────────────────────────────────────────────
# Inventario del dataset
# ─────────────────────────────────────────────

def build_inventory(kaggle_roots: list[Path | None], custom_root: Path) -> list[dict]:
    """
    Escanea los datasets de Kaggle y el custom_dataset para construir un inventario
    de imágenes con sus etiquetas.

    El inventario es una lista de diccionarios con las siguientes claves:
        path       : str  — Ruta absoluta al archivo de imagen
        fruit      : str  — Nombre de la fruta (ej. 'Apple')
        quality    : str  — Calidad ('Bueno', 'Malo' o 'Normal')
        fruit_idx  : int  — Índice numérico del tipo de fruta
        quality_idx: int  — Índice numérico de la calidad

    Args:
        kaggle_roots: Lista de rutas a directorios raíz de datasets de Kaggle.
        custom_root:  Ruta al directorio custom_dataset/ del proyecto.

    Returns:
        Lista de diccionarios con la información de cada imagen.
    """
    inventory = []
    fruit_to_idx   = {f: i for i, f in enumerate(FRUIT_CLASSES)}
    quality_to_idx = {q: i for i, q in enumerate(QUALITY_CLASSES)}
    
    # Mapeo de nombres en español (usados en dataset-frutas) a inglés (FRUIT_CLASSES)
    es_to_en_map = {
        "manzana": "Apple",
        "banano": "Banana",
        "guayaba": "Guava",
        "limon": "Lime",
        "naranja": "Orange",
        "granada": "Pomegranate"
    }

    # ── 1. Cargar desde Kaggle (Bueno y Malo) ──────────────────────────────
    for kaggle_root in kaggle_roots:
        if kaggle_root is not None:
            print(f"[preprocess] Explorando dataset en: {kaggle_root}")
            for img_file in kaggle_root.rglob("*"):
                if not img_file.is_file() or img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                
                path_parts = img_file.parts
                quality_label = None
                fruit_name = None
                
                # Buscar en los nombres de directorios/archivo para inferir clase y calidad
                for part in reversed(path_parts):
                    part_lower = part.lower()
                    
                    if not quality_label:
                        if "good" in part_lower or "bueno" in part_lower:
                            quality_label = "Bueno"
                        elif "bad" in part_lower or "malo" in part_lower:
                            quality_label = "Malo"
                    
                    if not fruit_name:
                        # 1. Intentar hacer match directo con nombres en inglés
                        for f in FRUIT_CLASSES:
                            if f.lower() in part_lower:
                                fruit_name = f
                                break
                        # 2. Intentar hacer match con el mapeo en español
                        if not fruit_name:
                            for es_name, en_name in es_to_en_map.items():
                                if es_name in part_lower:
                                    fruit_name = en_name
                                    break
                                
                    if quality_label and fruit_name:
                        break
                        
                if quality_label and fruit_name:
                    inventory.append({
                        "path":        str(img_file),
                        "fruit":       fruit_name,
                        "quality":     quality_label,
                        "fruit_idx":   fruit_to_idx[fruit_name],
                        "quality_idx": quality_to_idx[quality_label],
                    })

    # ── 2. Cargar desde custom_dataset (puede incluir Normal) ──────────────
    for quality_label in QUALITY_CLASSES:
        custom_quality_dir = custom_root / quality_label
        if not custom_quality_dir.exists() or not any(custom_quality_dir.iterdir()):
            continue  # Carpeta vacía o no existente, se omite

        for img_file in custom_quality_dir.iterdir():
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            # Para imágenes del custom dataset, intentar inferir la fruta del
            # nombre del archivo (ej. "Apple_001.jpg" → "Apple"), o marcar
            # como fruta más cercana basado en color (futuro paso).
            # Por ahora se requiere que el nombre del archivo empiece con
            # el nombre de la fruta (case-insensitive).
            fruit_name = None
            for f in FRUIT_CLASSES:
                if img_file.stem.lower().startswith(f.lower()):
                    fruit_name = f
                    break

            if fruit_name is None:
                print(f"[preprocess] AVISO: no se pudo inferir fruta de '{img_file.name}'. "
                      f"Nómbralo como 'Apple_001.jpg', 'Banana_002.jpg', etc.")
                continue

            inventory.append({
                "path":        str(img_file),
                "fruit":       fruit_name,
                "quality":     quality_label,
                "fruit_idx":   fruit_to_idx[fruit_name],
                "quality_idx": quality_to_idx[quality_label],
            })

    print(f"[preprocess] Inventario total: {len(inventory)} imágenes")
    return inventory


def balance_inventory(inventory: list[dict], key: str = "quality") -> list[dict]:
    """
    Balancea el inventario por submuestreo al tamaño de la clase minoritaria.

    El submuestreo es necesario porque Pomegranate_Good tiene ~5940 imágenes
    frente a ~1085 de Lime_Bad, lo que causaría un sesgo del clasificador hacia
    las clases con más muestras.

    Args:
        inventory: Lista completa de imágenes con etiquetas.
        key:       Clave de agrupación ('quality' o 'fruit').

    Returns:
        Lista balanceada, con el mismo número de muestras por cada grupo del 'key'.
    """
    groups = defaultdict(list)
    for item in inventory:
        groups[item[key]].append(item)

    min_count = min(len(g) for g in groups.values())
    print(f"[preprocess] Balanceo por '{key}': {min_count} muestras por clase")
    for k, items in groups.items():
        print(f"  {k}: {len(items)} -> {min_count}")

    balanced = []
    random.seed(RANDOM_SEED)
    for items in groups.values():
        balanced.extend(random.sample(items, min_count))

    random.shuffle(balanced)
    return balanced


# ─────────────────────────────────────────────
# Dataset de PyTorch
# ─────────────────────────────────────────────

class FruitImageDataset(Dataset):
    """
    Dataset de PyTorch para imágenes de frutas.

    Carga imágenes desde el inventario y aplica las transformaciones
    configuradas (normalización + augmentación opcional).

    Args:
        inventory:  Lista de diccionarios del inventario de imágenes.
        transform:  Transformaciones de torchvision a aplicar.
        target:     'combined' (fruta * n_quality + calidad, etiqueta única),
                    'multi' (devuelve tupla (fruit_idx, quality_idx)).
    """

    def __init__(self, inventory: list[dict], transform=None, target: str = "multi"):
        self.inventory = inventory
        self.transform = transform
        self.target = target

    def __len__(self) -> int:
        return len(self.inventory)

    def __getitem__(self, idx: int):
        item = self.inventory[idx]
        img = Image.open(item["path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.target == "multi":
            return img, item["fruit_idx"], item["quality_idx"]
        else:
            # Etiqueta combinada: codifica (fruta, calidad) como un único entero
            combined = item["fruit_idx"] * len(QUALITY_CLASSES) + item["quality_idx"]
            return img, combined


# ─────────────────────────────────────────────
# Clase principal: FruitDatasetBuilder
# ─────────────────────────────────────────────

class FruitDatasetBuilder:
    """
    Clase principal que orquesta todo el pipeline de datos.

    Uso típico:
        builder = FruitDatasetBuilder()
        builder.build()

        # Para modelos tradicionales (Random Forest, SVM):
        X_train, X_val, X_test, y_train, y_val, y_test = builder.get_tabular_splits()

        # Para la CNN:
        train_loader, val_loader, test_loader = builder.get_dataloaders(batch_size=32)
    """

    def __init__(self, balance: bool = True, img_size: int = IMG_SIZE):
        self.balance = balance
        self.img_size = img_size
        self.inventory_full: list[dict] = []
        self.train_inv: list[dict] = []
        self.val_inv:   list[dict] = []
        self.test_inv:  list[dict] = []
        self._is_built = False

    def build(self):
        """
        Ejecuta el pipeline completo:
          1. Encuentra los datasets.
          2. Construye el inventario de imágenes con etiquetas.
          3. Balancea las clases si se requiere.
          4. Divide en train / val / test (70 / 15 / 15).
        """
        kaggle_root_1 = find_kaggle_dataset_path("ryandpark/fruit-quality-classification")
        kaggle_root_2 = find_kaggle_dataset_path("sebastiancos21/dataset-frutas")
        custom_root  = get_custom_dataset_path()

        self.inventory_full = build_inventory([kaggle_root_1, kaggle_root_2], custom_root)

        if len(self.inventory_full) == 0:
            raise RuntimeError(
                "[preprocess] No se encontraron imágenes. "
                "Descarga los datasets con download_dataset.py"
            )

        if self.balance:
            self.inventory_full = balance_inventory(self.inventory_full, key="quality")

        # División estratificada por calidad para mantener proporción de clases
        # 70% entrenamiento, 15% validación, 15% prueba
        labels_quality = [item["quality_idx"] for item in self.inventory_full]

        train_val, test = train_test_split(
            self.inventory_full,
            test_size=0.15,
            stratify=labels_quality,
            random_state=RANDOM_SEED
        )
        labels_train_val = [item["quality_idx"] for item in train_val]
        train, val = train_test_split(
            train_val,
            test_size=0.1765,   # ≈ 15/85 del total para obtener 15% final
            stratify=labels_train_val,
            random_state=RANDOM_SEED
        )

        self.train_inv = train
        self.val_inv   = val
        self.test_inv  = test
        self._is_built = True

        print(f"[preprocess] División del dataset:")
        print(f"  Entrenamiento : {len(self.train_inv)} imágenes")
        print(f"  Validación    : {len(self.val_inv)} imágenes")
        print(f"  Prueba        : {len(self.test_inv)} imágenes")

    # ── Características tabulares para ML tradicional ──────────────────────

    def get_tabular_splits(self, label: str = "quality"):
        """
        Extrae vectores de características tabulares (HSV + HOG) para los
        conjuntos de entrenamiento, validación y prueba.

        Args:
            label: 'quality' para clasificar por calidad,
                   'fruit'   para clasificar por tipo de fruta,
                   'combined' para clasificar por (fruta, calidad) juntos.

        Returns:
            Tupla (X_train, X_val, X_test, y_train, y_val, y_test) como arrays numpy.
        """
        if not self._is_built:
            raise RuntimeError("Ejecuta build() primero.")

        def _extract_set(inv):
            X, y = [], []
            for item in inv:
                feats = extract_tabular_features(item["path"])
                if feats is None:
                    continue
                X.append(feats)
                if label == "quality":
                    y.append(item["quality_idx"])
                elif label == "fruit":
                    y.append(item["fruit_idx"])
                else:  # combined
                    y.append(item["fruit_idx"] * len(QUALITY_CLASSES) + item["quality_idx"])
            return np.array(X), np.array(y)

        print("[preprocess] Extrayendo características tabulares (esto toma varios minutos)...")
        X_train, y_train = _extract_set(self.train_inv)
        X_val,   y_val   = _extract_set(self.val_inv)
        X_test,  y_test  = _extract_set(self.test_inv)

        print(f"[preprocess] Forma del vector de caracteristicas: {X_train.shape[1]} dims")
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ── DataLoaders de PyTorch para la CNN ────────────────────────────────

    def get_dataloaders(self, batch_size: int = 32, num_workers: int = 0):
        """
        Construye y devuelve los DataLoaders de PyTorch para entrenamiento,
        validación y prueba.

        Transformaciones de entrenamiento (Data Augmentation):
          - Redimensionado a IMG_SIZE × IMG_SIZE
          - Volteo horizontal aleatorio (p=0.5)
          - Rotación aleatoria ±15°
          - Variación aleatoria de brillo y contraste (ColorJitter)
          - Conversión a tensor y normalización ImageNet

        Transformaciones de validación/prueba (sin augmentación):
          - Redimensionado a IMG_SIZE × IMG_SIZE
          - Conversión a tensor y normalización ImageNet

        Args:
            batch_size:   Tamaño del mini-batch.
            num_workers:  Procesos paralelos de carga (0 = sin paralelismo, seguro en Windows).

        Returns:
            Tupla (train_loader, val_loader, test_loader).
        """
        if not self._is_built:
            raise RuntimeError("Ejecuta build() primero.")

        normalize = T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)

        train_transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            normalize,
        ])

        eval_transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            normalize,
        ])

        train_ds = FruitImageDataset(self.train_inv, transform=train_transform, target="multi")
        val_ds   = FruitImageDataset(self.val_inv,   transform=eval_transform,  target="multi")
        test_ds  = FruitImageDataset(self.test_inv,  transform=eval_transform,  target="multi")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=False)

        return train_loader, val_loader, test_loader

    # ── Información del inventario ─────────────────────────────────────────

    def get_class_info(self) -> dict:
        """Devuelve información de las clases disponibles."""
        return {
            "fruit_classes":    FRUIT_CLASSES,
            "quality_classes":  QUALITY_CLASSES,
            "n_fruits":         len(FRUIT_CLASSES),
            "n_qualities":      len(QUALITY_CLASSES),
            "fruit_to_idx":     {f: i for i, f in enumerate(FRUIT_CLASSES)},
            "quality_to_idx":   {q: i for i, q in enumerate(QUALITY_CLASSES)},
            "idx_to_fruit":     {i: f for i, f in enumerate(FRUIT_CLASSES)},
            "idx_to_quality":   {i: q for i, q in enumerate(QUALITY_CLASSES)},
            "ood_threshold":    OOD_CONFIDENCE_THRESHOLD,
        }

    def summary(self):
        """Imprime un resumen del inventario por fruta y calidad."""
        if not self._is_built:
            print("Dataset no construido. Ejecuta build() primero.")
            return

        print("\n" + "="*60)
        print("RESUMEN DEL DATASET")
        print("="*60)
        counts = defaultdict(lambda: defaultdict(int))
        for item in self.inventory_full:
            counts[item["fruit"]][item["quality"]] += 1

        header = f"{'Fruta':<15}" + "".join(f"{q:<10}" for q in QUALITY_CLASSES) + "Total"
        print(header)
        print("-" * 60)
        for fruit in FRUIT_CLASSES:
            row = f"{fruit:<15}"
            total = 0
            for quality in QUALITY_CLASSES:
                n = counts[fruit][quality]
                row += f"{n:<10}"
                total += n
            print(row + str(total))
        print("="*60 + "\n")


# ─────────────────────────────────────────────
# Punto de entrada (prueba rápida)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    builder = FruitDatasetBuilder(balance=True)
    builder.build()
    builder.summary()

    # Probar extracción de características en una muestra pequeña
    print("[TEST] Probando extracción de características en 5 imágenes...")
    sample = builder.train_inv[:5]
    for item in sample:
        feats = extract_tabular_features(item["path"])
        print(f"  {item['fruit']} ({item['quality']}): vector de {feats.shape[0]} dims")

    # Probar DataLoaders
    print("[TEST] Probando DataLoaders de PyTorch...")
    train_loader, val_loader, test_loader = builder.get_dataloaders(batch_size=4)
    imgs, fruit_labels, quality_labels = next(iter(train_loader))
    print(f"  Batch shape : {imgs.shape}")
    print(f"  Frutas      : {[FRUIT_CLASSES[i] for i in fruit_labels]}")
    print(f"  Calidades   : {[QUALITY_CLASSES[i] for i in quality_labels]}")
    print("[TEST] Preprocesamiento OK.")
