import cv2
import numpy as np
import math
import json
import os

class SizeEstimator:
    def __init__(self, stats_path="experiments/size_stats.json"):
        self.stats_path = stats_path
        self.stats = {}
        self._load_stats()

    def _load_stats(self):
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)

    def _save_stats(self):
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        with open(self.stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=4)

    def process_image(self, image_input):
        """
        Recibe una ruta de imagen o un array de numpy (BGR).
        Devuelve el diámetro equivalente normalizado y el contorno, o None si falla.
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        else:
            img = image_input

        if img is None:
            return None

        h, w = img.shape[:2]
        
        # Convertir a HSV y tomar el canal S (Saturación) o V (Valor) para Otsu.
        # Generalmente, las frutas tienen alta saturación frente al fondo (blanco/negro).
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        
        # Aplicar desenfoque para reducir ruido
        blurred = cv2.GaussianBlur(s_channel, (5, 5), 0)
        
        # Umbralización de Otsu
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Operaciones morfológicas para rellenar huecos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # Tomar el contorno más grande asumiendo que es la fruta
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Normalizar área
        img_area = h * w
        area_norm = area / img_area
        
        # Filtro Geométrico OOD (Out-of-Distribution)
        if area_norm < 0.02 or area_norm > 0.95:
            # Demasiado pequeño (ruido) o demasiado grande (ocupa toda la imagen)
            return None
            
        x, y, bw, bh = cv2.boundingRect(largest_contour)
        aspect_ratio = bw / float(bh)
        
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            # Muy alargado o muy achatado, probablemente no es una fruta
            return None
            
        # Calcular diámetro equivalente
        # D = sqrt(4 * Area / π)
        equivalent_diameter = math.sqrt((4 * area) / math.pi)
        
        # Normalizar respecto a la resolución de la imagen
        d_norm = equivalent_diameter / max(w, h)
        
        return d_norm

    def fit(self, dataset_items):
        """
        Calcula media y desviación estándar del diámetro por tipo de fruta.
        dataset_items: lista de diccionarios con {'path': ..., 'fruit': ...}
        """
        diameters = {}
        for item in dataset_items:
            fruit = item['fruit']
            if fruit == 'Normal': # Ignorar clase normal
                continue
                
            d_norm = self.process_image(item['path'])
            if d_norm is not None:
                if fruit not in diameters:
                    diameters[fruit] = []
                diameters[fruit].append(d_norm)
                
        self.stats = {}
        for fruit, vals in diameters.items():
            if len(vals) > 0:
                self.stats[fruit] = {
                    'mu': float(np.mean(vals)),
                    'sigma': float(np.std(vals))
                }
        self._save_stats()
        print(f"[SizeEstimator] Estadísticas calculadas para {len(self.stats)} frutas y guardadas en {self.stats_path}.")

    def estimate(self, image_input, fruit_type):
        """
        Clasifica el tamaño en Pequeño, Mediano, Grande o None (No es Fruta).
        """
        d_norm = self.process_image(image_input)
        if d_norm is None:
            return None, None # Rechazado por filtro geométrico
            
        if fruit_type not in self.stats:
            # Si no hay estadísticas, devolver solo el diámetro y un tamaño por defecto
            return 'Mediano', d_norm
            
        mu = self.stats[fruit_type]['mu']
        sigma = self.stats[fruit_type]['sigma']
        
        # Umbrales
        if d_norm < (mu - 0.5 * sigma):
            size_class = 'Pequeño'
        elif d_norm > (mu + 0.5 * sigma):
            size_class = 'Grande'
        else:
            size_class = 'Mediano'
            
        return size_class, d_norm

if __name__ == "__main__":
    # Test básico
    estimator = SizeEstimator()
    print("SizeEstimator instanciado. Stats:", estimator.stats)
