import os
import sys
import cv2
import numpy as np

# Añadir raíz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import find_kaggle_dataset_path

def analyze_dataset_limes():
    dataset_path = find_kaggle_dataset_path("ryandpark/fruit-quality-classification")
    if not dataset_path:
        print("Dataset no encontrado.")
        return
        
    lime_dirs = []
    for root, dirs, files in os.walk(dataset_path):
        if "lime" in root.lower():
            lime_dirs.append(root)
            
    print(f"Directorios de Lime encontrados: {lime_dirs}")
    
    # Leer hasta 5 imágenes de cada directorio y calcular su color promedio
    for lime_dir in lime_dirs:
        print(f"\nAnalizando imágenes en: {os.path.basename(os.path.dirname(lime_dir))} / {os.path.basename(lime_dir)}")
        files = [f for f in os.listdir(lime_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            continue
            
        for file in files[:3]:
            img_path = os.path.join(lime_dir, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Convertir a HSV para analizar el color
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Excluir el fondo blanco (donde saturación es muy baja y valor es muy alto)
            mask = (s > 30) & (v < 240)
            if np.sum(mask) == 0:
                mask = np.ones_like(s, dtype=bool)
                
            avg_h = np.mean(h[mask])
            avg_s = np.mean(s[mask])
            avg_v = np.mean(v[mask])
            
            # Interpretar Hue:
            # En OpenCV Hue va de 0 a 180.
            # 0-10: Rojo, 11-25: Naranja, 26-35: Amarillo, 36-85: Verde, 86-125: Azul/Cian, 126-180: Violeta/Rojo
            hue_deg = avg_h * 2
            color_desc = "Desconocido"
            if 0 <= hue_deg < 20 or 340 <= hue_deg <= 360:
                color_desc = "Rojo"
            elif 20 <= hue_deg < 45:
                color_desc = "Naranja"
            elif 45 <= hue_deg < 70:
                color_desc = "Amarillo"
            elif 70 <= hue_deg < 170:
                color_desc = "Verde"
            else:
                color_desc = "Azul/Violeta"
                
            print(f"  Archivo: {file} | Hue Promedio: {hue_deg:.1f}° ({color_desc}) | Sat: {avg_s:.1f} | Val: {avg_v:.1f}")

if __name__ == "__main__":
    analyze_dataset_limes()
