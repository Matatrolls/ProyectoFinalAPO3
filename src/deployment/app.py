import os
import io
import base64
import json
import math
import numpy as np
import cv2
from PIL import Image
import joblib

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

from src.data.preprocess import (
    FRUIT_CLASSES, QUALITY_CLASSES, NORMALIZE_MEAN, NORMALIZE_STD, IMG_SIZE,
    extract_hsv_histogram, extract_hog_features, OOD_CONFIDENCE_THRESHOLD
)
from src.data.size_estimator import SizeEstimator
from src.models.cnn_model import MultiHeadCNN

# Inicializar Flask y Socket.IO
app = Flask(__name__, template_folder="templates")
app.config['SECRET_KEY'] = 'secret_key_apo3_2026'
socketio = SocketIO(app, cors_allowed_origins="*")

# Dispositivo PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instanciar el estimador de tamaño
size_estimator = SizeEstimator()

# Cargar modelos en memoria
rf_model = None
svm_model = None
cnn_model = None

# Rutas de checkpoints
RF_PATH = "experiments/checkpoints/rf_best.joblib"
SVM_PATH = "experiments/checkpoints/svm_best.joblib"
CNN_PATH = "experiments/checkpoints/best_cnn_model.pth"

def load_models():
    global rf_model, svm_model, cnn_model
    
    if os.path.exists(RF_PATH):
        rf_model = joblib.load(RF_PATH)
        print("[app] Random Forest cargado con éxito.")
    else:
        print("[app] ADVERTENCIA: No se encontró RF_PATH.")
        
    if os.path.exists(SVM_PATH):
        svm_model = joblib.load(SVM_PATH)
        print("[app] SVM cargado con éxito.")
    else:
        print("[app] ADVERTENCIA: No se encontró SVM_PATH.")
        
    if os.path.exists(CNN_PATH):
        cnn_model = MultiHeadCNN(num_fruits=len(FRUIT_CLASSES), num_qualities=len(QUALITY_CLASSES)).to(device)
        cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=device))
        cnn_model.eval()
        print("[app] CNN Multi-Head cargada con éxito.")
    else:
        print("[app] ADVERTENCIA: No se encontró CNN_PATH.")

# Transformación para la CNN
cnn_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

def base64_to_cv2(b64_string):
    """Convierte un string base64 de imagen a formato BGR de OpenCV."""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def crop_contour(img, contour, padding=0.15):
    """
    Segmenta la fruta basándose en el contorno, limpia el fondo y la recorta con padding.
    """
    h, w = img.shape[:2]
    white_bg = np.ones_like(img) * 255
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    masked_img = np.where(mask[:, :, np.newaxis] == 255, img, white_bg)
    
    x, y, bw, bh = cv2.boundingRect(contour)
    if bw == 0 or bh == 0:
        return img
        
    pad_w = int(bw * padding)
    pad_h = int(bh * padding)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + bw + pad_w)
    y2 = min(h, y + bh + pad_h)
    
    cropped = masked_img[y1:y2, x1:x2]
    if cropped.size == 0:
        return img
    return cropped

FRUIT_COLORS = {
    "Apple": (16, 185, 129),       # Verde neón BGR
    "Banana": (0, 220, 220),       # Amarillo BGR
    "Orange": (0, 140, 255),       # Naranja BGR
    "Lime": (0, 255, 128),         # Verde limón BGR
    "Guava": (100, 220, 100),      # Verde suave BGR
    "Pomegranate": (68, 68, 239)   # Rojo BGR
}

FRUIT_TRANSLATIONS = {
    "Apple": "MANZANA",
    "Banana": "BANANA",
    "Orange": "NARANJA",
    "Lime": "LIMON",
    "Guava": "GUAYABA",
    "Pomegranate": "GRANADA"
}

def process_and_predict(img_bgr, model_name="cnn", threshold=None):
    """
    Realiza el pipeline completo de predicción para un frame/imagen con soporte multi-objeto:
      1. Detecta todos los contornos geométricos.
      2. Clasifica individualmente cada contorno candidato de fruta.
      3. Anota la imagen original con cajas de colores y banners.
    """
    global rf_model, svm_model, cnn_model
    
    if img_bgr is None:
        return {"status": "error", "message": "Imagen no válida."}
        
    # Usar el umbral enviado por el cliente o el por defecto del sistema
    conf_threshold = threshold if threshold is not None else OOD_CONFIDENCE_THRESHOLD
    
    # Inicializar respuesta por defecto
    result = {
        "status": "success",
        "fruit": "Desconocido",
        "quality": "Desconocido",
        "size_class": "N/A",
        "d_norm": 0.0,
        "confidence_fruit": 0.0,
        "confidence_quality": 0.0,
        "is_fruit": False,
        "message": "Ninguna fruta detectada.",
        "annotated_image": None,
        "detections": []
    }
    
    # --- Detección de Contornos ---
    h, w = img_bgr.shape[:2]
    img_area = h * w
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    blurred = cv2.GaussianBlur(s_channel, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_annotated = img_bgr.copy()
    detections = []
    
    # Filtrar contornos por tamaño relativo
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        area_norm = area / img_area
        if 0.015 <= area_norm <= 0.95:
            # Comprobar excentricidad básica
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw > 0 and bh > 0:
                aspect_ratio = bw / float(bh)
                if 0.2 <= aspect_ratio <= 5.0:
                    valid_contours.append((contour, area, area_norm, x, y, bw, bh))
                    
    # Ordenar por área de mayor a menor
    valid_contours = sorted(valid_contours, key=lambda item: item[1], reverse=True)
    
    for idx, (contour, area, area_norm, x, y, bw, bh) in enumerate(valid_contours):
        # Aislar y recortar la fruta sobre fondo blanco
        cropped_img = crop_contour(img_bgr, contour)
        
        pred_fruit = "Desconocido"
        pred_quality = "Desconocido"
        conf_f = 0.0
        conf_q = 0.0
        
        # ── Inferencia con el modelo correspondiente ──
        if model_name == "cnn" and cnn_model is not None:
            img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            tensor_img = cnn_transform(img_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out_fruit, out_quality = cnn_model(tensor_img)
                prob_fruit = F.softmax(out_fruit, dim=1)
                prob_quality = F.softmax(out_quality, dim=1)
                
                max_prob_f, pred_f = torch.max(prob_fruit, dim=1)
                max_prob_q, pred_q = torch.max(prob_quality, dim=1)
                
                conf_f = max_prob_f.item()
                conf_q = max_prob_q.item()
                
                if conf_f >= conf_threshold:
                    pred_fruit = FRUIT_CLASSES[pred_f.item()]
                    pred_quality = QUALITY_CLASSES[pred_q.item()]
                    
        elif model_name in ("random_forest", "svm") and (rf_model is not None or svm_model is not None):
            model = rf_model if model_name == "random_forest" else svm_model
            if model is not None:
                img_resized = cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE))
                hsv_feats = extract_hsv_histogram(img_resized)
                hog_feats = extract_hog_features(img_resized)
                feats = np.concatenate([hsv_feats, hog_feats]).reshape(1, -1)
                
                pred_comb = int(model.predict(feats)[0])
                probs = model.predict_proba(feats)[0]
                
                n_total = len(FRUIT_CLASSES) * len(QUALITY_CLASSES)
                full_probs = np.zeros(n_total)
                for c_idx, class_label in enumerate(model.classes_):
                    full_probs[int(class_label)] = probs[c_idx]
                
                fruit_idx = pred_comb // len(QUALITY_CLASSES)
                quality_idx = pred_comb % len(QUALITY_CLASSES)
                
                c_f = float(sum(full_probs[fruit_idx * len(QUALITY_CLASSES) + q]
                                   for q in range(len(QUALITY_CLASSES))))
                c_q = float(full_probs[pred_comb] / c_f) if c_f > 1e-9 else 0.0
                
                conf_f = c_f
                conf_q = min(1.0, c_q)
                
                if conf_f >= conf_threshold:
                    pred_fruit = FRUIT_CLASSES[fruit_idx]
                    pred_quality = QUALITY_CLASSES[quality_idx]

        # Calcular tamaño físico estimado para este objeto
        eq_diameter = math.sqrt((4 * area) / math.pi)
        d_norm_val = eq_diameter / max(w, h)
        size_class = "N/A"
        if pred_fruit != "Desconocido":
            size_class, _ = size_estimator.estimate(img_bgr, pred_fruit)
            if not size_class:
                size_class = "Mediano"
                
        # --- Dibujar en img_annotated ---
        color_box = FRUIT_COLORS.get(pred_fruit, (128, 128, 128))
        
        # Dibujar contorno y rectangulo
        cv2.drawContours(img_annotated, [contour], -1, color_box, 2)
        cv2.rectangle(img_annotated, (x, y), (x+bw, y+bh), color_box, 2)
        
        # Banner del texto
        fruit_label_es = FRUIT_TRANSLATIONS.get(pred_fruit, "DESCONOCIDO")
        label_text = f"{fruit_label_es} {int(conf_f * 100)}%"
        
        # Obtener dimensiones del texto para el banner
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        # Dibujar rectángulo relleno
        cv2.rectangle(img_annotated, (x, y - text_h - 10), (x + text_w + 10, y), color_box, -1)
        # Escribir texto en blanco
        cv2.putText(img_annotated, label_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Guardar detección
        detections.append({
            "fruit": pred_fruit,
            "quality": pred_quality,
            "confidence_fruit": conf_f,
            "confidence_quality": conf_q,
            "size_class": size_class,
            "d_norm": d_norm_val,
            "box": [x, y, bw, bh]
        })
        
    result["detections"] = detections
    
    # Si detectamos al menos un objeto válido, poblar el resultado principal con el mayor
    if len(detections) > 0:
        main_det = detections[0]
        result["fruit"] = main_det["fruit"]
        result["quality"] = main_det["quality"]
        result["size_class"] = main_det["size_class"]
        result["d_norm"] = main_det["d_norm"]
        result["confidence_fruit"] = main_det["confidence_fruit"]
        result["confidence_quality"] = main_det["confidence_quality"]
        result["is_fruit"] = main_det["fruit"] != "Desconocido"
        result["message"] = f"Se detectaron {len(detections)} objetos."
        
    # Codificar la imagen anotada a base64
    _, buffer = cv2.imencode('.jpg', img_annotated)
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')
    result["annotated_image"] = "data:image/jpeg;base64," + annotated_b64
    
    return result

# ─────────────────────────────────────────────
# Rutas HTTP de Flask
# ─────────────────────────────────────────────

@app.route('/')
def index():
    # Cargar modelos si no están cargados
    if cnn_model is None:
        load_models()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_upload():
    """Endpoint para clasificar imágenes cargadas manualmente."""
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No se subió ninguna imagen."}), 400
        
    file = request.files['image']
    model_name = request.form.get('model', 'cnn')
    threshold_str = request.form.get('threshold')
    
    threshold = float(threshold_str) if threshold_str else None
    
    try:
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        prediction = process_and_predict(img_bgr, model_name=model_name, threshold=threshold)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ─────────────────────────────────────────────
# Eventos Socket.IO para Video en Tiempo Real
# ─────────────────────────────────────────────

@socketio.on('video_frame')
def handle_video_frame(data):
    """Recibe fotogramas de la cámara y emite la predicción."""
    try:
        frame_data = data['image']
        model_name = data.get('model', 'cnn')
        threshold = data.get('threshold')
        is_formal = data.get('is_formal', False)
        
        if threshold is not None:
            threshold = float(threshold)
            
        img_bgr = base64_to_cv2(frame_data)
        prediction = process_and_predict(img_bgr, model_name=model_name, threshold=threshold)
        prediction['is_formal'] = is_formal
        
        emit('prediction_result', prediction)
    except Exception as e:
        emit('prediction_result', {"status": "error", "message": str(e)})

if __name__ == '__main__':
    load_models()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
