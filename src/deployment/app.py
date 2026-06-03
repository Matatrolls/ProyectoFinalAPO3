import os
import io
import base64
import json
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
    extract_hsv_histogram, extract_hog_features
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

def process_and_predict(img_bgr, model_name="cnn"):
    """
    Realiza el pipeline completo de predicción para un frame/imagen:
      1. Estimación de tamaño y contornos (OOD geométrico).
      2. Clasificación de tipo de fruta y calidad (CNN, SVM o RF).
      3. Filtro de confianza OOD.
    """
    if img_bgr is None:
        return {"status": "error", "message": "Imagen no válida."}
        
    # Inicializar respuesta por defecto
    result = {
        "status": "success",
        "fruit": "Desconocido",
        "quality": "Desconocido",
        "size_class": "Desconocido",
        "d_norm": 0.0,
        "confidence_fruit": 0.0,
        "confidence_quality": 0.0,
        "is_fruit": False,
        "message": "Escaneando..."
    }
    
    # ── 1. Estimador de tamaño geométrico y filtro OOD ──────────────────
    # Extraer diámetro normalizado para el filtro inicial
    d_norm = size_estimator.process_image(img_bgr)
    
    if d_norm is None:
        result["message"] = "Rechazado: El objeto no tiene la geometría o el tamaño de una fruta."
        result["fruit"] = "No es Fruta"
        result["quality"] = "N/A"
        result["size_class"] = "N/A"
        return result
        
    result["is_fruit"] = True
    result["d_norm"] = float(d_norm)
    
    # ── 2. Clasificación ────────────────────────────────────────────────
    predicted_fruit_name = "Desconocido"
    
    if model_name == "cnn" and cnn_model is not None:
        # Preprocesar para CNN
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor_img = cnn_transform(img_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out_fruit, out_quality = cnn_model(tensor_img)
            
            prob_fruit = F.softmax(out_fruit, dim=1)
            prob_quality = F.softmax(out_quality, dim=1)
            
            max_prob_f, pred_f = torch.max(prob_fruit, dim=1)
            max_prob_q, pred_q = torch.max(prob_quality, dim=1)
            
            conf_f = max_prob_f.item()
            conf_q = max_prob_q.item()
            
            fruit_idx = pred_f.item()
            quality_idx = pred_q.item()
            
            # Filtro OOD por umbral de confianza
            if conf_f >= 0.70:
                result["fruit"] = FRUIT_CLASSES[fruit_idx]
                predicted_fruit_name = FRUIT_CLASSES[fruit_idx]
                result["confidence_fruit"] = float(conf_f)
            else:
                result["fruit"] = "Desconocido"
                result["message"] = "Rechazado: Confianza de clasificación de fruta muy baja."
                
            if conf_q >= 0.70 and conf_f >= 0.70:
                result["quality"] = QUALITY_CLASSES[quality_idx]
                result["confidence_quality"] = float(conf_q)
            else:
                result["quality"] = "Desconocido"
                
    elif model_name in ("random_forest", "svm"):
        model = rf_model if model_name == "random_forest" else svm_model
        
        if model is not None:
            # Extraer características tabulares
            img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
            hsv = extract_hsv_histogram(img_resized)
            hog = extract_hog_features(img_resized)
            feats = np.concatenate([hsv, hog]).reshape(1, -1)
            
            # Predicción conjunta (combinada)
            pred_comb = model.predict(feats)[0]
            probs = model.predict_proba(feats)[0]
            
            # Desempaquetar
            fruit_idx = pred_comb // len(QUALITY_CLASSES)
            quality_idx = pred_comb % len(QUALITY_CLASSES)
            
            conf_comb = np.max(probs)
            result["confidence_fruit"] = float(conf_comb)
            result["confidence_quality"] = float(conf_comb)
            
            if conf_comb >= 0.70:
                result["fruit"] = FRUIT_CLASSES[fruit_idx]
                predicted_fruit_name = FRUIT_CLASSES[fruit_idx]
                result["quality"] = QUALITY_CLASSES[quality_idx]
            else:
                result["fruit"] = "Desconocido"
                result["quality"] = "Desconocido"
                result["message"] = "Rechazado: Confianza conjunta muy baja."
        else:
            return {"status": "error", "message": f"Modelo '{model_name}' no cargado."}
    else:
        return {"status": "error", "message": "Modelo o configuración no disponible."}
        
    # ── 3. Clasificación de tamaño usando umbrales específicos ───────
    if result["fruit"] != "Desconocido" and result["fruit"] != "No es Fruta":
        # Recalcular tamaño con las estadísticas de la fruta detectada
        size_class, _ = size_estimator.estimate(img_bgr, predicted_fruit_name)
        result["size_class"] = size_class if size_class else "Rechazado"
    else:
        result["size_class"] = "N/A"
        
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
    
    try:
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        prediction = process_and_predict(img_bgr, model_name=model_name)
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
        
        img_bgr = base64_to_cv2(frame_data)
        prediction = process_and_predict(img_bgr, model_name=model_name)
        
        emit('prediction_result', prediction)
    except Exception as e:
        emit('prediction_result', {"status": "error", "message": str(e)})

if __name__ == '__main__':
    load_models()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
