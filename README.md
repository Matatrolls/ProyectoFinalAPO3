#  Sistema de Clasificación de Calidad de Frutas mediante Visión por Computadora

**Proyecto Final — Algoritmos y Programación III (APO III)**
Universidad Icesi · Facultad de Ingeniería, Diseño y Ciencias Aplicadas
Semestre 2026-1

[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.14-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-orange)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

---

##  Descripción del Proyecto

Este proyecto implementa un sistema automático e interactivo de visión por computadora para clasificar la calidad de frutas. Está diseñado bajo la metodología **CRISP-DM**, cubriendo desde el análisis exploratorio de datos (EDA) hasta el modelado matemático, entrenamiento de redes neuronales, evaluación cuantitativa rigurosa y el despliegue en un entorno web en tiempo real.

El sistema analiza imágenes individuales de frutas, asigna una **categoría de fruta** (Manzana, Banano, Guayaba, Limón, Naranja, Granada) y una **etiqueta de calidad** (Bueno / Malo / Normal). Adicionalmente, calcula el **diámetro geométrico equivalente** del contorno de la fruta y clasifica su tamaño relativo en (Pequeño / Mediano / Grande) utilizando límites estadísticos personalizados por tipo de fruta.

El pipeline integra:
1. **Modelos Tradicionales de Machine Learning**: Random Forest y Máquina de Soporte Vectorial (SVM) sintonizados mediante búsqueda en rejilla (GridSearchCV) con validación cruzada.
2. **Red Neuronal Convolucional (CNN)**: Una arquitectura multi-salida (multi-head) diseñada desde cero en **PyTorch** para predecir en paralelo el tipo de fruta y la calidad de la misma.
3. **Filtro de Detección Fuera de Distribución (OOD)**:
   - **Filtro Geométrico OOD**: Utiliza morfología matemática y contornos con OpenCV para rechazar objetos no circulares, muy pequeños o con excentricidades anómalas.
   - **Filtro de Confianza de Clasificación**: Filtra objetos desconocidos si la confianza del clasificador (Softmax o probabilidad conjunta) es inferior a $70\%$.
4. **Despliegue Web Interactivo**: Servidor Flask con WebSocket (Socket.IO) que emite diagnósticos en tiempo real y controla un simulador interactivo de faja transportadora con animaciones CSS (Glassmorphism premium) y alertas visuales.

---

##  Frutas Soportadas

*   **Manzana** (*Apple*): Calidades Bueno / Malo / Normal (Kaggle + Custom Dataset)
*   **Banano** (*Banana*): Calidades Bueno / Malo / Normal (Kaggle + Custom Dataset)
*   **Guayaba** (*Guava*): Calidades Bueno / Malo / Normal (Kaggle + Custom Dataset)
*   **Limón/Lima** (*Lime*): Calidades Bueno / Malo / Normal (Kaggle + Custom Dataset)
*   **Naranja** (*Orange*): Calidades Bueno / Malo / Normal (Kaggle + Custom Dataset)
*   **Granada** (*Pomegranate*): Calidades Bueno / Malo / Normal (Kaggle + Custom Dataset)

>  **Extensibilidad**: El pipeline es completamente dinámico. Si el grupo coloca imágenes en `data/custom_dataset/Normal`, el preprocesamiento las detectará, balanceará y entrenará de forma transparente.

---

##  Estructura del Repositorio

```
ProyectoFinalAPO3/
├── src/
│   ├── data/
│   │   ├── preprocess.py        # Pipeline de carga, extracción de HSV+HOG y PyTorch Datasets
│   │   └── size_estimator.py    # Estimador geométrico y filtros de contorno OpenCV (PI3)
│   ├── models/
│   │   ├── traditional_models.py # Definición de Random Forest y SVM
│   │   └── cnn_model.py          # Arquitectura CNN Multi-Head en PyTorch
│   ├── training/
│   │   ├── train_traditional.py  # GridSearchCV para RandomForest y SVM
│   │   └── train_cnn.py          # Entrenamiento optimizado con ReduceLROnPlateau
│   ├── evaluation/
│   │   └── evaluate.py           # Generador de matrices de confusión, curvas y JSON final
│   └── deployment/
│       ├── app.py                # Servidor Flask + Socket.IO (Webcam + uploads en tiempo real)
│       └── templates/
│           └── index.html        # Interfaz web interactiva en Glassmorphism y Faja Animada
├── data/
│   └── custom_dataset/        # Carpeta para imágenes locales del grupo (Bueno / Malo / Normal)
├── docs/
│   ├── math_formulation.md       # Soporte matemático y ecuaciones (PI3)
│   └── ethics_report.md          # Análisis ético e impacto tecnológico (PI1 / PI2)
├── notebooks/
│   └── 01_eda.ipynb              # Notebook interactivo de Análisis Exploratorio de Datos (EDA)
├── experiments/
│   ├── logs/                  # Registro histórico de entrenamiento CSV de la CNN
│   ├── checkpoints/           # Modelos óptimos guardados (.joblib y .pth)
│   └── results/               # Matrices de confusión vectoriales, curvas de pérdida y JSON comparativo
├── main.py                    # Punto de entrada principal (Train, Evaluate, Deploy)
├── requirements.txt           # Dependencias requeridas
└── pasos.md                   # Hoja de ruta del proyecto y estado de completitud
```

---

### 1. Instalar dependencias
Asegúrate de usar Python 3.10 o superior (el proyecto es compatible hasta Python 3.14). Ejecuta:
```bash
pip install -r requirements.txt
```

*Nota sobre PyTorch: `requirements.txt` por defecto instala PyTorch en modo CPU, el cual es ideal para pruebas locales. Si dispones de GPU NVIDIA con soporte CUDA, puedes reinstalar PyTorch con:*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Descargar el dataset de Kaggle
El sistema automatiza la descarga usando la biblioteca `kagglehub`. Si no lo tienes en caché, la primera ejecución de `main.py` lo descargará automáticamente. Opcionalmente, puedes descargarlo de forma manual ejecutando:
```bash
python download_dataset.py
```
El dataset original se guardará en `~/.cache/kagglehub/datasets/ryandpark/fruit-quality-classification`.

---

## Modos de Uso

El script unificado [main.py]permite controlar todas las etapas del pipeline desde la terminal:

### A. Modo Entrenamiento (`train`)
Entrena todos los modelos desde cero. Sintoniza los hiperparámetros de RandomForest y SVM, y luego entrena la CNN Multi-Head:
```bash
python main.py --mode train
```

**Parámetros y optimización adicionales:**
*   **Omitir ML Tradicional**: Si ya entrenaste el Random Forest y el SVM y solo deseas reentrenar la CNN, agrega el flag `--cnn_only`:
    ```bash
    python main.py --mode train --cnn_only
    ```
*   **Ajustar Épocas de la CNN**: Por defecto entrena por 10 épocas, pero puedes especificarlo (ej. 5 épocas):
    ```bash
    python main.py --mode train --cnn_only --epochs 5
    ```
*   **Tamaño de Batch**: Configura el tamaño del lote (default: 32):
    ```bash
    python main.py --mode train --cnn_only --batch_size 64
    ```

*Optimización en CPU: Si el script no detecta una GPU NVIDIA, reducirá automáticamente el tamaño del dataset de entrenamiento y validación al 25% para evitar esperas prolongadas. Esto permite que el entrenamiento en CPU tome aproximadamente 1.5 a 2 minutos por época manteniendo el comportamiento y la convergencia.*

### B. Modo Evaluación (`evaluate`)
Evalúa los modelos guardados contra un conjunto de prueba independiente (20% del total de datos que nunca vieron los clasificadores). Genera matrices de confusión para cada modelo, curvas de aprendizaje de la CNN y un JSON de comparación en `experiments/results/`:
```bash
python main.py --mode evaluate
```

### C. Modo Despliegue (`deploy`)
Lanza la aplicación web interactiva local de Flask en el puerto `8080`:

```bash
python main.py --mode deploy --port 8080
```
Abre en tu navegador: `http://localhost:8080`.

---

##  Modelos y Metodología de Visión

### 1. Extracción de Características Tabulares (ML Tradicional)
Para alimentar a Random Forest y SVM, cada imagen es transformada en un vector de características de 1860 dimensiones:
*   **Histograma de Color HSV**: 3 canales (Hue, Saturation, Value) con 32 bins por canal. Captura las diferencias tonales de las frutas y el color asociado a imperfecciones o pudrición ($3 \times 32 = 96$ dimensiones).
*   **Histograma de Gradientes Orientados (HOG)**: Captura la forma geométrica, bordes y texturas de la fruta mediante una discretización de gradientes de intensidad en celdas de 8x8 píxeles ($1764$ dimensiones).

### 2. Random Forest
*   **Clasificador**: Clasifica una etiqueta combinada ($6\text{ frutas} \times 3\text{ calidades} = 18\text{ clases}$).
*   **Hiperparámetros óptimos**: `n_estimators: 200`, `max_depth: None` (obtenido tras GridSearchCV con Validación Cruzada K=5).
*   **Rendimiento en Test**: Excelente precisión (Fruta: $94.55\%$, Calidad: $96.86\%$).

### 3. Máquina de Soporte Vectorial (SVM)
*   **Clasificador**: SVM con Kernel RBF y salidas probabilísticas (Platt scaling).
*   **Hiperparámetros óptimos**: `C: 10`, `gamma: 'scale'`.
*   **Rendimiento en Test**: Sólido y robusto (Fruta: $90.53\%$, Calidad: $95.58\%$).

### 4. Red Neuronal Convolucional (CNN Multi-Head)
Diseñada desde cero en PyTorch utilizando una arquitectura convolucional multi-salida:
```
Imagen (128x128x3) -> ConvBlock1 -> ConvBlock2 -> ConvBlock3 -> GlobalAvgPool -> Dense (256) -> Dropout (0.3)
                                                                                                    ├── Head Frutas (6 salidas)
                                                                                                    └── Head Calidades (3 salidas)
```
Cada bloque convolucional consta de `Conv2D (3x3)` + `BatchNorm2d` + `ReLU` + `MaxPool2d (2x2)`. 
El entrenamiento aplica Data Augmentation en tiempo real (rotaciones aleatorias, volteos de imagen y variación de brillo) para evitar el sobreajuste.

---

##  Estimación de Tamaño y Filtros OOD (PI3)

### 1. Diámetro Geométrico Equivalente
Utilizando segmentación en espacio de color HSV (umbralización de Otsu), el estimador detecta la silueta de la fruta y calcula su diámetro geométrico equivalente normalizado con respecto a la resolución de la imagen:
$$D_{\text{eq}} = \sqrt{\frac{4 \times \text{Área}}{\pi}}$$
$$D_{\text{norm}} = \frac{D_{\text{eq}}}{\max(\text{ancho}, \text{alto})}$$

### 2. Clasificación de Tamaño
Se calculó previamente la media ($\mu$) y desviación estándar ($\sigma$) del diámetro normalizado de cada tipo de fruta en el dataset. El clasificador geométrico utiliza estos umbrales dinámicos:
*   **Pequeño**: $D_{\text{norm}} < \mu - 0.75\sigma$
*   **Mediano**: $\mu - 0.75\sigma \leq D_{\text{norm}} \leq \mu + 0.75\sigma$
*   **Grande**: $D_{\text{norm}} > \mu + 0.75\sigma$

### 3. Filtro Geométrico OOD
Para rechazar objetos que no son frutas, el módulo [size_estimator.py] analiza los contornos:
*   Si el área detectada es inferior a un umbral mínimo o superior a un máximo, se rechaza.
*   Si la excentricidad del objeto (relación de aspecto) es muy alargada (no cumple con la forma circular/ovalada esperada en una fruta), se marca como **No es Fruta** y la faja transportadora se detiene con alarma en la interfaz.

---

El conjunto de prueba independiente cuenta con **2037 imágenes**. A continuación se detallan las métricas comparativas obtenidas por cada modelo:

*   **Random Forest**: Exactitud Fruta: **94.55%**, Exactitud Calidad: **96.86%**, Tasa de Rechazo (OOD < 70%): 48.94%
*   **SVM (RBF Kernel)**: Exactitud Fruta: 90.53%, Exactitud Calidad: 95.58%, Tasa de Rechazo (OOD < 70%): **22.14%**
*   **CNN Multi-Head**: Exactitud Fruta: 68.48%, Exactitud Calidad: 88.22%, Tasa de Rechazo (OOD < 70%): 42.22%

*Interpretación de resultados:*
*   Los modelos tradicionales de ML (Random Forest y SVM) que combinan histogramas HSV y descriptores HOG obtuvieron puntuaciones de clasificación sobresalientes debido a la naturaleza de fondo uniforme y controlado del dataset.
*   La SVM obtuvo la menor tasa de rechazo ($22.14\%$), demostrando ser el modelo más robusto frente a variaciones sin descartar muestras innecesariamente.
*   La CNN Multi-Head (entrenada por 5 épocas en CPU) logró una muy buena precisión de calidad ($88.22\%$) y fruta ($68.48\%$). Es un excelente punto de partida que se beneficiaría de un entrenamiento más prolongado sobre GPU.

---

##  Interfaz Web y Simulación en Vivo

La aplicación web construida con Flask y Socket.IO destaca por su diseño premium y funcionalidad interactiva:
1.  **Panel de Diagnóstico**: Permite subir fotos de frutas o capturar en tiempo real con la webcam del dispositivo. Muestra la predicción de la fruta, estado de calidad, confianza de los estimadores, diámetro e indicador de tamaño físico.
2.  **Simulador de Faja Transportadora**: Las frutas se desplazan sobre una cinta en movimiento. La "cámara" del simulador las escanea de una en una y las clasifica. Los "brazos mecánicos" desvían las frutas a bandejas clasificadas según su calidad (Bueno, Malo o Normal).
3.  **Alarma de Objeto OOD**: Si la cámara detecta un objeto que no es fruta (filtro geométrico) o si la confianza del clasificador seleccionado (CNN, SVM o RF) cae por debajo del 70%, la cinta transportadora detiene su animación de inmediato, activa una señal lumínica de emergencia roja y reporta la anomalía.
4.  **Estadísticas de Producción en Vivo**: Integración de gráficos mediante `Chart.js` que muestran la distribución histórica de frutas procesadas de manera interactiva.

---

## Equipo de Trabajo

*   **Sebastian Cosme Benitez** (GitHub: [SebCos21])
*   **Rodolfo Morneo Gutierrez** (GitHub: [Matatrolls])
*   **Nicolas Gongora Rincon** (GitHub: [itsnigori98])
*   **Alejandro Quinones Caicedo** (GitHub: [Legnatbird])

*   **Asignatura**: Algoritmos y Programación III (APO III) - Universidad Icesi.