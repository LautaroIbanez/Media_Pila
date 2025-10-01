# Machine Learning Pipeline para Detección de Medias

Este directorio contiene los recursos necesarios para entrenar y mejorar los modelos de detección y emparejamiento de medias.

## Estructura de Directorios

```
ml/
├── dataset/
│   ├── raw/                    # Imágenes originales sin procesar
│   ├── annotated/              # Imágenes anotadas con bounding boxes
│   └── dataset_info.json       # Metadatos del dataset
├── models/
│   ├── sock_detector.tflite   # Modelo TFLite para detección de medias
│   ├── sock_matcher.tflite    # Modelo TFLite para emparejamiento
│   └── model_metadata.json    # Información sobre los modelos
├── scripts/
│   ├── prepare_dataset.py     # Prepara y valida el dataset
│   ├── train_detector.py      # Entrena el modelo de detección
│   ├── train_matcher.py       # Entrena el modelo de emparejamiento
│   └── export_to_tflite.py    # Exporta modelos a TensorFlow Lite
└── README.md                   # Este archivo
```

## Requisitos

### Python y Dependencias

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install tensorflow==2.14.0
pip install tensorflow-hub
pip install pillow
pip install numpy
pip install opencv-python
pip install labelImg  # Para anotar imágenes
```

### Herramientas de Anotación

Recomendamos usar **LabelImg** para anotar manualmente las imágenes:

```bash
pip install labelImg
labelImg
```

## Flujo de Trabajo

### 1. Recolección de Datos

1. Coloca las imágenes de medias en `dataset/raw/`
2. Las imágenes deben tener buena iluminación y múltiples medias por imagen
3. Recomendado: al menos 500 imágenes con ~2000 instancias de medias

**Estructura recomendada:**
```
dataset/raw/
├── IMG_001.jpg
├── IMG_002.jpg
├── ...
```

### 2. Anotación de Datos

#### Para Detección de Objetos

Usa LabelImg para dibujar bounding boxes alrededor de cada media:

```bash
cd dataset
labelImg raw/ annotated/
```

**Configuración de clases:**
- `sock` - Para todas las medias individuales

Guarda las anotaciones en formato **PASCAL VOC** (.xml) o **YOLO** (.txt).

#### Para Emparejamiento

Crea un archivo JSON con pares de medias:

```json
{
  "pairs": [
    {
      "image": "IMG_001.jpg",
      "sock1": {"x": 100, "y": 150, "w": 80, "h": 120},
      "sock2": {"x": 250, "y": 140, "w": 85, "h": 125},
      "match": true
    }
  ]
}
```

### 3. Preparación del Dataset

Ejecuta el script de preparación:

```bash
python scripts/prepare_dataset.py
```

Este script:
- Valida las anotaciones
- Divide el dataset en train/val/test (70%/15%/15%)
- Genera archivos TFRecord para entrenamiento
- Crea estadísticas del dataset

### 4. Entrenamiento del Detector

**Versión Simplificada (Recomendada para Demostración):**
```bash
python scripts/train_detector_simple.py \
    --dataset_path dataset/processed \
    --output_dir models/ \
    --num_epochs 50 \
    --batch_size 16
```

**Versión Completa (Requiere TensorFlow Object Detection API):**
```bash
python scripts/train_detector.py \
    --dataset_path dataset/processed \
    --output_dir models/ \
    --num_epochs 50 \
    --batch_size 16
```

**Parámetros:**
- `--num_epochs`: Número de épocas de entrenamiento (default: 50)
- `--batch_size`: Tamaño del batch (default: 16)
- `--learning_rate`: Tasa de aprendizaje (default: 0.001)
- `--input_size`: Tamaño de entrada (default: 320)

**Modelo recomendado:** SSD MobileNet V2 FPNLite 320x320

### 5. Entrenamiento del Modelo de Similitud

**Versión Simplificada (Recomendada para Demostración):**
```bash
python scripts/train_matcher_simple.py \
    --dataset_path dataset/processed \
    --output_dir models/ \
    --num_epochs 30 \
    --embedding_size 128
```

**Versión Completa:**
```bash
python scripts/train_matcher.py \
    --dataset_path dataset/processed \
    --output_dir models/ \
    --num_epochs 30 \
    --embedding_size 128
```

**Arquitectura:**
- Red siamesa con MobileNetV2 como feature extractor
- Capa de embedding de 128 dimensiones
- Distancia coseno para similitud
- Triplet loss para entrenamiento

### 6. Exportación a TensorFlow Lite

Convierte los modelos a TFLite para uso en Android:

```bash
python scripts/export_to_tflite_simple.py \
    --detector_path models/saved_model \
    --matcher_path models/matcher_saved_model \
    --quantize \
    --optimize_gpu
```

**Opciones de cuantización:**
- `--quantize`: Cuantización post-entrenamiento (reduce tamaño ~4x)
- `--quantize_full`: Cuantización completa int8 (mejor rendimiento en móviles)
- `--optimize_gpu`: Optimización para GPU delegate

**Metadata de Modelos:**
El script genera automáticamente `model_metadata.json` con información completa sobre los modelos, incluyendo:
- Versión y fecha de creación
- Hashes SHA256 para verificación de integridad
- Configuración de cuantización y optimización
- Información de compatibilidad
- Metadatos del dataset de entrenamiento

### 7. Integración en la App Android

Una vez que tengas los modelos `.tflite`:

1. Copia los modelos y metadata a `app/src/main/assets/`:
```bash
cp models/sock_detector.tflite ../app/src/main/assets/
cp models/sock_matcher.tflite ../app/src/main/assets/
cp models/model_metadata.json ../app/src/main/assets/
```

2. Las dependencias ya están configuradas en `app/build.gradle.kts`:
```kotlin
dependencies {
    // TensorFlow Lite
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")
}
```

3. El `MLSockDetector` ya está implementado con validación automática de modelos.

### 8. Validación de Modelos y Fallback Automático

El sistema incluye validación automática de modelos con fallback al detector heurístico:

**Validaciones Automáticas:**
- Verificación de existencia de archivos de modelo
- Validación de metadata JSON
- Verificación de hashes SHA256 para integridad
- Detección de modelos dummy (no aptos para producción)
- Verificación de compatibilidad con la versión de Android

**Comportamiento del Sistema:**
1. Al inicializar, `SockDetectionViewModel` valida automáticamente los modelos
2. Si los modelos son válidos, se usa `MLSockDetector`
3. Si los modelos no son válidos o son dummy, se usa `SockDetector` (heurístico)
4. El usuario puede revalidar manualmente con `revalidateModels()`

**Formato de Metadata (`model_metadata.json`):**
```json
{
  "version": "1.0.0",
  "created_date": "2025-01-10T18:00:00",
  "is_dummy": false,
  "models": {
    "detector": {
      "name": "ssd_mobilenet_v2_fpnlite_320x320",
      "input_size": 320,
      "file_hash": "sha256_hash_here",
      "quantized": true,
      "quantization_type": "dynamic"
    },
    "matcher": {
      "name": "siamese_mobilenetv2",
      "input_size": 64,
      "embedding_size": 128,
      "file_hash": "sha256_hash_here"
    }
  },
  "compatibility": {
    "tensorflow_lite_version": "2.14.0",
    "android_min_sdk": 21,
    "input_format": "RGB",
    "normalization": "0-1"
  }
}
```

**Para QA y Release:**
- Los modelos marcados como `"is_dummy": true` serán rechazados automáticamente
- Solo modelos con `"is_dummy": false` y hashes válidos se usarán en producción
- El sistema registra la razón del rechazo en los logs para debugging

## Métricas de Evaluación

### Para Detección
- **mAP (mean Average Precision)**: > 0.85 recomendado
- **IoU (Intersection over Union)**: > 0.5 para true positive
- **FPS**: > 10 en dispositivos móviles

### Para Emparejamiento
- **Accuracy**: > 90%
- **Precision/Recall**: > 0.85/0.85
- **F1-Score**: > 0.87

## Dataset Públicos Recomendados

Aunque no existen datasets específicos de medias, puedes usar:

1. **COCO Dataset**: Para pre-entrenamiento general
2. **Open Images**: Objetos comunes para transfer learning
3. **Crear tu propio dataset**: ¡Lo más recomendado para este caso específico!

## Consejos de Mejora

### Aumento de Datos (Data Augmentation)
```python
augmentation_config = {
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'brightness_range': [0.8, 1.2]
}
```

### Optimización de Modelos
- Usa cuantización para reducir tamaño
- Prueba diferentes backbones (EfficientNet, MobileNetV3)
- Implementa NMS (Non-Maximum Suppression) eficiente
- Usa GPU delegate para inferencia más rápida

### Transfer Learning
- Comienza con modelos pre-entrenados en COCO
- Fine-tune solo las últimas capas primero
- Gradualmente desbloquea capas anteriores

## Solución de Problemas

**Error: "Out of memory"**
- Reduce batch_size
- Reduce resolución de entrada
- Usa cuantización

**Detecciones con baja confianza**
- Aumenta el dataset
- Mejora las anotaciones
- Ajusta el threshold de confianza
- Usa más data augmentation

**Modelo muy lento en Android**
- Usa cuantización int8
- Reduce resolución de entrada
- Usa GPU delegate
- Considera usar NNAPI

## Referencias

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [ML Kit Object Detection](https://developers.google.com/ml-kit/vision/object-detection)
- [MediaPipe Solutions](https://google.github.io/mediapipe/)
- [Siamese Networks Paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

## Contribución

¿Mejoraste el modelo? ¡Comparte tus resultados!

1. Documenta tu configuración
2. Reporta métricas de evaluación
3. Comparte insights del entrenamiento
4. Considera publicar el modelo entrenado

---

**Nota:** El entrenamiento de modelos ML requiere tiempo y recursos computacionales. Se recomienda usar GPU para entrenamientos largos. Puedes usar Google Colab para acceso gratuito a GPUs.



