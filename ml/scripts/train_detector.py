#!/usr/bin/env python3
"""
Script para entrenar el modelo detector de medias usando TensorFlow Object Detection API.

Requisitos:
    pip install tensorflow==2.14.0
    pip install tensorflow-hub
    pip install tf-models-official

Uso:
    python train_detector.py --dataset_path dataset/processed --output_dir models/
"""

import argparse
import os
import json
from pathlib import Path

def parse_args():
    """Parse argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrenar detector de medias')
    parser.add_argument('--dataset_path', type=str, default='../dataset/processed',
                        help='Ruta al dataset procesado')
    parser.add_argument('--output_dir', type=str, default='../models',
                        help='Directorio de salida para el modelo')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Tamaño del batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Tasa de aprendizaje inicial')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Usar modelo pre-entrenado')
    parser.add_argument('--input_size', type=int, default=320,
                        help='Tamaño de entrada (320, 640, etc.)')
    return parser.parse_args()

def check_dependencies():
    """Verifica que las dependencias estén instaladas."""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow versión: {tf.__version__}")
    except ImportError:
        print("❌ Error: TensorFlow no está instalado")
        print("   Instala con: pip install tensorflow==2.14.0")
        return False
    
    try:
        import tensorflow_hub as hub
        print(f"✅ TensorFlow Hub disponible")
    except ImportError:
        print("⚠️  Advertencia: TensorFlow Hub no disponible")
    
    return True

def load_dataset_info(dataset_path: Path):
    """Carga información del dataset."""
    info_path = dataset_path / 'dataset_info.json'
    
    if not info_path.exists():
        print(f"❌ Error: {info_path} no existe")
        print("   Ejecuta primero: python scripts/prepare_dataset.py")
        return None
    
    with open(info_path, 'r') as f:
        return json.load(f)

def create_model_config(args, dataset_info):
    """Crea configuración del modelo."""
    config = {
        'model_name': 'ssd_mobilenet_v2_fpnlite_320x320',
        'input_size': args.input_size,
        'num_classes': dataset_info['num_classes'],
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'use_pretrained': args.pretrained
    }
    return config

def train_placeholder(args, dataset_info):
    """
    Placeholder para el entrenamiento real.
    
    En una implementación completa, este método:
    1. Cargaría el modelo base (SSD MobileNet V2)
    2. Configuraría el pipeline de datos
    3. Entrenaría el modelo
    4. Guardaría checkpoints
    5. Exportaría el modelo final
    """
    print("\n🏋️  Entrenando modelo detector...")
    print(f"   Configuración:")
    print(f"      - Modelo: SSD MobileNet V2 FPNLite")
    print(f"      - Tamaño de entrada: {args.input_size}x{args.input_size}")
    print(f"      - Número de clases: {dataset_info['num_classes']}")
    print(f"      - Batch size: {args.batch_size}")
    print(f"      - Learning rate: {args.learning_rate}")
    print(f"      - Épocas: {args.num_epochs}")
    
    # TODO: Implementar entrenamiento real
    # Este es un placeholder que muestra la estructura esperada
    
    print("\n⚠️  NOTA: Este es un script de ejemplo.")
    print("   Para entrenamiento real, necesitas:")
    print("   1. Implementar el pipeline de datos (TFRecord)")
    print("   2. Cargar el modelo pre-entrenado de TensorFlow Model Zoo")
    print("   3. Configurar el training loop")
    print("   4. Implementar callbacks (checkpoints, tensorboard, etc.)")
    print("   5. Exportar el modelo final a SavedModel y TFLite")
    
    print("\n📚 Referencias:")
    print("   - TF Object Detection API: https://github.com/tensorflow/models/tree/master/research/object_detection")
    print("   - Model Zoo: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md")
    print("   - Tutorial completo: https://tensorflow-object-detection-api-tutorial.readthedocs.io/")
    
    return True

def main():
    """Función principal."""
    args = parse_args()
    
    print("🚀 Iniciando entrenamiento del detector de medias...")
    
    # Verificar dependencias
    if not check_dependencies():
        return
    
    # Verificar dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Error: {dataset_path} no existe")
        print("   Ejecuta primero: python scripts/prepare_dataset.py")
        return
    
    # Cargar información del dataset
    dataset_info = load_dataset_info(dataset_path)
    if dataset_info is None:
        return
    
    print(f"\n📊 Dataset:")
    print(f"   Train: {dataset_info['split']['train']} imágenes")
    print(f"   Val:   {dataset_info['split']['val']} imágenes")
    print(f"   Test:  {dataset_info['split']['test']} imágenes")
    print(f"   Clases: {', '.join(dataset_info['class_names'])}")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Entrenar modelo
    success = train_placeholder(args, dataset_info)
    
    if success:
        # Guardar configuración
        config = create_model_config(args, dataset_info)
        config_path = output_dir / 'detector_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Configuración guardada: {config_path}")
        print(f"\n💡 Siguiente paso: python scripts/export_to_tflite.py")

if __name__ == '__main__':
    main()


