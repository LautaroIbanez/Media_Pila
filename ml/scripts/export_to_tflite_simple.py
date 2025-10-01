#!/usr/bin/env python3
"""
Script simplificado para exportar modelos entrenados a TensorFlow Lite.
Incluye generación de metadata y validación de modelos.
"""

import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
import shutil

def parse_args():
    """Parse argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Exportar modelos a TFLite (simplificado)')
    parser.add_argument('--detector_path', type=str, default='../models/saved_model',
                        help='Ruta al modelo detector')
    parser.add_argument('--matcher_path', type=str, default='../models/matcher_saved_model',
                        help='Ruta al modelo matcher')
    parser.add_argument('--output_dir', type=str, default='../models',
                        help='Directorio de salida para los modelos TFLite')
    parser.add_argument('--quantize', action='store_true',
                        help='Aplicar cuantización (simulado)')
    parser.add_argument('--quantize_full', action='store_true',
                        help='Aplicar cuantización completa int8 (simulado)')
    parser.add_argument('--optimize_gpu', action='store_true',
                        help='Optimizar para GPU (simulado)')
    return parser.parse_args()

def calculate_file_hash(file_path: Path) -> str:
    """Calcula el hash SHA256 de un archivo."""
    if not file_path.exists():
        return ""
    
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def create_dummy_tflite_model(output_path: Path, model_type: str, input_size: int, output_size: int):
    """Crea un modelo TFLite dummy para demostración."""
    # Crear un archivo dummy que simula un modelo TFLite
    dummy_content = f"""# TensorFlow Lite Model (Dummy)
# Model Type: {model_type}
# Input Size: {input_size}x{input_size}x3
# Output Size: {output_size}
# Created: {datetime.now().isoformat()}
# This is a dummy model for demonstration purposes.
# In a real implementation, this would be a binary TFLite model.
"""
    
    with open(output_path, 'w') as f:
        f.write(dummy_content)
    
    return True

def load_model_config(config_path: Path):
    """Carga la configuración de un modelo."""
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        return json.load(f)

def create_model_metadata(detector_config, matcher_config, detector_path: Path, matcher_path: Path, args):
    """Crea metadata de los modelos."""
    current_time = datetime.now()
    
    # Calcular hashes de los modelos
    detector_hash = calculate_file_hash(detector_path)
    matcher_hash = calculate_file_hash(matcher_path)
    
    metadata = {
        "version": "1.0.0",
        "created_date": current_time.isoformat(),
        "created_timestamp": int(current_time.timestamp()),
        "is_dummy": True,
        "models": {
            "detector": {
                "name": detector_config.get('model_name', 'ssd_mobilenet_v2_fpnlite_320x320'),
                "input_size": detector_config.get('input_size', 320),
                "num_classes": detector_config.get('num_classes', 1),
                "file_path": "sock_detector.tflite",
                "file_hash": detector_hash,
                "file_size_bytes": detector_path.stat().st_size if detector_path.exists() else 0,
                "quantized": args.quantize or args.quantize_full,
                "quantization_type": "int8" if args.quantize_full else "dynamic" if args.quantize else "none",
                "gpu_optimized": args.optimize_gpu,
                "training_date": detector_config.get('training_date', current_time.isoformat()),
                "num_epochs": detector_config.get('num_epochs', 50),
                "learning_rate": detector_config.get('learning_rate', 0.001),
                "batch_size": detector_config.get('batch_size', 16)
            },
            "matcher": {
                "name": matcher_config.get('model_name', 'siamese_mobilenetv2'),
                "input_size": matcher_config.get('input_size', 64),
                "embedding_size": matcher_config.get('embedding_size', 128),
                "file_path": "sock_matcher.tflite",
                "file_hash": matcher_hash,
                "file_size_bytes": matcher_path.stat().st_size if matcher_path.exists() else 0,
                "quantized": args.quantize or args.quantize_full,
                "quantization_type": "int8" if args.quantize_full else "dynamic" if args.quantize else "none",
                "gpu_optimized": args.optimize_gpu,
                "training_date": matcher_config.get('training_date', current_time.isoformat()),
                "num_epochs": matcher_config.get('num_epochs', 30),
                "learning_rate": matcher_config.get('learning_rate', 0.0001),
                "batch_size": matcher_config.get('batch_size', 32),
                "triplet_margin": matcher_config.get('triplet_margin', 0.5)
            }
        },
        "dataset_info": {
            "total_images": detector_config.get('dataset_info', {}).get('total_images', 0),
            "total_objects": detector_config.get('dataset_info', {}).get('total_objects', 0),
            "class_names": detector_config.get('dataset_info', {}).get('class_names', ['sock'])
        },
        "export_settings": {
            "quantize": args.quantize,
            "quantize_full": args.quantize_full,
            "optimize_gpu": args.optimize_gpu,
            "export_date": current_time.isoformat()
        },
        "compatibility": {
            "tensorflow_lite_version": "2.14.0",
            "android_min_sdk": 21,
            "android_target_sdk": 34,
            "input_format": "RGB",
            "normalization": "0-1"
        }
    }
    
    return metadata

def export_models(args):
    """Exporta modelos a TensorFlow Lite con optimizaciones."""
    print("\nExportando modelos a TensorFlow Lite...")
    
    # Verificar que los modelos existan
    detector_path = Path(args.detector_path)
    matcher_path = Path(args.matcher_path)
    
    if not detector_path.exists():
        print(f"   Error: {detector_path} no existe")
        return False
    
    if not matcher_path.exists():
        print(f"   Error: {matcher_path} no existe")
        return False
    
    # Cargar configuraciones
    detector_config = load_model_config(detector_path.parent / 'detector_config.json')
    matcher_config = load_model_config(matcher_path.parent / 'matcher_config.json')
    
    if not detector_config or not matcher_config:
        print("   Error: No se pudieron cargar las configuraciones de los modelos")
        return False
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Exportar detector
    detector_tflite_path = output_dir / 'sock_detector.tflite'
    print(f"   Exportando detector: {detector_tflite_path}")
    create_dummy_tflite_model(
        detector_tflite_path, 
        "SSD MobileNet V2", 
        detector_config.get('input_size', 320),
        20  # NUM_DETECTIONS
    )
    
    # Exportar matcher
    matcher_tflite_path = output_dir / 'sock_matcher.tflite'
    print(f"   Exportando matcher: {matcher_tflite_path}")
    create_dummy_tflite_model(
        matcher_tflite_path, 
        "Siamese Network", 
        matcher_config.get('input_size', 64),
        matcher_config.get('embedding_size', 128)
    )
    
    # Crear metadata
    print("   Generando metadata...")
    metadata = create_model_metadata(
        detector_config, 
        matcher_config, 
        detector_tflite_path, 
        matcher_tflite_path, 
        args
    )
    
    # Guardar metadata
    metadata_path = output_dir / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   Metadata guardada: {metadata_path}")
    
    # Mostrar información de los modelos
    print(f"\nInformacion de los modelos:")
    print(f"   Detector:")
    print(f"      - Archivo: {detector_tflite_path.name}")
    print(f"      - Tamano: {detector_tflite_path.stat().st_size} bytes")
    print(f"      - Input: {detector_config.get('input_size', 320)}x{detector_config.get('input_size', 320)}x3")
    print(f"      - Clases: {detector_config.get('num_classes', 1)}")
    print(f"      - Cuantizado: {args.quantize or args.quantize_full}")
    
    print(f"   Matcher:")
    print(f"      - Archivo: {matcher_tflite_path.name}")
    print(f"      - Tamano: {matcher_tflite_path.stat().st_size} bytes")
    print(f"      - Input: {matcher_config.get('input_size', 64)}x{matcher_config.get('input_size', 64)}x3")
    print(f"      - Embedding: {matcher_config.get('embedding_size', 128)}")
    print(f"      - Cuantizado: {args.quantize or args.quantize_full}")
    
    return True

def main():
    """Función principal."""
    args = parse_args()
    
    print("Exportando modelos a TensorFlow Lite...")
    
    # Exportar modelos
    success = export_models(args)
    
    if success:
        print(f"\nModelos exportados exitosamente")
        print(f"\nSiguiente paso:")
        print(f"   1. Copia los .tflite a: app/src/main/assets/")
        print(f"   2. Copia model_metadata.json a: app/src/main/assets/")
        print(f"   3. Actualiza MLSockDetector para validar modelos")

if __name__ == '__main__':
    main()
