#!/usr/bin/env python3
"""
Script simplificado para entrenar el modelo detector de medias.
Esta es una versión de demostración que crea modelos dummy para el pipeline.
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np

def parse_args():
    """Parse argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrenar detector de medias (simplificado)')
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
    parser.add_argument('--input_size', type=int, default=320,
                        help='Tamaño de entrada (320, 640, etc.)')
    return parser.parse_args()

def load_dataset_info(dataset_path: Path):
    """Carga información del dataset."""
    info_path = dataset_path / 'dataset_info.json'
    
    if not info_path.exists():
        print(f"Error: {info_path} no existe")
        print("   Ejecuta primero: python scripts/prepare_dataset.py")
        return None
    
    with open(info_path, 'r') as f:
        return json.load(f)

def create_dummy_model(output_dir: Path, args, dataset_info):
    """Crea un modelo dummy para demostración."""
    print("\nCreando modelo detector dummy...")
    print(f"   Configuracion:")
    print(f"      - Modelo: SSD MobileNet V2 FPNLite (dummy)")
    print(f"      - Tamano de entrada: {args.input_size}x{args.input_size}")
    print(f"      - Numero de clases: {dataset_info['num_classes']}")
    print(f"      - Batch size: {args.batch_size}")
    print(f"      - Learning rate: {args.learning_rate}")
    print(f"      - Epocas: {args.num_epochs}")
    
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear modelo dummy
    saved_model_dir = output_dir / 'saved_model'
    saved_model_dir.mkdir(exist_ok=True)
    
    # Crear archivo dummy que simula un modelo entrenado
    with open(saved_model_dir / 'dummy_detector.txt', 'w') as f:
        f.write("Modelo detector dummy para demostracion\n")
        f.write(f"Entrenado con {args.num_epochs} epocas\n")
        f.write(f"Input size: {args.input_size}x{args.input_size}\n")
        f.write(f"Numero de clases: {dataset_info['num_classes']}\n")
        f.write(f"Clases: {', '.join(dataset_info['class_names'])}\n")
    
    # Crear configuración del modelo
    config = {
        'model_name': 'ssd_mobilenet_v2_fpnlite_320x320',
        'input_size': args.input_size,
        'num_classes': dataset_info['num_classes'],
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'use_pretrained': True,
        'is_dummy': True,
        'training_date': '2025-01-10',
        'dataset_info': {
            'total_images': dataset_info['statistics']['total_images'],
            'total_objects': dataset_info['statistics']['total_objects'],
            'class_names': dataset_info['class_names']
        }
    }
    
    # Guardar configuración
    config_path = output_dir / 'detector_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"   Configuracion guardada: {config_path}")
    print(f"   Modelo dummy creado: {saved_model_dir}")
    
    return True

def main():
    """Función principal."""
    args = parse_args()
    
    print("Iniciando entrenamiento del detector de medias (simplificado)...")
    
    # Verificar dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: {dataset_path} no existe")
        print("   Ejecuta primero: python scripts/prepare_dataset.py")
        return
    
    # Cargar información del dataset
    dataset_info = load_dataset_info(dataset_path)
    if dataset_info is None:
        return
    
    print(f"\nDataset:")
    print(f"   Train: {dataset_info['split']['train']} imagenes")
    print(f"   Val:   {dataset_info['split']['val']} imagenes")
    print(f"   Test:  {dataset_info['split']['test']} imagenes")
    print(f"   Clases: {', '.join(dataset_info['class_names'])}")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear modelo dummy
    success = create_dummy_model(output_dir, args, dataset_info)
    
    if success:
        print(f"\nModelo detector dummy creado exitosamente")
        print(f"\nSiguiente paso: python scripts/train_matcher_simple.py")

if __name__ == '__main__':
    main()
