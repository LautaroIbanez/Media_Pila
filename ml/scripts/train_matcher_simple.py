#!/usr/bin/env python3
"""
Script simplificado para entrenar el modelo de emparejamiento de medias.
Esta es una versión de demostración que crea modelos dummy para el pipeline.
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np

def parse_args():
    """Parse argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrenar modelo de emparejamiento (simplificado)')
    parser.add_argument('--dataset_path', type=str, default='../dataset/processed',
                        help='Ruta al dataset procesado')
    parser.add_argument('--output_dir', type=str, default='../models',
                        help='Directorio de salida para el modelo')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Tamaño del batch')
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='Tamaño del embedding')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Tasa de aprendizaje inicial')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='Margen para triplet loss')
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

def create_dummy_matcher(output_dir: Path, args, dataset_info):
    """Crea un modelo matcher dummy para demostración."""
    print("\nCreando modelo de emparejamiento dummy...")
    print(f"   Configuracion:")
    print(f"      - Arquitectura: Siamese Network (dummy)")
    print(f"      - Backbone: MobileNetV2 (dummy)")
    print(f"      - Embedding size: {args.embedding_size}")
    print(f"      - Batch size: {args.batch_size}")
    print(f"      - Learning rate: {args.learning_rate}")
    print(f"      - Triplet margin: {args.margin}")
    print(f"      - Epocas: {args.num_epochs}")
    
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear modelo dummy
    saved_model_dir = output_dir / 'matcher_saved_model'
    saved_model_dir.mkdir(exist_ok=True)
    
    # Crear archivo dummy que simula un modelo entrenado
    with open(saved_model_dir / 'dummy_matcher.txt', 'w') as f:
        f.write("Modelo matcher dummy para demostracion\n")
        f.write(f"Entrenado con {args.num_epochs} epocas\n")
        f.write(f"Embedding size: {args.embedding_size}\n")
        f.write(f"Input size: 64x64\n")
        f.write(f"Triplet margin: {args.margin}\n")
    
    # Crear configuración del modelo
    config = {
        'model_name': 'siamese_mobilenetv2',
        'embedding_size': args.embedding_size,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'triplet_margin': args.margin,
        'input_size': 64,
        'is_dummy': True,
        'training_date': '2025-01-10',
        'dataset_info': {
            'total_images': dataset_info['statistics']['total_images'],
            'total_objects': dataset_info['statistics']['total_objects'],
            'class_names': dataset_info['class_names']
        }
    }
    
    # Guardar configuración
    config_path = output_dir / 'matcher_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"   Configuracion guardada: {config_path}")
    print(f"   Modelo dummy creado: {saved_model_dir}")
    
    return True

def main():
    """Función principal."""
    args = parse_args()
    
    print("Iniciando entrenamiento del modelo de emparejamiento (simplificado)...")
    
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
    success = create_dummy_matcher(output_dir, args, dataset_info)
    
    if success:
        print(f"\nModelo de emparejamiento dummy creado exitosamente")
        print(f"\nSiguiente paso: python scripts/export_to_tflite_simple.py")

if __name__ == '__main__':
    main()
