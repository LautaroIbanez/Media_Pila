#!/usr/bin/env python3
"""
Script para entrenar el modelo de emparejamiento de medias usando una red siamesa.

Arquitectura:
    - Feature extractor: MobileNetV2 pre-entrenado
    - Embedding layer: 128 dimensiones
    - Loss: Triplet loss
    - Métrica: Similitud coseno

Uso:
    python train_matcher.py --dataset_path dataset/processed --output_dir models/
"""

import argparse
import os
import json
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import random

def parse_args():
    """Parse argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrenar modelo de emparejamiento')
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

def create_siamese_model(input_size, embedding_size):
    """Crea modelo siamesa con MobileNetV2 como backbone."""
    # Backbone: MobileNetV2 pre-entrenado
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar las primeras capas del backbone
    for layer in backbone.layers[:-20]:
        layer.trainable = False
    
    # Capas de embedding
    global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
    dropout = tf.keras.layers.Dropout(0.2)
    dense = tf.keras.layers.Dense(embedding_size, activation='linear')
    l2_norm = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))
    
    # Construir modelo completo
    input_layer = tf.keras.layers.Input(shape=(input_size, input_size, 3))
    x = backbone(input_layer)
    x = global_avg_pool(x)
    x = dropout(x)
    x = dense(x)
    output = l2_norm(x)
    
    model = tf.keras.Model(input_layer, output)
    return model

def triplet_loss(y_true, y_pred, margin=0.5):
    """Implementa triplet loss para entrenamiento siamesa."""
    anchor, positive, negative = tf.split(y_pred, 3, axis=0)
    
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(loss)

def create_triplet_dataset(dataset_path, batch_size, input_size):
    """Crea dataset de tripletes para entrenamiento."""
    # En una implementación real, aquí cargarías las imágenes
    # y crearías tripletes (anchor, positive, negative)
    
    # Para demostración, creamos datos dummy
    def generate_dummy_batch():
        # Generar imágenes dummy
        anchor = tf.random.normal((batch_size, input_size, input_size, 3))
        positive = anchor + tf.random.normal((batch_size, input_size, input_size, 3)) * 0.1
        negative = tf.random.normal((batch_size, input_size, input_size, 3))
        
        return tf.concat([anchor, positive, negative], axis=0)
    
    dataset = tf.data.Dataset.from_generator(
        generate_dummy_batch,
        output_signature=tf.TensorSpec(shape=(batch_size * 3, input_size, input_size, 3), dtype=tf.float32)
    )
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_matcher(args):
    """Entrena el modelo de emparejamiento usando red siamesa."""
    print("\n🏋️  Entrenando modelo de emparejamiento...")
    print(f"   Configuración:")
    print(f"      - Arquitectura: Siamese Network")
    print(f"      - Backbone: MobileNetV2")
    print(f"      - Embedding size: {args.embedding_size}")
    print(f"      - Batch size: {args.batch_size}")
    print(f"      - Learning rate: {args.learning_rate}")
    print(f"      - Triplet margin: {args.margin}")
    print(f"      - Épocas: {args.num_epochs}")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear modelo
    model = create_siamese_model(64, args.embedding_size)
    
    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=triplet_loss,
        metrics=['accuracy']
    )
    
    print(f"   ✅ Modelo creado: {model.count_params()} parámetros")
    
    # Crear dataset
    train_dataset = create_triplet_dataset(args.dataset_path, args.batch_size, 64)
    
    # Configurar callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'matcher_checkpoints' / 'ckpt-{epoch:02d}'),
            save_best_only=True,
            monitor='loss'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_dir / 'matcher_logs'),
            histogram_freq=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    print("   🚀 Iniciando entrenamiento...")
    print("   ⚠️  NOTA: Este es un entrenamiento simulado.")
    print("   Para entrenamiento real, necesitas:")
    print("   1. Dataset de pares de medias (matching/non-matching)")
    print("   2. Implementar generación de tripletes real")
    print("   3. Configurar data augmentation")
    print("   4. Validación con métricas de similitud")
    
    # Simular entrenamiento
    import time
    for epoch in range(min(3, args.num_epochs)):  # Solo 3 épocas para demo
        print(f"   Época {epoch + 1}/{args.num_epochs}...")
        time.sleep(1)  # Simular tiempo de entrenamiento
    
    print("   ✅ Entrenamiento completado (simulado)")
    
    # Guardar modelo
    model_path = output_dir / 'saved_model'
    model_path.mkdir(exist_ok=True)
    
    # Crear archivo dummy para indicar que el modelo está "entrenado"
    with open(model_path / 'dummy_matcher.txt', 'w') as f:
        f.write("Modelo matcher dummy para demostración")
    
    return True

def main():
    """Función principal."""
    args = parse_args()
    
    print("🚀 Iniciando entrenamiento del modelo de emparejamiento...")
    
    # Verificar dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Error: {dataset_path} no existe")
        print("   Ejecuta primero: python scripts/prepare_dataset.py")
        return
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Entrenar modelo
    success = train_matcher(args)
    
    if success:
        # Guardar configuración
        config = {
            'model_name': 'siamese_mobilenetv2',
            'embedding_size': args.embedding_size,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
            'triplet_margin': args.margin
        }
        
        config_path = output_dir / 'matcher_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Configuración guardada: {config_path}")
        print(f"\n💡 Siguiente paso: python scripts/export_to_tflite.py")

if __name__ == '__main__':
    main()



