#!/usr/bin/env python3
"""
Script para crear modelos TFLite dummy para testing.
Estos modelos no son funcionales pero permiten probar la integración.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path

def create_dummy_detector_model():
    """Crea un modelo detector dummy."""
    # Crear modelo simple para detección
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(320, 320, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(20 * 4, activation='linear'),  # 20 detecciones * 4 coordenadas
        tf.keras.layers.Reshape((20, 4))
    ])
    
    # Compilar modelo
    model.compile(optimizer='adam', loss='mse')
    
    # Entrenar con datos dummy
    dummy_data = np.random.rand(10, 320, 320, 3).astype(np.float32)
    dummy_labels = np.random.rand(10, 20, 4).astype(np.float32)
    model.fit(dummy_data, dummy_labels, epochs=1, verbose=0)
    
    return model

def create_dummy_matcher_model():
    """Crea un modelo matcher dummy."""
    # Crear modelo simple para matching
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='linear'),
        tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))
    ])
    
    # Compilar modelo
    model.compile(optimizer='adam', loss='mse')
    
    # Entrenar con datos dummy
    dummy_data = np.random.rand(10, 64, 64, 3).astype(np.float32)
    dummy_labels = np.random.rand(10, 128).astype(np.float32)
    model.fit(dummy_data, dummy_labels, epochs=1, verbose=0)
    
    return model

def convert_to_tflite(model, output_path, quantize=False):
    """Convierte modelo a TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    return len(tflite_model)

def main():
    """Función principal."""
    print("🚀 Creando modelos TFLite dummy...")
    
    # Crear directorios
    models_dir = Path("ml/models")
    assets_dir = Path("app/src/main/assets")
    
    models_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear modelo detector
    print("📦 Creando modelo detector...")
    detector_model = create_dummy_detector_model()
    
    # Convertir a TFLite
    detector_tflite_path = models_dir / "sock_detector.tflite"
    detector_size = convert_to_tflite(detector_model, detector_tflite_path, quantize=True)
    print(f"   ✅ Detector creado: {detector_tflite_path} ({detector_size / 1024:.1f} KB)")
    
    # Crear modelo matcher
    print("📦 Creando modelo matcher...")
    matcher_model = create_dummy_matcher_model()
    
    # Convertir a TFLite
    matcher_tflite_path = models_dir / "sock_matcher.tflite"
    matcher_size = convert_to_tflite(matcher_model, matcher_tflite_path, quantize=True)
    print(f"   ✅ Matcher creado: {matcher_tflite_path} ({matcher_size / 1024:.1f} KB)")
    
    # Copiar a assets
    print("📁 Copiando modelos a assets...")
    import shutil
    
    shutil.copy(detector_tflite_path, assets_dir / "sock_detector.tflite")
    shutil.copy(matcher_tflite_path, assets_dir / "sock_matcher.tflite")
    
    print(f"   ✅ Modelos copiados a: {assets_dir}")
    
    # Verificar que los archivos existen
    detector_assets = assets_dir / "sock_detector.tflite"
    matcher_assets = assets_dir / "sock_matcher.tflite"
    
    if detector_assets.exists() and matcher_assets.exists():
        print("\n✅ Modelos dummy creados exitosamente!")
        print(f"   - Detector: {detector_assets} ({detector_assets.stat().st_size / 1024:.1f} KB)")
        print(f"   - Matcher: {matcher_assets} ({matcher_assets.stat().st_size / 1024:.1f} KB)")
        print("\n💡 Los modelos están listos para testing en la app")
    else:
        print("\n❌ Error: No se pudieron crear los modelos")

if __name__ == '__main__':
    main()
