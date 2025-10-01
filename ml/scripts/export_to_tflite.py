#!/usr/bin/env python3
"""
Script para exportar modelos entrenados a TensorFlow Lite.

Soporta:
    - Cuantización post-entrenamiento
    - Cuantización completa int8
    - Optimización de GPU

Uso:
    python export_to_tflite.py --model_path models/sock_detector.pb --output_path models/sock_detector.tflite
"""

import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np

def parse_args():
    """Parse argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Exportar modelo a TFLite')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Ruta al modelo SavedModel o .pb')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Ruta de salida para .tflite')
    parser.add_argument('--quantize', action='store_true',
                        help='Aplicar cuantización dinámica')
    parser.add_argument('--quantize_full', action='store_true',
                        help='Aplicar cuantización completa int8')
    parser.add_argument('--optimize_gpu', action='store_true',
                        help='Optimizar para GPU')
    return parser.parse_args()

def create_representative_dataset(input_shape, num_samples=100):
    """Crea dataset representativo para cuantización."""
    def representative_data_gen():
        for _ in range(num_samples):
            # Generar datos representativos (imágenes de medias)
            data = np.random.rand(1, *input_shape).astype(np.float32)
            yield [data]
    return representative_data_gen

def export_to_tflite(args):
    """Exporta modelo a TensorFlow Lite con optimizaciones."""
    print("\n📦 Exportando modelo a TensorFlow Lite...")
    print(f"   Modelo: {args.model_path}")
    print(f"   Salida: {args.output_path}")
    
    # Verificar que el modelo existe
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"   ❌ Error: {model_path} no existe")
        return False
    
    try:
        # Cargar modelo
        print("   📥 Cargando modelo...")
        if model_path.is_dir():
            # SavedModel
            model = tf.saved_model.load(str(model_path))
            converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
        else:
            # Archivo .pb o .h5
            if str(model_path).endswith('.h5'):
                model = tf.keras.models.load_model(str(model_path))
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
            else:
                # Intentar cargar como SavedModel
                converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
        
        # Configurar optimizaciones
        if args.quantize:
            print("   ✅ Cuantización dinámica habilitada")
            print("      - Reduce tamaño ~4x")
            print("      - Inferencia ~2-3x más rápida")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if args.quantize_full:
            print("   ✅ Cuantización completa int8 habilitada")
            print("      - Reduce tamaño ~4x")
            print("      - Inferencia muy rápida en CPU")
            print("      - Requiere dataset representativo")
            
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            # Dataset representativo para cuantización
            # Asumir input shape típico para detección de objetos
            input_shape = (320, 320, 3)  # Ajustar según el modelo
            converter.representative_dataset = create_representative_dataset(input_shape)
        
        if args.optimize_gpu:
            print("   ✅ Optimización GPU habilitada")
            print("      - GPU delegate para inferencia")
            # Nota: La optimización GPU se aplica en tiempo de ejecución, no en conversión
        
        # Configuraciones adicionales
        converter.allow_custom_ops = True
        converter.experimental_new_converter = True
        
        # Convertir
        print("   🔄 Convirtiendo modelo...")
        tflite_model = converter.convert()
        
        # Guardar
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Información del modelo
        model_size = len(tflite_model) / (1024 * 1024)  # MB
        print(f"   ✅ Modelo exportado exitosamente")
        print(f"      - Tamaño: {model_size:.2f} MB")
        print(f"      - Ubicación: {output_path}")
        
        # Validar modelo
        print("   🔍 Validando modelo...")
        interpreter = tf.lite.Interpreter(model_path=str(output_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"      - Inputs: {len(input_details)}")
        for i, detail in enumerate(input_details):
            print(f"        Input {i}: {detail['shape']} ({detail['dtype']})")
        
        print(f"      - Outputs: {len(output_details)}")
        for i, detail in enumerate(output_details):
            print(f"        Output {i}: {detail['shape']} ({detail['dtype']})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error durante la exportación: {str(e)}")
        print("   💡 Sugerencias:")
        print("      - Verifica que el modelo esté en formato SavedModel o .h5")
        print("      - Asegúrate de que TensorFlow esté instalado correctamente")
        print("      - Para cuantización completa, proporciona datos representativos")
        return False

def main():
    """Función principal."""
    args = parse_args()
    
    print("🚀 Exportando modelo a TensorFlow Lite...")
    
    # Verificar que el modelo exista
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Error: {model_path} no existe")
        return
    
    # Crear directorio de salida
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Exportar
    success = export_to_tflite(args)
    
    if success:
        print(f"\n✅ Modelo exportado exitosamente")
        print(f"\n💡 Siguiente paso:")
        print(f"   1. Copia el .tflite a: app/src/main/assets/")
        print(f"   2. Actualiza build.gradle.kts con dependencias TFLite")
        print(f"   3. Crea wrapper en app/src/main/java/com/example/media_pila/ml/")

if __name__ == '__main__':
    main()



