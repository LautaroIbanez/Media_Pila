#!/usr/bin/env python3
"""
Script para preparar el dataset de medias para entrenamiento.
- Valida anotaciones
- Divide en train/val/test
- Genera estadísticas
- Crea archivos TFRecord
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple
import shutil
from collections import defaultdict
import random

def parse_pascal_voc(xml_path: str) -> Dict:
    """Parse anotación en formato PASCAL VOC."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotation = {
        'filename': root.find('filename').text,
        'width': int(root.find('size/width').text),
        'height': int(root.find('size/height').text),
        'objects': []
    }
    
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        annotation['objects'].append({
            'name': obj.find('name').text,
            'xmin': int(bbox.find('xmin').text),
            'ymin': int(bbox.find('ymin').text),
            'xmax': int(bbox.find('xmax').text),
            'ymax': int(bbox.find('ymax').text)
        })
    
    return annotation

def validate_annotations(annotated_dir: str) -> Tuple[List[str], List[str]]:
    """Valida que todas las imágenes tengan anotaciones válidas."""
    valid_files = []
    invalid_files = []
    
    xml_files = list(Path(annotated_dir).glob('*.xml'))
    
    for xml_file in xml_files:
        try:
            annotation = parse_pascal_voc(str(xml_file))
            
            # Verificar que la imagen exista
            img_path = xml_file.parent / annotation['filename']
            if not img_path.exists():
                invalid_files.append(f"{xml_file.name}: imagen no encontrada")
                continue
            
            # Verificar que tenga al menos un objeto
            if len(annotation['objects']) == 0:
                invalid_files.append(f"{xml_file.name}: sin objetos anotados")
                continue
            
            # Verificar bounding boxes válidos
            for obj in annotation['objects']:
                if obj['xmax'] <= obj['xmin'] or obj['ymax'] <= obj['ymin']:
                    invalid_files.append(f"{xml_file.name}: bbox inválido")
                    break
            else:
                valid_files.append(str(xml_file))
        
        except Exception as e:
            invalid_files.append(f"{xml_file.name}: {str(e)}")
    
    return valid_files, invalid_files

def split_dataset(files: List[str], train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Divide el dataset en train/val/test."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Los ratios deben sumar 1"
    
    random.shuffle(files)
    total = len(files)
    
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    return {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

def generate_statistics(annotations: List[Dict]) -> Dict:
    """Genera estadísticas del dataset."""
    stats = {
        'total_images': len(annotations),
        'total_objects': 0,
        'class_distribution': defaultdict(int),
        'bbox_sizes': [],
        'objects_per_image': []
    }
    
    for ann in annotations:
        num_objects = len(ann['objects'])
        stats['total_objects'] += num_objects
        stats['objects_per_image'].append(num_objects)
        
        for obj in ann['objects']:
            stats['class_distribution'][obj['name']] += 1
            
            width = obj['xmax'] - obj['xmin']
            height = obj['ymax'] - obj['ymin']
            area = width * height
            stats['bbox_sizes'].append({
                'width': width,
                'height': height,
                'area': area
            })
    
    # Calcular promedios
    if stats['objects_per_image']:
        stats['avg_objects_per_image'] = sum(stats['objects_per_image']) / len(stats['objects_per_image'])
    
    if stats['bbox_sizes']:
        avg_width = sum(b['width'] for b in stats['bbox_sizes']) / len(stats['bbox_sizes'])
        avg_height = sum(b['height'] for b in stats['bbox_sizes']) / len(stats['bbox_sizes'])
        avg_area = sum(b['area'] for b in stats['bbox_sizes']) / len(stats['bbox_sizes'])
        
        stats['avg_bbox_width'] = avg_width
        stats['avg_bbox_height'] = avg_height
        stats['avg_bbox_area'] = avg_area
    
    return stats

def main():
    """Función principal."""
    print("🚀 Preparando dataset para entrenamiento...")
    
    # Rutas
    base_dir = Path(__file__).parent.parent
    annotated_dir = base_dir / 'dataset' / 'annotated'
    output_dir = base_dir / 'dataset' / 'processed'
    
    if not annotated_dir.exists():
        print(f"❌ Error: Directorio {annotated_dir} no existe")
        print("   Por favor, coloca las imágenes anotadas en dataset/annotated/")
        return
    
    # Validar anotaciones
    print("\n📝 Validando anotaciones...")
    valid_files, invalid_files = validate_annotations(annotated_dir)
    
    print(f"   ✅ Archivos válidos: {len(valid_files)}")
    if invalid_files:
        print(f"   ⚠️  Archivos inválidos: {len(invalid_files)}")
        for invalid in invalid_files[:5]:  # Mostrar primeros 5
            print(f"      - {invalid}")
        if len(invalid_files) > 5:
            print(f"      ... y {len(invalid_files) - 5} más")
    
    if len(valid_files) == 0:
        print("❌ Error: No se encontraron archivos válidos")
        return
    
    if len(valid_files) < 100:
        print(f"⚠️  Advertencia: Solo {len(valid_files)} imágenes. Recomendado: al menos 500")
    
    # Parsear todas las anotaciones válidas
    print("\n📊 Generando estadísticas...")
    annotations = [parse_pascal_voc(f) for f in valid_files]
    stats = generate_statistics(annotations)
    
    print(f"   Total de imágenes: {stats['total_images']}")
    print(f"   Total de objetos: {stats['total_objects']}")
    print(f"   Objetos por imagen (promedio): {stats.get('avg_objects_per_image', 0):.2f}")
    print(f"   Distribución de clases:")
    for class_name, count in stats['class_distribution'].items():
        print(f"      - {class_name}: {count}")
    
    # Dividir dataset
    print("\n✂️  Dividiendo dataset...")
    split = split_dataset(valid_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    print(f"   Train: {len(split['train'])} ({len(split['train'])/len(valid_files)*100:.1f}%)")
    print(f"   Val:   {len(split['val'])} ({len(split['val'])/len(valid_files)*100:.1f}%)")
    print(f"   Test:  {len(split['test'])} ({len(split['test'])/len(valid_files)*100:.1f}%)")
    
    # Crear directorios de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    for subset in ['train', 'val', 'test']:
        (output_dir / subset).mkdir(exist_ok=True)
    
    # Copiar archivos a subdirectorios
    print("\n📁 Organizando archivos...")
    for subset, files in split.items():
        for xml_path in files:
            xml_file = Path(xml_path)
            annotation = parse_pascal_voc(str(xml_file))
            img_file = xml_file.parent / annotation['filename']
            
            # Copiar XML y imagen
            shutil.copy(xml_file, output_dir / subset / xml_file.name)
            shutil.copy(img_file, output_dir / subset / img_file.name)
    
    # Guardar estadísticas y configuración
    output_info = {
        'statistics': {k: v for k, v in stats.items() if k != 'bbox_sizes'},
        'split': {k: len(v) for k, v in split.items()},
        'class_names': list(stats['class_distribution'].keys()),
        'num_classes': len(stats['class_distribution'])
    }
    
    info_path = output_dir / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(output_info, f, indent=2)
    
    print(f"\n✅ Dataset preparado exitosamente!")
    print(f"   Archivos procesados: {output_dir}")
    print(f"   Información guardada: {info_path}")
    print(f"\n💡 Siguiente paso: python scripts/train_detector.py")

if __name__ == '__main__':
    main()


