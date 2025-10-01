#!/usr/bin/env python3
"""
Script para crear una imagen de prueba para el dataset.
"""

from PIL import Image, ImageDraw
import os

def create_test_sock_image():
    """Crea una imagen de prueba con medias."""
    # Crear imagen de 640x480
    img = Image.new('RGB', (640, 480), color='white')
    draw = ImageDraw.Draw(img)
    
    # Dibujar dos medias simples
    # Media 1 (izquierda)
    draw.rectangle([100, 150, 200, 300], fill='red', outline='black', width=2)
    draw.rectangle([110, 160, 190, 290], fill='darkred')
    
    # Media 2 (derecha)
    draw.rectangle([300, 140, 400, 290], fill='red', outline='black', width=2)
    draw.rectangle([310, 150, 390, 280], fill='darkred')
    
    # Agregar algunos detalles
    draw.ellipse([120, 200, 180, 250], fill='white')  # Talón media 1
    draw.ellipse([320, 190, 380, 240], fill='white')  # Talón media 2
    
    return img

def main():
    """Función principal."""
    # Crear directorio si no existe
    output_dir = os.path.join('..', 'dataset', 'annotated')
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear imagen de prueba
    img = create_test_sock_image()
    
    # Guardar imagen
    output_path = os.path.join(output_dir, 'sample_socks.jpg')
    img.save(output_path, 'JPEG', quality=95)
    
    print(f"Imagen de prueba creada: {output_path}")
    print("Dimensiones: 640x480")
    print("Contenido: 2 medias rojas")

if __name__ == '__main__':
    main()
