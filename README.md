# Detector de Medias - Android App

Una aplicación Android desarrollada en Kotlin que utiliza CameraX y Jetpack Compose para detectar medias (calcetines) individuales y emparejarlas automáticamente basándose en similitud de colores.

## Características

- **Vista previa en tiempo real**: Utiliza CameraX para mostrar la cámara en tiempo real
- **Detección automática**: Analiza frames de la cámara para detectar medias sueltas
- **Emparejamiento inteligente**: Empareja medias basándose en similitud de color HSV
- **Interfaz moderna**: UI construida con Jetpack Compose y Material Design 3
- **Arquitectura MVVM**: Código modular y mantenible
- **Procesamiento en segundo plano**: Análisis de imagen sin bloquear la UI

## Tecnologías Utilizadas

- **Kotlin**: Lenguaje principal
- **Jetpack Compose**: UI declarativa
- **CameraX**: API moderna para cámara
- **ViewModel & LiveData**: Arquitectura MVVM
- **Coroutines**: Programación asíncrona
- **Palette API**: Análisis de colores dominantes
- **Material Design 3**: Sistema de diseño moderno

## Estructura del Proyecto

```
app/src/main/java/com/example/media_pila/
├── data/
│   └── Models.kt                 # Modelos de datos (Sock, SockPair, etc.)
├── image/
│   └── SockDetector.kt           # Procesador de imágenes y detección
├── ui/
│   ├── components/
│   │   ├── CameraPreview.kt      # Componente de vista previa de cámara
│   │   └── DetectionOverlay.kt   # Overlay para mostrar detecciones
│   └── screens/
│       └── MainScreen.kt         # Pantalla principal
├── viewmodel/
│   └── SockDetectionViewModel.kt # ViewModel principal
└── MainActivity.kt               # Actividad principal
```

## Algoritmo de Detección

### 1. Análisis de Regiones
- Divide la imagen en una cuadrícula de 8x8
- Analiza cada región para características de medias
- Filtra regiones con baja saturación (fondos)

### 2. Detección de Medias
- Utiliza Palette API para extraer colores dominantes
- Calcula confianza basada en:
  - Variación de color (40%)
  - Saturación del color dominante (30%)
  - Tamaño de la región (30%)

### 3. Emparejamiento
- Compara medias usando espacio de color HSV
- Calcula similitud considerando:
  - Diferencia de tono (Hue) - 60%
  - Diferencia de saturación - 20%
  - Diferencia de valor - 20%
- Empareja medias con confianza > 60%

## Instalación y Uso

### Requisitos
- Android Studio Arctic Fox o superior
- Android SDK 24+ (API Level 24)
- Dispositivo Android con cámara

### Pasos de Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone <repository-url>
   cd Media_pila
   ```

2. **Abrir en Android Studio**:
   - Abrir Android Studio
   - Seleccionar "Open an existing project"
   - Navegar a la carpeta del proyecto

3. **Sincronizar dependencias**:
   - Esperar a que Gradle sincronice las dependencias
   - Resolver cualquier error de dependencias si es necesario

4. **Ejecutar la aplicación**:
   - Conectar un dispositivo Android o usar un emulador
   - Presionar "Run" (▶️) en Android Studio

### Uso de la Aplicación

1. **Permisos**: La app solicitará permiso de cámara al iniciar
2. **Detección**: Apunta la cámara hacia medias sueltas
3. **Visualización**: Las medias detectadas se mostrarán con bounding boxes
4. **Emparejamiento**: Los pares se conectarán con líneas de colores
5. **Controles**:
   - **Reintentar**: Reinicia la detección
   - **Guardar Pares**: Simula guardar los pares detectados

## Configuración de Dependencias

Las dependencias principales están configuradas en `app/build.gradle.kts`:

```kotlin
// CameraX
implementation("androidx.camera:camera-core:1.3.1")
implementation("androidx.camera:camera-camera2:1.3.1")
implementation("androidx.camera:camera-lifecycle:1.3.1")
implementation("androidx.camera:camera-view:1.3.1")

// ViewModel y LiveData
implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
implementation("androidx.lifecycle:lifecycle-runtime-compose:2.7.0")

// Coroutines
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

// Image processing
implementation("androidx.palette:palette-ktx:1.0.0")
```

## Permisos Requeridos

La aplicación requiere los siguientes permisos (ya configurados en `AndroidManifest.xml`):

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" 
    android:maxSdkVersion="28" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" 
    android:maxSdkVersion="32" />
```

## Mejoras Futuras

### Integración con ML Kit
Para mejorar la detección, se puede integrar ML Kit:

```kotlin
// Agregar dependencias
implementation("com.google.mlkit:object-detection:17.0.0")
implementation("com.google.mlkit:object-detection-custom:17.0.0")

// Usar ObjectDetection para detección más precisa
val detector = ObjectDetection.getClient(ObjectDetectorOptions.Builder()
    .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
    .enableMultipleObjects()
    .build())
```

### Integración con OpenCV
Para procesamiento de imagen más avanzado:

```kotlin
// Agregar OpenCV Android SDK
implementation("org.opencv:opencv-android:4.8.0")

// Usar para detección de contornos y análisis de textura
```

### Base de Datos Local
Para persistir pares detectados:

```kotlin
// Room Database
implementation("androidx.room:room-runtime:2.6.1")
implementation("androidx.room:room-ktx:2.6.1")
kapt("androidx.room:room-compiler:2.6.1")
```

## Solución de Problemas

### Error de Permisos
Si la app no solicita permisos de cámara:
- Verificar que `AndroidManifest.xml` incluya los permisos
- Asegurar que el dispositivo tenga cámara disponible

### Problemas de Rendimiento
Si la detección es lenta:
- Reducir la frecuencia de análisis en `ImageAnalysis`
- Optimizar el tamaño de la cuadrícula en `SockDetector`
- Usar `STRATEGY_KEEP_ONLY_LATEST` para evitar colas

### Errores de Compilación
Si hay errores de dependencias:
- Sincronizar proyecto con Gradle
- Verificar versiones de dependencias
- Limpiar y reconstruir el proyecto

## Contribución

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Contacto

Para preguntas o sugerencias, por favor abrir un issue en el repositorio. 