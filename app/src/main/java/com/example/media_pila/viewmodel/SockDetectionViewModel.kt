package com.example.media_pila.viewmodel

import android.app.Application
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewModelScope
import com.example.media_pila.data.*
import com.example.media_pila.image.SockDetector
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class SockDetectionViewModel(application: Application) : AndroidViewModel(application) {
    
    private val sockDetector = SockDetector()
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private lateinit var cameraExecutor: ExecutorService
    
    // Estados
    private val _appState = MutableStateFlow<AppState>(AppState.Loading)
    val appState: StateFlow<AppState> = _appState.asStateFlow()
    
    private val _detectionResult = MutableStateFlow<DetectionResult?>(null)
    val detectionResult: StateFlow<DetectionResult?> = _detectionResult.asStateFlow()
    
    private val _isDetecting = MutableStateFlow(false)
    val isDetecting: StateFlow<Boolean> = _isDetecting.asStateFlow()
    
    private val _frameWidth = MutableStateFlow(0)
    val frameWidth: StateFlow<Int> = _frameWidth.asStateFlow()
    
    private val _frameHeight = MutableStateFlow(0)
    val frameHeight: StateFlow<Int> = _frameHeight.asStateFlow()
    
    init {
        cameraExecutor = Executors.newSingleThreadExecutor()
        checkCameraPermission()
    }
    
    /**
     * Verifica si se tienen permisos de cámara
     */
    fun checkCameraPermission() {
        val context = getApplication<Application>()
        when {
            ContextCompat.checkSelfPermission(
                context,
                android.Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                _appState.value = AppState.Detecting
                // startCamera() será llamado desde la UI con el lifecycleOwner
            }
            else -> {
                _appState.value = AppState.CameraPermissionRequired
            }
        }
    }
    
    /**
     * Inicia la cámara con CameraX
     */
    fun startCamera(lifecycleOwner: LifecycleOwner) {
        val context = getApplication<Application>()
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            // Crear preview (sin SurfaceProvider aquí, se asignará en setPreviewView)
            val preview = Preview.Builder()
                .build()
            
            // Crear imageAnalyzer
            imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImage(imageProxy)
                    }
                }
            
            try {
                cameraProvider.unbindAll()
                
                // Asegurar que imageAnalyzer no sea null antes de bindToLifecycle
                val analyzer = imageAnalyzer
                if (analyzer != null) {
                    camera = cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        analyzer
                    )
                    
                    _appState.value = AppState.Detecting
                } else {
                    _appState.value = AppState.Error("Error: ImageAnalyzer no pudo ser inicializado")
                }
                
            } catch (exc: Exception) {
                _appState.value = AppState.Error("Error al iniciar la cámara: ${exc.message}")
            }
            
        }, ContextCompat.getMainExecutor(context))
    }
    
    /**
     * Convierte ImageProxy a Bitmap de forma segura
     */
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val image = imageProxy.image
            if (image == null) return null
            
            val width = image.width
            val height = image.height
            
            if (width <= 0 || height <= 0) return null
            
            // Obtener los planes de la imagen YUV
            val planes = image.planes
            val yBuffer = planes[0].buffer
            val uBuffer = planes[1].buffer
            val vBuffer = planes[2].buffer
            
            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()
            
            // Crear arrays para los datos YUV
            val nv21 = ByteArray(ySize + uSize + vSize)
            
            // Copiar datos Y
            yBuffer.get(nv21, 0, ySize)
            
            // Copiar datos U y V
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)
            
            // Crear YuvImage
            val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
            
            // Convertir a Bitmap
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
            val imageBytes = out.toByteArray()
            
            // Crear Bitmap desde bytes
            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            
            // Rotar el bitmap si es necesario (la cámara puede estar rotada)
            val rotatedBitmap = if (imageProxy.imageInfo.rotationDegrees != 0) {
                val matrix = Matrix()
                matrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                val rotated = Bitmap.createBitmap(
                    bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
                )
                // Solo reciclar el bitmap original si se creó uno nuevo
                bitmap.recycle()
                rotated
            } else {
                bitmap
            }
            
            // YuvImage se libera automáticamente por el GC, no necesita recycle()
            
            rotatedBitmap
            
        } catch (e: Exception) {
            null
        }
    }
    
    /**
     * Procesa cada frame de la cámara
     */
    private fun processImage(imageProxy: ImageProxy) {
        if (_isDetecting.value) return
        
        try {
            // ✅ 1. Verificar que imageProxy.image no sea nulo
            val image = imageProxy.image
            if (image == null) {
                return
            }
            
            // ✅ 2. Verificar que el buffer tenga el tamaño correcto
            val width = image.width
            val height = image.height
            
            if (width <= 0 || height <= 0) {
                return
            }
            
            // ✅ 3. Actualizar dimensiones del frame
            _frameWidth.value = width
            _frameHeight.value = height
            
            // ✅ 4. Convertir a Bitmap ANTES de cerrar imageProxy
            val bitmap = imageProxyToBitmap(imageProxy)
            if (bitmap == null) {
                return
            }
            
            // ✅ 5. Ahora sí procesar en corrutina con el Bitmap
            viewModelScope.launch {
                try {
                    _isDetecting.value = true
                    
                    val result = sockDetector.detectSocks(
                        bitmap = bitmap,
                        frameWidth = width,
                        frameHeight = height
                    )
                    
                    _detectionResult.value = result
                    
                    if (result.socks.isNotEmpty()) {
                        _appState.value = AppState.Detected(result)
                    }
                    
                } catch (e: Exception) {
                    _appState.value = AppState.Error("Error al procesar imagen: ${e.message}")
                } finally {
                    _isDetecting.value = false
                    // ✅ Limpiar el bitmap después de usarlo (solo si es un Bitmap válido)
                    try {
                        if (bitmap != null && !bitmap.isRecycled) {
                            bitmap.recycle()
                        }
                    } catch (e: Exception) {
                        // Ignorar errores al reciclar
                    }
                }
            }
            
        } catch (e: Exception) {
            _appState.value = AppState.Error("Error al procesar frame: ${e.message}")
        } finally {
            // ✅ 6. Siempre cerrar imageProxy en finally
            try {
                imageProxy.close()
            } catch (e: Exception) {
                // Ignorar errores al cerrar imageProxy
            }
        }
    }
    
    /**
     * Reinicia la detección
     */
    fun retryDetection() {
        _detectionResult.value = null
        _appState.value = AppState.Detecting
        _isDetecting.value = false
    }
    
    /**
     * Simula guardar los pares detectados
     */
    fun savePairs() {
        val result = _detectionResult.value
        if (result != null && result.pairs.isNotEmpty()) {
            // Aquí se podría implementar la lógica para guardar en base de datos
            // Por ahora solo mostramos un mensaje de éxito
            _appState.value = AppState.Detected(result.copy())
        }
    }
    
    /**
     * Obtiene el ImageAnalysis para la UI
     */
    fun getImageAnalysis(): ImageAnalysis? = imageAnalyzer
    
    /**
     * Configura la vista previa de la cámara
     */
    fun setPreviewView(previewView: PreviewView, lifecycleOwner: LifecycleOwner) {
        val context = getApplication<Application>()
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            // Crear preview con SurfaceProvider asignado
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
            
            // Crear imageAnalyzer si no existe
            if (imageAnalyzer == null) {
                imageAnalyzer = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor) { imageProxy ->
                            processImage(imageProxy)
                        }
                    }
            }
            
            try {
                cameraProvider.unbindAll()
                
                // Asegurar que imageAnalyzer no sea null antes de bindToLifecycle
                val analyzer = imageAnalyzer
                if (analyzer != null) {
                    camera = cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        analyzer
                    )
                } else {
                    _appState.value = AppState.Error("Error: ImageAnalyzer no pudo ser inicializado")
                }
                
            } catch (exc: Exception) {
                _appState.value = AppState.Error("Error al configurar la vista previa: ${exc.message}")
            }
            
        }, ContextCompat.getMainExecutor(context))
    }
    
    override fun onCleared() {
        super.onCleared()
        cameraExecutor.shutdown()
    }
} 