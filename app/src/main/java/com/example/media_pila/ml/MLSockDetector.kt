package com.example.media_pila.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.graphics.Color
import android.os.Build
import androidx.palette.graphics.Palette
import com.example.media_pila.data.Sock
import com.example.media_pila.data.DetectionResult
import com.example.media_pila.data.SockPair
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.*
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate

/**
 * Detector de medias usando modelos de Machine Learning (TensorFlow Lite).
 * 
 * NOTA: Esta es una clase de ejemplo/placeholder que muestra la estructura esperada.
 * Para uso real, necesitas:
 * 1. Entrenar modelos TFLite (sock_detector.tflite y sock_matcher.tflite)
 * 2. Colocar los modelos en app/src/main/assets/
 * 3. Implementar la carga e inferencia de los modelos
 * 
 * Dependencias requeridas (ya incluidas en build.gradle.kts):
 * - org.tensorflow:tensorflow-lite:2.14.0
 * - org.tensorflow:tensorflow-lite-support:0.4.4
 * - org.tensorflow:tensorflow-lite-gpu:2.14.0
 */
class MLSockDetector(private val context: Context) {
    
    private var detectorInterpreter: Interpreter? = null
    private var matcherInterpreter: Interpreter? = null
    private var detectorDelegate: GpuDelegate? = null
    private var matcherDelegate: GpuDelegate? = null
    
    private var isInitialized = false
    
    companion object {
        private const val DETECTOR_MODEL_PATH = "sock_detector.tflite"
        private const val MATCHER_MODEL_PATH = "sock_matcher.tflite"
        private const val MIN_CONFIDENCE = 0.5f
        private const val INPUT_SIZE = 320 // Tamaño de entrada del detector
        private const val MATCHER_INPUT_SIZE = 64 // Tamaño de entrada del matcher
        private const val NUM_DETECTIONS = 20 // Número máximo de detecciones esperado del modelo
        private const val EMBEDDING_SIZE = 128
    }
    
    /**
     * Inicializa los modelos ML.
     * Debe llamarse antes de usar el detector.
     */
    suspend fun initialize() = withContext(Dispatchers.IO) {
        try {
            println("🤖 [MLSockDetector] Inicializando modelos ML...")
            val detectorModel = loadModelFromAssets(DETECTOR_MODEL_PATH)
            val matcherModel = loadModelFromAssets(MATCHER_MODEL_PATH)
            
            val detectorOptions = Interpreter.Options().apply {
                setNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
            }
            val matcherOptions = Interpreter.Options().apply {
                setNumThreads((Runtime.getRuntime().availableProcessors() / 2).coerceAtLeast(1))
            }
            
            // Intentar usar GPU delegate para cada intérprete si está disponible
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                detectorDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                detectorDelegate?.let { detectorOptions.addDelegate(it) }
                matcherDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                matcherDelegate?.let { matcherOptions.addDelegate(it) }
                println("🤖 [MLSockDetector] GPU Delegate habilitado para detector y matcher")
            } else {
                println("🤖 [MLSockDetector] GPU Delegate no soportado, usando CPU")
            }
            
            detectorInterpreter = Interpreter(detectorModel, detectorOptions)
            println("🤖 [MLSockDetector] Detector inicializado (threads=${detectorOptions.numThreads})")
            matcherInterpreter = Interpreter(matcherModel, matcherOptions)
            println("🤖 [MLSockDetector] Matcher inicializado (threads=${matcherOptions.numThreads})")
            
            isInitialized = true
            println("🤖 [MLSockDetector] Modelos cargados exitosamente")
            
        } catch (e: Exception) {
            println("🤖 [MLSockDetector] Error al cargar modelos: ${e.message}")
            println("   Asegúrate de que los modelos .tflite estén en app/src/main/assets/")
            isInitialized = false
            throw e
        }
    }
    
    /**
     * Detecta medias en una imagen usando el modelo ML.
     */
    suspend fun detectSocks(bitmap: Bitmap, frameWidth: Int, frameHeight: Int): DetectionResult = withContext(Dispatchers.Default) {
        if (!isInitialized) {
            throw IllegalStateException("MLSockDetector no ha sido inicializado. Llama a initialize() primero.")
        }
        
        val startTime = System.currentTimeMillis()
        
        println("🤖 [MLSockDetector] Detectando medias con ML...")
        // 1. Preprocesar imagen
        val inputBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val inputBuffer = preprocessImageRGB(inputBitmap, INPUT_SIZE, INPUT_SIZE)
        
        // 2. Ejecutar inferencia (SSD MobileNetv2 estilo)
        val outputLocations = Array(1) { Array(NUM_DETECTIONS) { FloatArray(4) } }  // [1, N, 4] (ymin, xmin, ymax, xmax)
        val outputClasses = Array(1) { FloatArray(NUM_DETECTIONS) }                 // [1, N]
        val outputScores = Array(1) { FloatArray(NUM_DETECTIONS) }                  // [1, N]
        val numDetections = FloatArray(1)                                           // [1]
        
        val outputs: MutableMap<Int, Any> = HashMap()
        outputs[0] = outputLocations
        outputs[1] = outputClasses
        outputs[2] = outputScores
        outputs[3] = numDetections
        
        detectorInterpreter?.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
        
        // 3. Procesar resultados
        val detectedSocks = mutableListOf<Sock>()
        val detCount = numDetections.getOrNull(0)?.toInt() ?: NUM_DETECTIONS
        for (i in 0 until detCount.coerceAtMost(NUM_DETECTIONS)) {
            val score = outputScores[0][i]
            if (score < MIN_CONFIDENCE) continue
            val box = outputLocations[0][i]
            // Desnormalizar usando frame
            val top = (box[0] * frameHeight)
            val left = (box[1] * frameWidth)
            val bottom = (box[2] * frameHeight)
            val right = (box[3] * frameWidth)
            
            val boundingBox = RectF(
                left.coerceIn(0f, frameWidth.toFloat()),
                top.coerceIn(0f, frameHeight.toFloat()),
                right.coerceIn(0f, frameWidth.toFloat()),
                bottom.coerceIn(0f, frameHeight.toFloat())
            )
            if (boundingBox.width() < 1f || boundingBox.height() < 1f) continue
            
            val sockBitmap = extractRegion(bitmap, boundingBox)
            val dominantColor = extractDominantColor(sockBitmap)
            
            detectedSocks.add(
                Sock(
                    id = UUID.randomUUID().toString(),
                    boundingBox = RectF(
                        boundingBox.left / frameWidth,
                        boundingBox.top / frameHeight,
                        boundingBox.right / frameWidth,
                        boundingBox.bottom / frameHeight
                    ),
                    dominantColor = dominantColor,
                    colorName = getColorName(dominantColor),
                    confidence = score,
                    bitmap = createThumbnail(sockBitmap)
                )
            )
        }
        
        println("🤖 [MLSockDetector] Medias detectadas: ${detectedSocks.size}")
        
        // Emparejar medias usando el modelo de similitud
        val pairs = pairSocksWithML(detectedSocks)
        
        val processingTime = System.currentTimeMillis() - startTime
        
        DetectionResult(
            socks = detectedSocks,
            pairs = pairs,
            processingTime = processingTime,
            frameCount = 1
        )
    }
    
    /**
     * Empareja medias usando el modelo de similitud ML.
     */
    private suspend fun pairSocksWithML(socks: List<Sock>): List<SockPair> {
        if (socks.size < 2) return emptyList()
        
        val pairs = mutableListOf<SockPair>()
        val usedSocks = mutableSetOf<String>()
        
        for (i in socks.indices) {
            if (usedSocks.contains(socks[i].id)) continue
            var bestMatch: Sock? = null
            var bestScore = 0f
            for (j in i + 1 until socks.size) {
                if (usedSocks.contains(socks[j].id)) continue
                val similarity = calculateSimilarityWithML(socks[i], socks[j])
                if (similarity > bestScore && similarity > 0.7f) {
                    bestScore = similarity
                    bestMatch = socks[j]
                }
            }
            if (bestMatch != null) {
                pairs.add(
                    SockPair(
                        sock1 = socks[i],
                        sock2 = bestMatch,
                        matchConfidence = bestScore,
                        pairId = UUID.randomUUID().toString()
                    )
                )
                usedSocks.add(socks[i].id)
                usedSocks.add(bestMatch.id)
            }
        }
        
        return pairs
    }
    
    /**
     * Calcula similitud entre dos medias usando el modelo ML.
     */
    private fun calculateSimilarityWithML(sock1: Sock, sock2: Sock): Float {
        val embedding1 = extractEmbedding(sock1.bitmap)
        val embedding2 = extractEmbedding(sock2.bitmap)
        if (embedding1.isEmpty() || embedding2.isEmpty()) return 0f
        return cosineSimilarity(embedding1, embedding2)
    }
    
    /**
     * Extrae embedding de una imagen usando el modelo.
     */
    private fun extractEmbedding(bitmap: Bitmap?): FloatArray {
        val interpreter = matcherInterpreter ?: return FloatArray(0)
        if (bitmap == null || bitmap.isRecycled) return FloatArray(0)
        val resized = Bitmap.createScaledBitmap(bitmap, MATCHER_INPUT_SIZE, MATCHER_INPUT_SIZE, true)
        val input = preprocessImageRGB(resized, MATCHER_INPUT_SIZE, MATCHER_INPUT_SIZE)
        val output = Array(1) { FloatArray(EMBEDDING_SIZE) }
        interpreter.run(input, output)
        return l2Normalize(output[0])
    }
    
    /**
     * Obtiene el nombre del color dominante (helper method).
     */
    private fun getColorName(color: Int): String {
        val hsv = FloatArray(3)
        Color.colorToHSV(color, hsv)
        return when {
            hsv[0] < 15 || hsv[0] > 345 -> "Rojo"
            hsv[0] < 45 -> "Naranja"
            hsv[0] < 75 -> "Amarillo"
            hsv[0] < 165 -> "Verde"
            hsv[0] < 240 -> "Azul"
            hsv[0] < 285 -> "Violeta"
            else -> "Rosa"
        }
    }
    
    /**
     * Libera recursos de los modelos.
     */
    fun close() {
        try {
            detectorInterpreter?.close()
        } catch (_: Exception) {}
        try {
            matcherInterpreter?.close()
        } catch (_: Exception) {}
        try {
            detectorDelegate?.close()
        } catch (_: Exception) {}
        try {
            matcherDelegate?.close()
        } catch (_: Exception) {}
        detectorInterpreter = null
        matcherInterpreter = null
        detectorDelegate = null
        matcherDelegate = null
        isInitialized = false
    }
    
    /**
     * Verifica si los modelos están disponibles.
     */
    fun areModelsAvailable(): Boolean {
        return try {
            context.assets.open(DETECTOR_MODEL_PATH).close()
            context.assets.open(MATCHER_MODEL_PATH).close()
            true
        } catch (e: Exception) {
            false
        }
    }

    // ---- Helpers de carga/preprocesamiento ----

    private fun loadModelFromAssets(assetPath: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(assetPath)
        val inputStream = assetFileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun preprocessImageRGB(bitmap: Bitmap, width: Int, height: Int): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(4 * width * height * 3)
        inputBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(width * height)
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height)
        var pixelIndex = 0
        // Normalización a [0,1]
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = intValues[pixelIndex++]
                val r = (pixel shr 16 and 0xFF) / 255f
                val g = (pixel shr 8 and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f
                inputBuffer.putFloat(r)
                inputBuffer.putFloat(g)
                inputBuffer.putFloat(b)
            }
        }
        inputBuffer.rewind()
        return inputBuffer
    }

    private fun extractRegion(source: Bitmap, box: RectF): Bitmap {
        val left = box.left.coerceAtLeast(0f).toInt()
        val top = box.top.coerceAtLeast(0f).toInt()
        val width = (box.width()).toInt().coerceAtLeast(1)
        val height = (box.height()).toInt().coerceAtLeast(1)
        val safeWidth = if (left + width > source.width) source.width - left else width
        val safeHeight = if (top + height > source.height) source.height - top else height
        return Bitmap.createBitmap(source, left, top, safeWidth.coerceAtLeast(1), safeHeight.coerceAtLeast(1))
    }

    private fun extractDominantColor(bitmap: Bitmap): Int {
        return try {
            val palette = Palette.from(bitmap).generate()
            palette.getDominantColor(Color.GRAY)
        } catch (e: Exception) {
            Color.GRAY
        }
    }

    private fun createThumbnail(bitmap: Bitmap): Bitmap? {
        return try {
            Bitmap.createScaledBitmap(bitmap, 64, 64, true)
        } catch (_: Exception) {
            null
        }
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b.getOrElse(i) { 0f }
            normA += a[i] * a[i]
            val bv = b.getOrElse(i) { 0f }
            normB += bv * bv
        }
        val denom = (kotlin.math.sqrt(normA.toDouble()) * kotlin.math.sqrt(normB.toDouble())).toFloat()
        return if (denom > 0f) (dot / denom).coerceIn(0f, 1f) else 0f
    }

    private fun l2Normalize(vec: FloatArray): FloatArray {
        var sum = 0f
        for (v in vec) sum += v * v
        val norm = kotlin.math.sqrt(sum.toDouble()).toFloat()
        if (norm <= 0f) return vec
        for (i in vec.indices) vec[i] /= norm
        return vec
    }
}

/**
 * EJEMPLO DE USO EN VIEWMODEL:
 * 
 * class SockDetectionViewModel(application: Application) : AndroidViewModel(application) {
 *     private val mlDetector = MLSockDetector(application)
 *     private val fallbackDetector = SockDetector() // Detector heurístico actual
 *     
 *     init {
 *         viewModelScope.launch {
 *             try {
 *                 mlDetector.initialize()
 *             } catch (e: Exception) {
 *                 println("ML models not available, using fallback detector")
 *             }
 *         }
 *     }
 *     
 *     suspend fun detectSocks(bitmap: Bitmap, width: Int, height: Int): DetectionResult {
 *         return if (mlDetector.areModelsAvailable()) {
 *             mlDetector.detectSocks(bitmap, width, height)
 *         } else {
 *             fallbackDetector.detectSocks(bitmap, width, height)
 *         }
 *     }
 * }
 */



