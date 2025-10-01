package com.example.media_pila.data

import android.graphics.Color
import android.graphics.RectF

import android.graphics.Bitmap

/**
 * Representa una media individual detectada
 */
data class Sock(
    val id: String,
    val boundingBox: RectF,
    val dominantColor: Int,
    val colorName: String,
    val confidence: Float,
    val texture: String = "plain", // plain, striped, dotted, etc.
    val bitmap: Bitmap? = null // miniatura de la media para comparación por histograma
)

/**
 * Representa un par de medias emparejadas
 */
data class SockPair(
    val sock1: Sock,
    val sock2: Sock,
    val matchConfidence: Float,
    val pairId: String
)

/**
 * Representa un color en formato HSV
 */
data class HSVColor(
    val hue: Float,
    val saturation: Float,
    val value: Float
) {
    fun toInt(): Int {
        return Color.HSVToColor(floatArrayOf(hue, saturation, value))
    }
    
    companion object {
        fun fromInt(color: Int): HSVColor {
            val hsv = FloatArray(3)
            Color.colorToHSV(color, hsv)
            return HSVColor(hsv[0], hsv[1], hsv[2])
        }
    }
}

/**
 * Resultado del procesamiento de imagen
 */
data class DetectionResult(
    val socks: List<Sock>,
    val pairs: List<SockPair>,
    val processingTime: Long,
    val frameCount: Int
)

/**
 * Estados de la aplicación
 */
sealed class AppState {
    object Loading : AppState()
    object CameraPermissionRequired : AppState()
    object Detecting : AppState()
    data class Detected(val result: DetectionResult) : AppState()
    data class StaticImageDetected(val result: DetectionResult, val imageBitmap: Bitmap) : AppState()
    data class Error(val message: String) : AppState()
} 