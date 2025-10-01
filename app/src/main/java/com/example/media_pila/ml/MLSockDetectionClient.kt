package com.example.media_pila.ml

import android.content.Context
import android.graphics.Bitmap
import com.example.media_pila.data.DetectionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Cliente helper para ejecutar detección desde un Bitmap sin depender del ViewModel/UI.
 */
class MLSockDetectionClient(private val context: Context) {
    private val detector by lazy { MLSockDetector(context) }

    suspend fun detectFromBitmap(bitmap: Bitmap): DetectionResult = withContext(Dispatchers.Default) {
        if (!detector.areModelsAvailable()) {
            throw IllegalStateException("Modelos ML no disponibles en assets")
        }
        detector.initialize()
        try {
            detector.detectSocks(bitmap, bitmap.width, bitmap.height)
        } finally {
            detector.close()
        }
    }
}


