package com.example.media_pila.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.media_pila.data.DetectionResult
import com.example.media_pila.data.Sock
import com.example.media_pila.data.SockPair
import kotlin.math.roundToInt

@Composable
fun DetectionOverlay(
    detectionResult: DetectionResult?,
    modifier: Modifier = Modifier
) {
    if (detectionResult == null) return
    
    // Generar numeración estable por color/par
    val sockNumbering = remember(detectionResult) {
        generateStableNumbering(detectionResult)
    }
    
    Canvas(
        modifier = modifier.fillMaxSize()
    ) {
        val canvasWidth = size.width
        val canvasHeight = size.height
        
        // Obtener medias que ya están emparejadas para evitar duplicados
        val pairedSockIds = detectionResult.pairs.flatMap { pair ->
            listOf(pair.sock1.id, pair.sock2.id)
        }.toSet()
        
        // Dibujar medias individuales (no emparejadas) con su color dominante
        detectionResult.socks
            .filter { sock -> !pairedSockIds.contains(sock.id) }
            .forEach { sock ->
                val sockColor = Color(sock.dominantColor).copy(alpha = 0.8f)
                val sockNumber = sockNumbering[sock.id] ?: 1
                val label = "Media ${sock.colorName} $sockNumber (${(sock.confidence * 100).roundToInt()}%)"
                
                drawSockBoundingBox(
                    sock = sock,
                    canvasWidth = canvasWidth,
                    canvasHeight = canvasHeight,
                    color = sockColor,
                    label = label
                )
            }
        
        // Dibujar pares con colores derivados del color dominante
        detectionResult.pairs.forEachIndexed { pairIndex, pair ->
            // Usar el color dominante de la primera media del par como base
            val baseColor = Color(pair.sock1.dominantColor)
            val pairColor = baseColor.copy(alpha = 0.8f)
            
            val matchPercentage = (pair.matchConfidence * 100).roundToInt()
            val pairNumber = pairIndex + 1
            
            // Obtener números de las medias del par
            val sock1Number = sockNumbering[pair.sock1.id] ?: 1
            val sock2Number = sockNumbering[pair.sock2.id] ?: 2
            
            // Dibujar bounding box para sock1 del par
            val label1 = "Media ${pair.sock1.colorName} $sock1Number similitud ${matchPercentage}%"
            drawSockBoundingBox(
                sock = pair.sock1,
                canvasWidth = canvasWidth,
                canvasHeight = canvasHeight,
                color = pairColor,
                label = label1
            )
            
            // Dibujar bounding box para sock2 del par
            val label2 = "Media ${pair.sock2.colorName} $sock2Number similitud ${matchPercentage}%"
            drawSockBoundingBox(
                sock = pair.sock2,
                canvasWidth = canvasWidth,
                canvasHeight = canvasHeight,
                color = pairColor,
                label = label2
            )
            
            // Dibujar línea de conexión entre pares
            drawPairConnection(pair, canvasWidth, canvasHeight, pairColor)
        }
    }
}

/**
 * Genera numeración estable por color/par para las medias detectadas.
 * Las medias del mismo color tendrán números consecutivos.
 */
private fun generateStableNumbering(detectionResult: DetectionResult): Map<String, Int> {
    val numbering = mutableMapOf<String, Int>()
    
    // Agrupar medias por color
    val socksByColor = detectionResult.socks.groupBy { it.colorName }
    
    var currentNumber = 1
    
    // Asignar números por color, manteniendo orden estable
    socksByColor.toSortedMap().forEach { (colorName, socks) ->
        socks.sortedBy { it.id }.forEach { sock ->
            numbering[sock.id] = currentNumber++
        }
    }
    
    return numbering
}

private fun DrawScope.drawSockBoundingBox(
    sock: Sock,
    canvasWidth: Float,
    canvasHeight: Float,
    color: Color,
    label: String
) {
    val rect = sock.boundingBox
    
    // Mapear coordenadas normalizadas al tamaño real del canvas
    val left = rect.left * canvasWidth
    val top = rect.top * canvasHeight
    val right = rect.right * canvasWidth
    val bottom = rect.bottom * canvasHeight
    
    // Convertir 4dp a píxeles (aproximadamente 16f para densidad estándar)
    val strokeWidth = 16f
    
    // Dibujar rectángulo del bounding box
    drawRect(
        color = color,
        topLeft = Offset(left, top),
        size = Size(right - left, bottom - top),
        style = Stroke(width = strokeWidth)
    )
    
    // Dibujar texto arriba del bounding box
    drawIntoCanvas { canvas ->
        val paint = android.graphics.Paint().apply {
            this.color = android.graphics.Color.WHITE
            textSize = 36f * 3.5f // Convertir 36sp a píxeles (aproximadamente)
            setShadowLayer(3f, 1f, 1f, android.graphics.Color.BLACK)
            isAntiAlias = true
        }
        
        // Posicionar texto arriba del bounding box
        val textX = left + (right - left) / 2 // Centrar horizontalmente
        val textY = top - 10f // 10 píxeles arriba del bounding box
        
        // Centrar el texto horizontalmente
        val textBounds = android.graphics.Rect()
        paint.getTextBounds(label, 0, label.length, textBounds)
        val centeredX = textX - textBounds.width() / 2
        
        canvas.nativeCanvas.drawText(
            label,
            centeredX,
            textY,
            paint
        )
    }
}

/**
 * Dibuja una línea de conexión entre dos medias de un par
 */
private fun DrawScope.drawPairConnection(
    pair: SockPair,
    canvasWidth: Float,
    canvasHeight: Float,
    color: Color
) {
    val sock1 = pair.sock1
    val sock2 = pair.sock2
    
    val center1 = Offset(
        (sock1.boundingBox.left + sock1.boundingBox.right) / 2 * canvasWidth,
        (sock1.boundingBox.top + sock1.boundingBox.bottom) / 2 * canvasHeight
    )
    
    val center2 = Offset(
        (sock2.boundingBox.left + sock2.boundingBox.right) / 2 * canvasWidth,
        (sock2.boundingBox.top + sock2.boundingBox.bottom) / 2 * canvasHeight
    )
    
    // Dibujar línea de conexión
    drawLine(
        color = color,
        start = center1,
        end = center2,
        strokeWidth = 8f
    )
    
    // Dibujar círculos en los centros
    drawCircle(
        color = color,
        radius = 12f,
        center = center1
    )
    
    drawCircle(
        color = color,
        radius = 12f,
        center = center2
    )
} 