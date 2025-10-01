package com.example.media_pila.image

import android.graphics.*
import androidx.palette.graphics.Palette
import com.example.media_pila.data.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.pow
import kotlin.math.sqrt
import java.util.*

class SockDetector {
    
    companion object {
        private const val MIN_SOCK_SIZE = 0.02f // 2% del frame (más pequeño para detectar más medias)
        private const val MAX_SOCK_SIZE = 0.6f  // 60% del frame (más restrictivo)
        private const val COLOR_SIMILARITY_THRESHOLD = 0.15f // umbral de similitud de color (reducido de 0.3f a 0.15f)
        private const val MIN_CONFIDENCE = 0.05f // umbral reducido temporalmente para diagnóstico (era 0.25f)
        private const val GRID_SIZE = 16 // cuadrícula más fina para mejor detección
        private const val MIN_DISTANCE_BETWEEN_SOCKS = 50f // distancia mínima entre medias
        private const val MAX_ASPECT_RATIO = 3.0f // relación de aspecto máxima para medias
        private const val MIN_ASPECT_RATIO = 0.5f // relación de aspecto mínima para medias
    }
    
    /**
     * Procesa un frame de la cámara para detectar medias
     */
    suspend fun detectSocks(bitmap: Bitmap, frameWidth: Int, frameHeight: Int): DetectionResult = withContext(Dispatchers.Default) {
        val startTime = System.currentTimeMillis()
        
        println("🔍 [SockDetector] Iniciando detección de medias...")
        println("🔍 [SockDetector] Bitmap size: ${bitmap.width}x${bitmap.height}")
        println("🔍 [SockDetector] Frame: ${frameWidth}x${frameHeight}")
        println("🔍 [SockDetector] MIN_CONFIDENCE: $MIN_CONFIDENCE")
        println("🔍 [SockDetector] Usando comparación por histogramas RGB (64 bins por canal)")
        
        // Detectar regiones candidatas
        val candidateRegions = detectCandidateRegions(bitmap)
        println("🔍 [SockDetector] Regiones candidatas detectadas: ${candidateRegions.size}")
        
        // Analizar cada región para detectar medias
        val detectedSocks = candidateRegions.mapNotNull { region ->
            analyzeRegion(bitmap, region, frameWidth, frameHeight)
        }.filter { it.confidence >= MIN_CONFIDENCE }
        
        println("🔍 [SockDetector] Medias detectadas: ${detectedSocks.size}")
        detectedSocks.forEachIndexed { index, sock ->
            println("  ${index + 1}. Media ${sock.id.take(8)} - Color: ${sock.colorName} - Confianza: ${String.format("%.3f", sock.confidence)}")
        }
        
        // Emparejar medias
        val pairs = pairSocks(detectedSocks)
        
        val processingTime = System.currentTimeMillis() - startTime
        
        println("🔍 [SockDetector] Procesamiento completado en ${processingTime}ms")
        println("🔍 [SockDetector] Resumen: ${detectedSocks.size} medias, ${pairs.size} pares")
        
        // Limpiar bitmaps después del emparejamiento para liberar memoria
        cleanupSockBitmaps(detectedSocks)
        
        DetectionResult(
            socks = detectedSocks,
            pairs = pairs,
            processingTime = processingTime,
            frameCount = 1
        )
    }
    
    /**
     * Convierte Image de CameraX a Bitmap (método alternativo si se necesita)
     */
    private fun imageToBitmap(image: android.media.Image): Bitmap {
        val planes = image.planes
        val buffer = planes[0].buffer
        val pixelStride = planes[0].pixelStride
        val rowStride = planes[0].rowStride
        val rowPadding = rowStride - pixelStride * image.width
        
        val bitmap = Bitmap.createBitmap(
            image.width + rowPadding / pixelStride,
            image.height,
            Bitmap.Config.ARGB_8888
        )
        bitmap.copyPixelsFromBuffer(buffer)
        
        return bitmap
    }
    
    /**
     * Detecta regiones candidatas usando análisis de contornos y color
     */
    private fun detectCandidateRegions(bitmap: Bitmap): List<RectF> {
        val regions = mutableListOf<RectF>()
        val width = bitmap.width
        val height = bitmap.height
        
        println("🔍 [SockDetector] detectCandidateRegions - Bitmap: ${width}x${height}")
        println("🔍 [SockDetector] detectCandidateRegions - GRID_SIZE: $GRID_SIZE")
        
        // Dividir la imagen en una cuadrícula más fina para mejor detección
        val cellWidth = width / GRID_SIZE
        val cellHeight = height / GRID_SIZE
        
        println("🔍 [SockDetector] detectCandidateRegions - Cell size: ${cellWidth}x${cellHeight}")
        
        // Usar múltiples escalas para detectar medias de diferentes tamaños
        val scales = listOf(1.0f, 1.5f, 2.0f)
        
        for (scale in scales) {
            val scaledCellWidth = (cellWidth * scale).toInt()
            val scaledCellHeight = (cellHeight * scale).toInt()
            
            println("🔍 [SockDetector] detectCandidateRegions - Scale: $scale, Cell: ${scaledCellWidth}x${scaledCellHeight}")
            
            for (row in 0 until (height / scaledCellHeight)) {
                for (col in 0 until (width / scaledCellWidth)) {
                    val left = col * scaledCellWidth
                    val top = row * scaledCellHeight
                    val right = left + scaledCellWidth
                    val bottom = top + scaledCellHeight
                    
                    // Asegurar que la región esté dentro de los límites
                    if (right <= width && bottom <= height) {
                        val region = RectF(left.toFloat(), top.toFloat(), right.toFloat(), bottom.toFloat())
                        
                        // Analizar si esta región tiene características de media
                        if (isSockCandidate(bitmap, region)) {
                            regions.add(region)
                        }
                    }
                }
            }
        }
        
        println("🔍 [SockDetector] detectCandidateRegions - Regiones antes de filtros: ${regions.size}")
        
        // Filtrar regiones por tamaño y relación de aspecto
        val filteredRegions = regions.filter { region ->
            val area = region.width() * region.height()
            val normalizedArea = area / (width * height)
            val aspectRatio = region.width() / region.height()
            
            val passesSize = normalizedArea >= MIN_SOCK_SIZE && normalizedArea <= MAX_SOCK_SIZE
            val passesAspect = aspectRatio >= MIN_ASPECT_RATIO && aspectRatio <= MAX_ASPECT_RATIO
            
            if (!passesSize || !passesAspect) {
                println("🔍 [SockDetector] detectCandidateRegions - Región filtrada: area=${String.format("%.6f", normalizedArea)}, aspect=${String.format("%.3f", aspectRatio)}")
            }
            
            passesSize && passesAspect
        }
        
        println("🔍 [SockDetector] detectCandidateRegions - Regiones después de filtros: ${filteredRegions.size}")
        
        // Agrupar regiones cercanas pero mantener separadas las que están suficientemente lejos
        val mergedRegions = mergeNearbyRegions(filteredRegions)
        println("🔍 [SockDetector] detectCandidateRegions - Regiones finales: ${mergedRegions.size}")
        
        return mergedRegions
    }
    
    /**
     * Determina si una región es candidata para ser una media
     */
    private fun isSockCandidate(bitmap: Bitmap, region: RectF): Boolean {
        try {
            val croppedBitmap = Bitmap.createBitmap(
                bitmap,
                region.left.toInt(),
                region.top.toInt(),
                region.width().toInt(),
                region.height().toInt()
            )
            
            // Analizar paleta de colores
            val palette = Palette.from(croppedBitmap).generate()
            val dominantColor = palette.getDominantColor(Color.GRAY)
            
            // Verificar si el color dominante no es muy gris (fondo)
            val hsv = FloatArray(3)
            Color.colorToHSV(dominantColor, hsv)
            
            // Si la saturación es muy baja, probablemente es fondo
            if (hsv[1] < 0.08f) {
                println("🔍 [SockDetector] isSockCandidate - ❌ Saturación muy baja: ${String.format("%.3f", hsv[1])}")
                return false
            }
            
            // Verificar si hay suficiente variación de color
            val vibrantColor = palette.getVibrantColor(Color.GRAY)
            val mutedColor = palette.getMutedColor(Color.GRAY)
            
            val colorVariation = Math.abs(Color.red(vibrantColor) - Color.red(mutedColor)) +
                    Math.abs(Color.green(vibrantColor) - Color.green(mutedColor)) +
                    Math.abs(Color.blue(vibrantColor) - Color.blue(mutedColor))
            
            // Umbral más bajo para detectar más medias
            if (colorVariation < 30) {
                println("🔍 [SockDetector] isSockCandidate - ❌ Variación de color insuficiente: $colorVariation")
                return false
            }
            
            // Verificar que el valor (brillo) no sea muy bajo (no es sombra)
            if (hsv[2] < 0.15f) {
                println("🔍 [SockDetector] isSockCandidate - ❌ Brillo muy bajo: ${String.format("%.3f", hsv[2])}")
                return false
            }
            
            // Verificar que el valor no sea muy alto (no es blanco puro)
            if (hsv[2] > 0.95f && hsv[1] < 0.1f) {
                println("🔍 [SockDetector] isSockCandidate - ❌ Blanco puro detectado")
                return false
            }
            
            // Verificar relación de aspecto (las medias suelen ser más altas que anchas)
            val aspectRatio = region.width() / region.height()
            if (aspectRatio < MIN_ASPECT_RATIO || aspectRatio > MAX_ASPECT_RATIO) {
                println("🔍 [SockDetector] isSockCandidate - ❌ Aspect ratio fuera de rango: ${String.format("%.3f", aspectRatio)}")
                return false
            }
            
            println("🔍 [SockDetector] isSockCandidate - ✅ Región candidata válida")
            return true
            
        } catch (e: Exception) {
            println("🔍 [SockDetector] isSockCandidate - ❌ Error: ${e.message}")
            return false
        }
    }
    
    /**
     * Agrupa regiones cercanas pero mantiene separadas las medias distintas
     */
    private fun mergeNearbyRegions(regions: List<RectF>): List<RectF> {
        if (regions.isEmpty()) return emptyList()
        
        val merged = mutableListOf<RectF>()
        val used = mutableSetOf<Int>()
        
        for (i in regions.indices) {
            if (used.contains(i)) continue
            
            var currentRegion = regions[i]
            used.add(i)
            
            for (j in i + 1 until regions.size) {
                if (used.contains(j)) continue
                
                // Solo fusionar si las regiones están muy cerca (mismo objeto)
                if (regionsOverlap(currentRegion, regions[j]) || 
                    getDistanceBetweenRegions(currentRegion, regions[j]) < MIN_DISTANCE_BETWEEN_SOCKS) {
                    currentRegion = mergeRegions(currentRegion, regions[j])
                    used.add(j)
                }
            }
            
            merged.add(currentRegion)
        }
        
        return merged
    }
    
    /**
     * Calcula la distancia entre dos regiones
     */
    private fun getDistanceBetweenRegions(rect1: RectF, rect2: RectF): Float {
        val center1 = PointF((rect1.left + rect1.right) / 2, (rect1.top + rect1.bottom) / 2)
        val center2 = PointF((rect2.left + rect2.right) / 2, (rect2.top + rect2.bottom) / 2)
        
        return sqrt((center1.x - center2.x).pow(2) + (center1.y - center2.y).pow(2)).toFloat()
    }
    
    private fun regionsOverlap(rect1: RectF, rect2: RectF): Boolean {
        return !(rect1.right < rect2.left || rect2.right < rect1.left ||
                rect1.bottom < rect2.top || rect2.bottom < rect1.top)
    }
    
    private fun mergeRegions(rect1: RectF, rect2: RectF): RectF {
        return RectF(
            minOf(rect1.left, rect2.left),
            minOf(rect1.top, rect2.top),
            maxOf(rect1.right, rect2.right),
            maxOf(rect1.bottom, rect2.bottom)
        )
    }
    
    /**
     * Analiza una región específica para determinar si contiene una media
     */
    private fun analyzeRegion(bitmap: Bitmap, region: RectF, frameWidth: Int, frameHeight: Int): Sock? {
        println("🔍 [SockDetector] analyzeRegion - Analizando región: (${region.left.toInt()}, ${region.top.toInt()}, ${region.width().toInt()}, ${region.height().toInt()})")
        
        val croppedBitmap = Bitmap.createBitmap(
            bitmap,
            region.left.toInt(),
            region.top.toInt(),
            region.width().toInt(),
            region.height().toInt()
        )
        
        val palette = Palette.from(croppedBitmap).generate()
        val dominantColor = palette.getDominantColor(Color.GRAY)
        
        // Calcular confianza basada en características de la región
        val confidence = calculateConfidence(croppedBitmap, palette)
        
        println("🔍 [SockDetector] analyzeRegion - Confidence: ${String.format("%.3f", confidence)} (umbral: $MIN_CONFIDENCE)")
        
        if (confidence < MIN_CONFIDENCE) {
            println("🔍 [SockDetector] analyzeRegion - ❌ Región rechazada por baja confianza")
            return null
        }
        
        // Crear miniatura para comparación por histograma (64x64 píxeles)
        val thumbnailBitmap = try {
            Bitmap.createScaledBitmap(croppedBitmap, 64, 64, true)
        } catch (e: Exception) {
            println("🔍 [SockDetector] Error creando miniatura: ${e.message}")
            null
        }
        
        // Normalizar coordenadas al frame completo
        val normalizedRegion = RectF(
            region.left / frameWidth,
            region.top / frameHeight,
            region.right / frameWidth,
            region.bottom / frameHeight
        )
        
        val sock = Sock(
            id = UUID.randomUUID().toString(),
            boundingBox = normalizedRegion,
            dominantColor = dominantColor,
            colorName = getColorName(dominantColor),
            confidence = confidence,
            bitmap = thumbnailBitmap
        )
        
        println("🔍 [SockDetector] analyzeRegion - ✅ Media detectada: ${sock.colorName} (${String.format("%.3f", confidence)})")
        
        return sock
    }
    
    /**
     * Calcula la confianza de que una región contiene una media
     */
    private fun calculateConfidence(bitmap: Bitmap, palette: Palette): Float {
        var confidence = 0f
        
        // Factor 1: Variación de color
        val vibrantColor = palette.getVibrantColor(Color.GRAY)
        val mutedColor = palette.getMutedColor(Color.GRAY)
        val colorVariation = Math.abs(Color.red(vibrantColor) - Color.red(mutedColor)) +
                Math.abs(Color.green(vibrantColor) - Color.green(mutedColor)) +
                Math.abs(Color.blue(vibrantColor) - Color.blue(mutedColor))
        
        confidence += (colorVariation / 765f) * 0.4f // Máximo 40% del score
        
        // Factor 2: Saturación del color dominante
        val hsv = FloatArray(3)
        Color.colorToHSV(palette.getDominantColor(Color.GRAY), hsv)
        confidence += hsv[1] * 0.3f // Máximo 30% del score
        
        // Factor 3: Tamaño de la región (preferir tamaños medios)
        val area = bitmap.width * bitmap.height
        val normalizedArea = area / (1920f * 1080f) // Normalizar a resolución típica
        val sizeScore = when {
            normalizedArea < 0.01f -> 0f
            normalizedArea < 0.1f -> 1f
            normalizedArea < 0.3f -> 0.8f
            else -> 0.3f
        }
        confidence += sizeScore * 0.3f // Máximo 30% del score
        
        return confidence.coerceIn(0f, 1f)
    }
    
    /**
     * Obtiene el nombre del color dominante
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
     * Empareja medias basándose en múltiples criterios
     */
    private fun pairSocks(socks: List<Sock>): List<SockPair> {
        println("🔍 [SockDetector] pairSocks - Iniciando emparejamiento de ${socks.size} medias")
        
        if (socks.size < 2) {
            println("🔍 [SockDetector] pairSocks - ❌ No hay suficientes medias para emparejar (${socks.size} medias)")
            return emptyList()
        }
        
        println("🔍 [SockDetector] pairSocks - Umbral de similitud: $COLOR_SIMILARITY_THRESHOLD")
        
        val pairs = mutableListOf<SockPair>()
        val usedSocks = mutableSetOf<String>()
        
        // Ordenar medias por confianza para priorizar las más seguras
        val sortedSocks = socks.sortedByDescending { it.confidence }
        
        println("🔍 [SockDetector] pairSocks - Medias ordenadas por confianza:")
        sortedSocks.forEachIndexed { index, sock ->
            println("  ${index + 1}. Media ${sock.id.take(8)} - Color: ${sock.colorName} - Confianza: ${String.format("%.3f", sock.confidence)}")
        }
        
        for (i in sortedSocks.indices) {
            if (usedSocks.contains(sortedSocks[i].id)) {
                println("🔍 [SockDetector] Media ${sortedSocks[i].id.take(8)} ya emparejada, saltando...")
                continue
            }
            
            println("🔍 [SockDetector] Buscando pareja para media ${sortedSocks[i].id.take(8)} (${sortedSocks[i].colorName})")
            
            var bestMatch: Sock? = null
            var bestScore = 0f
            var comparisonCount = 0
            
            for (j in i + 1 until sortedSocks.size) {
                if (usedSocks.contains(sortedSocks[j].id)) {
                    println("  ⏭️  Media ${sortedSocks[j].id.take(8)} ya emparejada, saltando...")
                    continue
                }
                
                comparisonCount++
                val matchScore = calculateMatchScore(sortedSocks[i], sortedSocks[j])
                
                                 println("  🔄 Comparando con media ${sortedSocks[j].id.take(8)} (${sortedSocks[j].colorName})")
                 println("     Match Score: ${String.format("%.3f", matchScore)} (umbral: ${String.format("%.3f", COLOR_SIMILARITY_THRESHOLD)})")
                 
                 // Log adicional para histograma
                 val histogramScore = calculateHistogramSimilarity(sortedSocks[i].bitmap, sortedSocks[j].bitmap)
                 println("     📊 Histograma Score: ${String.format("%.3f", histogramScore)}")
                
                if (matchScore > bestScore && matchScore > COLOR_SIMILARITY_THRESHOLD) {
                    bestScore = matchScore
                    bestMatch = sortedSocks[j]
                    println("     ✅ Nuevo mejor match encontrado!")
                } else if (matchScore > bestScore) {
                    println("     ❌ Score alto (${String.format("%.3f", matchScore)}) pero por debajo del umbral")
                } else {
                    println("     ❌ Score bajo (${String.format("%.3f", matchScore)})")
                }
            }
            
            if (bestMatch != null) {
                println("🔍 [SockDetector] ✅ Par formado: Media ${sortedSocks[i].id.take(8)} + Media ${bestMatch.id.take(8)}")
                println("   Score final: ${String.format("%.3f", bestScore)}")
                
                pairs.add(
                    SockPair(
                        sock1 = sortedSocks[i],
                        sock2 = bestMatch,
                        matchConfidence = bestScore,
                        pairId = UUID.randomUUID().toString()
                    )
                )
                usedSocks.add(sortedSocks[i].id)
                usedSocks.add(bestMatch.id)
            } else {
                println("🔍 [SockDetector] ❌ No se encontró pareja para media ${sortedSocks[i].id.take(8)}")
                if (comparisonCount == 0) {
                    println("   Razón: No hay medias disponibles para comparar")
                } else {
                    println("   Razón: Ningún match superó el umbral de ${String.format("%.3f", COLOR_SIMILARITY_THRESHOLD)}")
                }
            }
        }
        
        println("🔍 [SockDetector] pairSocks - Emparejamiento completado: ${pairs.size} pares formados de ${socks.size} medias")
        if (pairs.isNotEmpty()) {
            pairs.forEachIndexed { index, pair ->
                println("  Par ${index + 1}: Media ${pair.sock1.id.take(8)} + Media ${pair.sock2.id.take(8)} - Match: ${String.format("%.3f", pair.matchConfidence)}")
            }
        }
        return pairs
    }
    
    /**
     * Calcula un score de emparejamiento basado en múltiples criterios
     */
    private fun calculateMatchScore(sock1: Sock, sock2: Sock): Float {
        var totalScore = 0f
        var weightSum = 0f
        
        // 1. Similitud de histograma (50% del peso) - NUEVA MÉTRICA PRINCIPAL
        val histogramScore = calculateHistogramSimilarity(sock1.bitmap, sock2.bitmap)
        totalScore += histogramScore * 0.5f
        weightSum += 0.5f
        
        // 2. Similitud de color (20% del peso) - REDUCIDO
        val colorScore = calculateColorSimilarity(sock1, sock2)
        totalScore += colorScore * 0.2f
        weightSum += 0.2f
        
        // 3. Similitud de tamaño (15% del peso) - REDUCIDO
        val sizeScore = calculateSizeSimilarity(sock1, sock2)
        totalScore += sizeScore * 0.15f
        weightSum += 0.15f
        
        // 4. Similitud de forma (10% del peso) - REDUCIDO
        val shapeScore = calculateShapeSimilarity(sock1, sock2)
        totalScore += shapeScore * 0.1f
        weightSum += 0.1f
        
        // 5. Posición relativa (5% del peso) - REDUCIDO
        val positionScore = calculatePositionScore(sock1, sock2)
        totalScore += positionScore * 0.05f
        weightSum += 0.05f
        
        val finalScore = if (weightSum > 0) totalScore / weightSum else 0f
        
        // Log detallado de los componentes del score
        println("     📊 Componentes del Match Score:")
        println("        📊 Histograma: ${String.format("%.3f", histogramScore)} (peso: 50%)")
        println("        🎨 Color: ${String.format("%.3f", colorScore)} (peso: 20%)")
        println("        📏 Tamaño: ${String.format("%.3f", sizeScore)} (peso: 15%)")
        println("        🔷 Forma: ${String.format("%.3f", shapeScore)} (peso: 10%)")
        println("        📍 Posición: ${String.format("%.3f", positionScore)} (peso: 5%)")
        println("        🎯 Score final: ${String.format("%.3f", finalScore)}")
        
        return finalScore
    }
    
    /**
     * Calcula similitud de color entre dos medias
     */
    private fun calculateColorSimilarity(sock1: Sock, sock2: Sock): Float {
        val hsv1 = HSVColor.fromInt(sock1.dominantColor)
        val hsv2 = HSVColor.fromInt(sock2.dominantColor)
        
        // Calcular diferencia en HSV
        val hueDiff = Math.abs(hsv1.hue - hsv2.hue)
        val satDiff = Math.abs(hsv1.saturation - hsv2.saturation)
        val valDiff = Math.abs(hsv1.value - hsv2.value)
        
        // Normalizar diferencias
        val normalizedHueDiff = minOf(hueDiff, 360f - hueDiff) / 180f
        val normalizedSatDiff = satDiff
        val normalizedValDiff = valDiff
        
        // Calcular score de similitud (1 = idéntico, 0 = muy diferente)
        val colorScore = 1f - (normalizedHueDiff * 0.6f + normalizedSatDiff * 0.2f + normalizedValDiff * 0.2f)
        
        // Log detallado de similitud de color
        println("        🎨 Detalles de color:")
        println("          Color1: ${sock1.colorName} (H:${String.format("%.1f", hsv1.hue)}°, S:${String.format("%.3f", hsv1.saturation)}, V:${String.format("%.3f", hsv1.value)})")
        println("          Color2: ${sock2.colorName} (H:${String.format("%.1f", hsv2.hue)}°, S:${String.format("%.3f", hsv2.saturation)}, V:${String.format("%.3f", hsv2.value)})")
        println("          Diferencias - H:${String.format("%.1f", hueDiff)}°, S:${String.format("%.3f", satDiff)}, V:${String.format("%.3f", valDiff)}")
        println("          Score de color: ${String.format("%.3f", colorScore)}")
        
        return colorScore
    }
    
    /**
     * Calcula similitud de tamaño entre dos medias
     */
    private fun calculateSizeSimilarity(sock1: Sock, sock2: Sock): Float {
        val area1 = sock1.boundingBox.width() * sock1.boundingBox.height()
        val area2 = sock2.boundingBox.width() * sock2.boundingBox.height()
        
        val maxArea = maxOf(area1, area2)
        val minArea = minOf(area1, area2)
        
        val sizeScore = if (maxArea > 0) minArea / maxArea else 0f
        
        // Log detallado de similitud de tamaño
        println("        📏 Detalles de tamaño:")
        println("          Área1: ${String.format("%.6f", area1)}")
        println("          Área2: ${String.format("%.6f", area2)}")
        println("          Ratio: ${String.format("%.3f", sizeScore)}")
        
        return sizeScore
    }
    
    /**
     * Calcula similitud de forma entre dos medias
     */
    private fun calculateShapeSimilarity(sock1: Sock, sock2: Sock): Float {
        val aspectRatio1 = sock1.boundingBox.width() / sock1.boundingBox.height()
        val aspectRatio2 = sock2.boundingBox.width() / sock2.boundingBox.height()
        
        val maxRatio = maxOf(aspectRatio1, aspectRatio2)
        val minRatio = minOf(aspectRatio1, aspectRatio2)
        
        val shapeScore = if (maxRatio > 0) minRatio / maxRatio else 0f
        
        // Log detallado de similitud de forma
        println("        🔷 Detalles de forma:")
        println("          Aspect Ratio1: ${String.format("%.3f", aspectRatio1)}")
        println("          Aspect Ratio2: ${String.format("%.3f", aspectRatio2)}")
        println("          Ratio de similitud: ${String.format("%.3f", shapeScore)}")
        
        return shapeScore
    }
    
    /**
     * Calcula score de posición relativa (favorece medias cercanas)
     */
    private fun calculatePositionScore(sock1: Sock, sock2: Sock): Float {
        val center1 = PointF(
            (sock1.boundingBox.left + sock1.boundingBox.right) / 2,
            (sock1.boundingBox.top + sock1.boundingBox.bottom) / 2
        )
        val center2 = PointF(
            (sock2.boundingBox.left + sock2.boundingBox.right) / 2,
            (sock2.boundingBox.top + sock2.boundingBox.bottom) / 2
        )
        
        val distance = sqrt((center1.x - center2.x).pow(2) + (center1.y - center2.y).pow(2)).toFloat()
        
        // Favorecer medias cercanas pero no demasiado (evitar solapamientos)
        val positionScore = when {
            distance < 0.1f -> 0.5f // Muy cerca, posible solapamiento
            distance < 0.3f -> 1.0f // Distancia ideal
            distance < 0.5f -> 0.7f // Aceptable
            else -> 0.3f // Muy lejos
        }
        
        // Log detallado de posición
        println("        📍 Detalles de posición:")
        println("          Centro1: (${String.format("%.3f", center1.x)}, ${String.format("%.3f", center1.y)})")
        println("          Centro2: (${String.format("%.3f", center2.x)}, ${String.format("%.3f", center2.y)})")
        println("          Distancia: ${String.format("%.3f", distance)}")
        println("          Score de posición: ${String.format("%.3f", positionScore)}")
        
        return positionScore
    }
    
    /**
     * Calcula similitud entre dos bitmaps usando histogramas de color
     */
    private fun calculateHistogramSimilarity(bitmap1: Bitmap?, bitmap2: Bitmap?): Float {
        if (bitmap1 == null || bitmap2 == null) {
            println("🔍 [SockDetector] Bitmap nulo en comparación de histograma")
            return 0f
        }
        
        try {
            // Los bitmaps ya están en 64x64 desde analyzeRegion, no necesitamos redimensionar
            // Calcular histogramas RGB (64 bins por canal)
            val histogram1 = calculateRGBHistogram(bitmap1)
            val histogram2 = calculateRGBHistogram(bitmap2)
            
            // Normalizar histogramas
            val normalizedHist1 = normalizeHistogram(histogram1)
            val normalizedHist2 = normalizeHistogram(histogram2)
            
            // Calcular similitud usando distancia Chi-square
            val similarity = calculateChiSquareSimilarity(normalizedHist1, normalizedHist2)
            
            return similarity
            
        } catch (e: Exception) {
            println("🔍 [SockDetector] Error calculando similitud de histograma: ${e.message}")
            return 0f
        }
    }
    
    /**
     * Calcula histograma RGB de un bitmap
     */
    private fun calculateRGBHistogram(bitmap: Bitmap): IntArray {
        val histogram = IntArray(192) // 64 bins * 3 canales (R, G, B)
        
        for (y in 0 until bitmap.height) {
            for (x in 0 until bitmap.width) {
                val pixel = bitmap.getPixel(x, y)
                
                // Extraer componentes RGB
                val red = Color.red(pixel)
                val green = Color.green(pixel)
                val blue = Color.blue(pixel)
                
                // Calcular índices de bins (0-63 para cada canal)
                val redBin = (red * 64 / 256).coerceIn(0, 63)
                val greenBin = (green * 64 / 256).coerceIn(0, 63)
                val blueBin = (blue * 64 / 256).coerceIn(0, 63)
                
                // Incrementar contadores
                histogram[redBin]++
                histogram[64 + greenBin]++
                histogram[128 + blueBin]++
            }
        }
        
        return histogram
    }
    
    /**
     * Normaliza un histograma
     */
    private fun normalizeHistogram(histogram: IntArray): FloatArray {
        val total = histogram.sum().toFloat()
        return if (total > 0) {
            FloatArray(histogram.size) { histogram[it] / total }
        } else {
            FloatArray(histogram.size) { 0f }
        }
    }
    
    /**
     * Calcula similitud usando distancia Chi-square
     */
    private fun calculateChiSquareSimilarity(hist1: FloatArray, hist2: FloatArray): Float {
        var chiSquare = 0f
        
        for (i in hist1.indices) {
            val diff = hist1[i] - hist2[i]
            val sum = hist1[i] + hist2[i]
            
            if (sum > 0) {
                chiSquare += (diff * diff) / sum
            }
        }
        
        // Convertir Chi-square a similitud (0-1)
        // Chi-square menor = más similar
        val maxChiSquare = 2f // valor máximo esperado para histogramas normalizados
        val similarity = (maxChiSquare - chiSquare.coerceAtMost(maxChiSquare)) / maxChiSquare
        
        return similarity.coerceIn(0f, 1f)
    }
    
    /**
     * Limpia los bitmaps de una lista de medias para liberar memoria
     */
    private fun cleanupSockBitmaps(socks: List<Sock>) {
        socks.forEach { sock ->
            sock.bitmap?.recycle()
        }
    }
    
    /**
     * Función de testeo para ejecutar detección desde una imagen estática
     */
    suspend fun testFromStaticImage(bitmap: Bitmap): DetectionResult {
        println("🧪 [SockDetector] testFromStaticImage - Iniciando test con imagen estática")
        println("🧪 [SockDetector] testFromStaticImage - Bitmap: ${bitmap.width}x${bitmap.height}")
        
        // Usar el mismo flujo que detectSocks pero con el tamaño del bitmap como frame
        return detectSocks(bitmap, bitmap.width, bitmap.height)
    }
    
    /**
     * Calcula la confianza de emparejamiento entre dos medias (método legacy, mantenido por compatibilidad)
     */
    private fun calculateMatchConfidence(sock1: Sock, sock2: Sock): Float {
        return calculateMatchScore(sock1, sock2)
    }
} 