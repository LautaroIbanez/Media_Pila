package com.example.media_pila.ui.screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.media_pila.data.AppState
import com.example.media_pila.ui.components.CameraPreview
import com.example.media_pila.ui.components.DetectionOverlay
import com.example.media_pila.viewmodel.SockDetectionViewModel

@Composable
fun MainScreen(
    viewModel: SockDetectionViewModel,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val appState by viewModel.appState.collectAsStateWithLifecycle()
    val detectionResult by viewModel.detectionResult.collectAsStateWithLifecycle()
    val isDetecting by viewModel.isDetecting.collectAsStateWithLifecycle()
    
    // Launcher para permisos de cámara
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            viewModel.checkCameraPermission()
        }
    }
    
    Box(
        modifier = modifier.fillMaxSize()
    ) {
        when (appState) {
            is AppState.Loading -> {
                LoadingScreen()
            }
            
            is AppState.CameraPermissionRequired -> {
                PermissionScreen(
                    onRequestPermission = {
                        permissionLauncher.launch(android.Manifest.permission.CAMERA)
                    }
                )
            }
            
            is AppState.Detecting -> {
                CameraScreen(
                    viewModel = viewModel,
                    detectionResult = detectionResult,
                    isDetecting = isDetecting,
                    onRetry = { viewModel.retryDetection() },
                    onSave = { viewModel.savePairs() }
                )
            }
            
            is AppState.Detected -> {
                CameraScreen(
                    viewModel = viewModel,
                    detectionResult = (appState as AppState.Detected).result,
                    isDetecting = isDetecting,
                    onRetry = { viewModel.retryDetection() },
                    onSave = { viewModel.savePairs() }
                )
            }
            
            is AppState.Error -> {
                ErrorScreen(
                    message = (appState as AppState.Error).message,
                    onRetry = { viewModel.checkCameraPermission() }
                )
            }
        }
    }
}

@Composable
private fun LoadingScreen() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        CircularProgressIndicator()
    }
}

@Composable
private fun PermissionScreen(
    onRequestPermission: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Permiso de Cámara Requerido",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            textAlign = androidx.compose.ui.text.style.TextAlign.Center
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        Text(
            text = "Esta aplicación necesita acceso a la cámara para detectar y emparejar medias.",
            fontSize = 16.sp,
            textAlign = androidx.compose.ui.text.style.TextAlign.Center
        )
        
        Spacer(modifier = Modifier.height(32.dp))
        
        Button(
            onClick = onRequestPermission,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Conceder Permiso")
        }
    }
}

@Composable
private fun CameraScreen(
    viewModel: SockDetectionViewModel,
    detectionResult: com.example.media_pila.data.DetectionResult?,
    isDetecting: Boolean,
    onRetry: () -> Unit,
    onSave: () -> Unit
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    
    Box(
        modifier = Modifier.fillMaxSize()
    ) {
        // Vista previa de la cámara
        CameraPreview(
            onPreviewReady = { previewView ->
                // Configurar la vista previa en el ViewModel
                viewModel.setPreviewView(previewView, lifecycleOwner)
            },
            modifier = Modifier.fillMaxSize()
        )
        
        // Overlay de detecciones
        DetectionOverlay(
            detectionResult = detectionResult,
            modifier = Modifier.fillMaxSize()
        )
        
        // Indicador de detección
        if (isDetecting) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.3f)),
                contentAlignment = Alignment.Center
            ) {
                Card(
                    modifier = Modifier.padding(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surface
                    )
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(24.dp)
                        )
                        Spacer(modifier = Modifier.width(16.dp))
                        Text("Detectando medias...")
                    }
                }
            }
        }
        
        // Controles en la parte inferior
        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(
                    Color.Black.copy(alpha = 0.7f),
                    RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp)
                )
                .padding(16.dp)
        ) {
            Column {
                // Información de detección
                detectionResult?.let { result ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = "Medias: ${result.socks.size}",
                            color = Color.White,
                            fontSize = 16.sp
                        )
                        Text(
                            text = "Pares: ${result.pairs.size}",
                            color = Color.White,
                            fontSize = 16.sp
                        )
                        Text(
                            text = "${result.processingTime}ms",
                            color = Color.White,
                            fontSize = 16.sp
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(16.dp))
                }
                
                // Botones de control
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    Button(
                        onClick = onRetry,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.secondary
                        )
                    ) {
                        Text("Reintentar")
                    }
                    
                    Button(
                        onClick = onSave,
                        enabled = detectionResult?.pairs?.isNotEmpty() == true,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.primary
                        )
                    ) {
                        Text("Guardar Pares")
                    }
                }
            }
        }
    }
}

@Composable
private fun ErrorScreen(
    message: String,
    onRetry: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Error",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.error
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        Text(
            text = message,
            fontSize = 16.sp,
            textAlign = androidx.compose.ui.text.style.TextAlign.Center
        )
        
        Spacer(modifier = Modifier.height(32.dp))
        
        Button(
            onClick = onRetry,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Reintentar")
        }
    }
} 