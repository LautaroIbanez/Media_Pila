package com.example.media_pila.ui.screens

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddAPhoto
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
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
import java.io.IOException

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
    
    // Launcher para captura de foto con cámara
    val takePictureLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        bitmap?.let {
            viewModel.processStaticBitmap(it)
        }
    }
    
    // Launcher para elegir foto de galería
    val pickImageLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            try {
                val bitmap = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
                    val source = android.graphics.ImageDecoder.createSource(context.contentResolver, uri)
                    android.graphics.ImageDecoder.decodeBitmap(source)
                } else {
                    @Suppress("DEPRECATION")
                    MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
                }
                viewModel.processStaticBitmap(bitmap)
            } catch (e: IOException) {
                e.printStackTrace()
            }
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
                    onSave = { viewModel.savePairs() },
                    onTakePhoto = { takePictureLauncher.launch(null) },
                    onPickImage = { pickImageLauncher.launch("image/*") }
                )
            }
            
            is AppState.Detected -> {
                CameraScreen(
                    viewModel = viewModel,
                    detectionResult = (appState as AppState.Detected).result,
                    isDetecting = isDetecting,
                    onRetry = { viewModel.retryDetection() },
                    onSave = { viewModel.savePairs() },
                    onTakePhoto = { takePictureLauncher.launch(null) },
                    onPickImage = { pickImageLauncher.launch("image/*") }
                )
            }
            
            is AppState.StaticImageDetected -> {
                StaticImageScreen(
                    viewModel = viewModel,
                    result = (appState as AppState.StaticImageDetected).result,
                    imageBitmap = (appState as AppState.StaticImageDetected).imageBitmap,
                    onReturnToCamera = { viewModel.returnToCamera() },
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
    onSave: () -> Unit,
    onTakePhoto: () -> Unit,
    onPickImage: () -> Unit
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
                        ),
                        modifier = Modifier.weight(1f).padding(horizontal = 4.dp)
                    ) {
                        Text("Reintentar")
                    }
                    
                    Button(
                        onClick = onSave,
                        enabled = detectionResult?.pairs?.isNotEmpty() == true,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.primary
                        ),
                        modifier = Modifier.weight(1f).padding(horizontal = 4.dp)
                    ) {
                        Text("Guardar Pares")
                    }
                }
                
                Spacer(modifier = Modifier.height(12.dp))
                
                // Botones para foto estática
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    OutlinedButton(
                        onClick = onTakePhoto,
                        modifier = Modifier.weight(1f).padding(horizontal = 4.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.AddAPhoto,
                            contentDescription = "Tomar foto",
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Tomar Foto", fontSize = 12.sp)
                    }
                    
                    OutlinedButton(
                        onClick = onPickImage,
                        modifier = Modifier.weight(1f).padding(horizontal = 4.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.PhotoLibrary,
                            contentDescription = "Elegir foto",
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Galería", fontSize = 12.sp)
                    }
                }
            }
        }
    }
}

@Composable
private fun StaticImageScreen(
    viewModel: SockDetectionViewModel,
    result: com.example.media_pila.data.DetectionResult,
    imageBitmap: Bitmap,
    onReturnToCamera: () -> Unit,
    onSave: () -> Unit
) {
    Box(
        modifier = Modifier.fillMaxSize()
    ) {
        // Mostrar la imagen estática
        Image(
            bitmap = imageBitmap.asImageBitmap(),
            contentDescription = "Imagen analizada",
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Fit
        )
        
        // Overlay de detecciones
        DetectionOverlay(
            detectionResult = result,
            modifier = Modifier.fillMaxSize()
        )
        
        // Controles en la parte superior
        Row(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .background(
                    Color.Black.copy(alpha = 0.7f),
                    RoundedCornerShape(bottomStart = 16.dp, bottomEnd = 16.dp)
                )
                .padding(16.dp),
            horizontalArrangement = Arrangement.Center
        ) {
            Text(
                text = "Modo Foto Estática",
                color = Color.White,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
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
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "Medias: ${result.socks.size}",
                        color = Color.White,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Pares: ${result.pairs.size}",
                        color = Color.White,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "${result.processingTime}ms",
                        color = Color.White,
                        fontSize = 16.sp
                    )
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Botones de control
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    OutlinedButton(
                        onClick = onReturnToCamera,
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = Color.White
                        ),
                        modifier = Modifier.weight(1f).padding(horizontal = 4.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Videocam,
                            contentDescription = "Volver a cámara",
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Volver a Cámara")
                    }
                    
                    Button(
                        onClick = onSave,
                        enabled = result.pairs.isNotEmpty(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.primary
                        ),
                        modifier = Modifier.weight(1f).padding(horizontal = 4.dp)
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