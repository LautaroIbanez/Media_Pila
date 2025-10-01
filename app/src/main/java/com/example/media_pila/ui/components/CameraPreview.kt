package com.example.media_pila.ui.components

import android.content.Context
import androidx.camera.core.Preview
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat

@Composable
fun CameraPreview(
    onPreviewReady: (PreviewView) -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    
    // Conservar la instancia del PreviewView usando remember
    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    
    // Invocar onPreviewReady solo cuando se crea la vista
    LaunchedEffect(previewView) {
        onPreviewReady(previewView)
    }
    
    AndroidView(
        factory = { previewView },
        modifier = modifier.fillMaxSize()
    )
} 