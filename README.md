# Detector de Gestos Multiestado (Video e Imágenes)

Este proyecto es un panel interactivo basado en visión por computadora que utiliza la cámara web para detectar los movimientos de tus manos en tiempo real. Dependiendo de la postura o gesto que realices, el programa mostrará dinámicamente imágenes específicas o reproducirá un video.

Está construido utilizando **Python**, **OpenCV** y la API de **MediaPipe** (v0.10+).

## 🎮 Gestos Soportados

El programa evalúa tus movimientos en tiempo real y reacciona de la siguiente manera (en orden de prioridad):

1. **Ambas Manos en la Cabeza:** Muestra la imagen `cabeza.jpg`.
2. **Dos Manos Visibles + 1 Mano en la Nariz:** Reproduce el video en bucle `Scubaa.mp4`.
3. **Una Mano en la Boca:** Muestra la imagen `boca.jpg`.
4. **Dedo Índice Apuntando Arriba:** Muestra la imagen `arriba.jpg`.

*(Si no realizas ningún gesto válido, el visor multimedia se cerrará automáticamente).*

---

## 🛠️ Requisitos Previos

Asegúrate de tener instalado **Python 3.8 o superior** (probado en Python 3.11).

### Dependencias
Instala las librerías necesarias ejecutando el siguiente comando en tu terminal:

```bash
pip install mediapipe opencv-python numpy