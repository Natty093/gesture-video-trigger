import argparse
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import time

# ─────────────────────────────────────────────
# Configuración y Constantes
# ─────────────────────────────────────────────
ARCHIVO_VIDEO = "Scubaa.mp4"  

VERDE    = (0, 220, 60)
ROJO     = (0, 50, 220)
AMARILLO = (0, 210, 230)

CONEXIONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

# ─────────────────────────────────────────────
# Lógica de detección de gestos
# ─────────────────────────────────────────────
class DetectorGesto:
    def __init__(self, umbral_nariz: float = 0.15):
        self.umbral_nariz = umbral_nariz

    def actualizar(self, resultado) -> bool:
        # Condición 1: Deben detectarse al menos 2 manos
        if resultado is None or not resultado.hand_landmarks or len(resultado.hand_landmarks) < 2:
            return False

        centro_imagen = np.array([0.5, 0.4])  
        
        for hand_lm in resultado.hand_landmarks[:2]:
            indice = np.array([hand_lm[8].x, hand_lm[8].y]) 
            if float(np.linalg.norm(indice - centro_imagen)) < self.umbral_nariz:
                return True 
        return False

# ─────────────────────────────────────────────
# Dibujar landmarks manualmente
# ─────────────────────────────────────────────
def dibujar_mano(frame, hand_lm, w: int, h: int):
    puntos = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
    for a, b in CONEXIONES:
        cv2.line(frame, puntos[a], puntos[b], AMARILLO, 1)
    for p in puntos:
        cv2.circle(frame, p, 4, VERDE, -1)

# ─────────────────────────────────────────────
# Bucle principal
# ─────────────────────────────────────────────
def main(args: argparse.Namespace):
    # Configurar HandLandmarker
    base_options = mp_python.BaseOptions(
        model_asset_path="hand_landmarker.task",
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Cargando modelo MediaPipe...")
    landmarker = mp_vision.HandLandmarker.create_from_options(options)
    print("Modelo cargado OK.")

    detector = DetectorGesto(umbral_nariz=args.umbral_nariz)
    
    # Preparar cámara y video
    cam = cv2.VideoCapture(args.cam)
    if not cam.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara {args.cam}")
        return

    cap_video = cv2.VideoCapture(ARCHIVO_VIDEO)
    if not cap_video.isOpened():
        print(f"[ERROR] No se pudo abrir el video: {ARCHIVO_VIDEO}. Revisa que esté en la misma carpeta.")
        return

    nombre_ventana_video = "Video Gesto Activo"
    ventana_video_creada = False

    print("Iniciando... Presiona Q en la ventana de la cámara para salir.")

    while True:
        ok, frame = cam.read()
        if not ok:
            print("[WARN] No se recibió frame de la cámara.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ts_ms = int(time.time() * 1000)

        # Procesar frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        resultado = landmarker.detect_for_video(mp_image, ts_ms)

        # Dibujar landmarks de las manos
        if resultado.hand_landmarks:
            for hand_lm in resultado.hand_landmarks:
                dibujar_mano(frame, hand_lm, w, h)

        # Evaluar el gesto (2 manos en pantalla y 1 en la nariz)
        gesto_activo = detector.actualizar(resultado)

        if gesto_activo:
            if not ventana_video_creada:
                cv2.namedWindow(nombre_ventana_video, cv2.WINDOW_NORMAL)
                if args.fullscreen:
                    cv2.setWindowProperty(nombre_ventana_video, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                ventana_video_creada = True

            ok_vid, frame_vid = cap_video.read()
            # Si el video termina, lo reiniciamos (Loop)
            if not ok_vid:
                cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok_vid, frame_vid = cap_video.read()

            if ok_vid:
                cv2.imshow(nombre_ventana_video, frame_vid)
        else:
            # Si el gesto se rompe, destruimos la ventana inmediatamente
            if ventana_video_creada:
                cv2.destroyWindow(nombre_ventana_video)
                ventana_video_creada = False

        texto_estado = "GESTO ACTIVO (Reproduciendo)" if gesto_activo else "Esperando gesto..."
        color_estado = VERDE if gesto_activo else ROJO
        cv2.putText(frame, texto_estado, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_estado, 2)

        cv2.imshow("Detector Q para salir", frame)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break

    # Limpieza al salir
    cam.release()
    cap_video.release()
    landmarker.close()
    cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detector de gestos -> muestra video dinamicamente")
    parser.add_argument("--cam",          type=int,   default=0,
                        help="Índice de la cámara (default 0)")
    parser.add_argument("--umbral-nariz", type=float, default=0.15,
                        help="Radio de la zona de la nariz (default 0.15)")
    parser.add_argument("--fullscreen",   action="store_true",
                        help="Abre el video en pantalla completa")
    main(parser.parse_args())