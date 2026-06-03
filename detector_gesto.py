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
IMG_ARRIBA    = "arriba.jpg"
IMG_BOCA      = "boca.jpg"
IMG_CABEZA    = "cabeza.jpg"

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
# Funciones de ayuda
# ─────────────────────────────────────────────
def cargar_imagen_o_fallback(ruta: str, titulo: str):
    img = cv2.imread(ruta)
    if img is None:
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(img, f"Falta imagen:", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, ROJO, 2)
        cv2.putText(img, ruta, (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, AMARILLO, 2)
        cv2.putText(img, titulo, (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, VERDE, 2)
    return img

def dibujar_mano(frame, hand_lm, w: int, h: int):
    puntos = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
    for a, b in CONEXIONES:
        cv2.line(frame, puntos[a], puntos[b], AMARILLO, 1)
    for p in puntos:
        cv2.circle(frame, p, 4, VERDE, -1)

# ─────────────────────────────────────────────
# Lógica de detección de gestos
# ─────────────────────────────────────────────
class DetectorGesto:
    def __init__(self):
        self.zona_nariz = np.array([0.5, 0.4])
        self.zona_boca = np.array([0.5, 0.65])
        self.umbral_zona = 0.15
        self.limite_y_cabeza = 0.35

    def actualizar(self, resultado) -> str:
        if resultado is None or not resultado.hand_landmarks:
            return "NINGUNO"

        manos = resultado.hand_landmarks

        # Evaluar gestos que requieren 2 MANOS
        if len(manos) >= 2:
            mano1_arriba = manos[0][0].y < self.limite_y_cabeza or manos[0][8].y < self.limite_y_cabeza
            mano2_arriba = manos[1][0].y < self.limite_y_cabeza or manos[1][8].y < self.limite_y_cabeza
            
            # Prioridad 1: Manos en la cabeza
            if mano1_arriba and mano2_arriba:
                return "CABEZA"

            # Prioridad 2: Mano en la nariz (y la otra presente)
            for hand_lm in manos[:2]:
                indice = np.array([hand_lm[8].x, hand_lm[8].y])
                if float(np.linalg.norm(indice - self.zona_nariz)) < self.umbral_zona:
                    return "NARIZ"

        # Evaluar gestos de 1 MANO (buscamos en todas las detectadas)
        for hand_lm in manos:
            indice = np.array([hand_lm[8].x, hand_lm[8].y])
            
            # Prioridad 3: Mano en la boca
            if float(np.linalg.norm(indice - self.zona_boca)) < self.umbral_zona:
                return "BOCA"

            # Prioridad 4: Dedo arriba
            indice_estirado = hand_lm[8].y < hand_lm[6].y
            medio_doblado   = hand_lm[12].y > hand_lm[10].y
            anular_doblado  = hand_lm[16].y > hand_lm[14].y
            menique_doblado = hand_lm[20].y > hand_lm[18].y

            if indice_estirado and medio_doblado and anular_doblado and menique_doblado:
                return "ARRIBA"

        return "NINGUNO"

# ─────────────────────────────────────────────
# Bucle principal
# ─────────────────────────────────────────────
def main(args: argparse.Namespace):
    # Cargar recursos
    imagenes = {
        "ARRIBA": cargar_imagen_o_fallback(IMG_ARRIBA, "Gesto: Dedo Arriba"),
        "BOCA":   cargar_imagen_o_fallback(IMG_BOCA, "Gesto: Mano en Boca"),
        "CABEZA": cargar_imagen_o_fallback(IMG_CABEZA, "Gesto: Manos en Cabeza"),
    }

    cap_video = cv2.VideoCapture(ARCHIVO_VIDEO)
    if not cap_video.isOpened():
        print(f"[WARN] No se pudo abrir {ARCHIVO_VIDEO}. El gesto de nariz no mostrara video.")

    # Configurar MediaPipe
    base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task", delegate=mp_python.BaseOptions.Delegate.CPU)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options, running_mode=mp_vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)
    detector = DetectorGesto()
    
    cam = cv2.VideoCapture(args.cam)
    
    nombre_ventana_popup = "Visor Multimedia"
    ventana_creada = False

    print("Iniciando... Presiona Q en la ventana de la cámara para salir.")

    while True:
        ok, frame = cam.read()
        if not ok: break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ts_ms = int(time.time() * 1000)

        # Procesamiento MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        resultado = landmarker.detect_for_video(mp_image, ts_ms)

        if resultado.hand_landmarks:
            for hand_lm in resultado.hand_landmarks:
                dibujar_mano(frame, hand_lm, w, h)

        # Evaluar estado
        estado_actual = detector.actualizar(resultado)

        # ---------------------------------------------------------
        # Lógica de Ventanas Emergentes 
        # ---------------------------------------------------------
        if estado_actual in ["CABEZA", "BOCA", "ARRIBA"]:
            if not ventana_creada:
                cv2.namedWindow(nombre_ventana_popup, cv2.WINDOW_NORMAL)
                ventana_creada = True
            cv2.imshow(nombre_ventana_popup, imagenes[estado_actual])

        elif estado_actual == "NARIZ" and cap_video.isOpened():
            if not ventana_creada:
                cv2.namedWindow(nombre_ventana_popup, cv2.WINDOW_NORMAL)
                ventana_creada = True
            
            ok_vid, frame_vid = cap_video.read()
            if not ok_vid: # Reiniciar video si se acaba
                cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok_vid, frame_vid = cap_video.read()
            
            if ok_vid:
                cv2.imshow(nombre_ventana_popup, frame_vid)

        else:
            # Si no hay gesto, cerramos el visor
            if ventana_creada:
                cv2.destroyWindow(nombre_ventana_popup)
                ventana_creada = False

        # HUD en la cámara
        texto = f"ESTADO: {estado_actual}"
        color = ROJO if estado_actual == "NINGUNO" else VERDE
        cv2.putText(frame, texto, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Detector de gestos Q para salir", frame)

        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break

    # Limpieza
    cam.release()
    cap_video.release()
    landmarker.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0)
    main(parser.parse_args())