import cv2 as cv
import numpy as np
from ultralytics import YOLO
import json
import os

# ========= CONFIGURACIÓN DE EQUIPOS Y COLORES =========
# Archivo donde el bot guarda la configuración de los equipos
EQUIPOS_FILE = "equipos.json"

# Umbral mínimo de píxeles para considerar un color
UMBRAL_COBERTURA = 0.08

# Kernel para limpieza morfológica
KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))

def cargar_rangos_equipos():
    """Carga los datos de los equipos desde el archivo JSON."""
    if not os.path.exists(EQUIPOS_FILE):
        print("Error: El archivo de equipos 'equipos.json' no se encontró. No se puede cargar la configuración.")
        return {}
    
    with open(EQUIPOS_FILE, "r") as f:
        equipos_data = json.load(f)

    team_ranges = {}
    for nombre_equipo, datos in equipos_data.items():
        color = datos["color"].lower()
        if "rojo" in color:
            team_ranges[nombre_equipo] = [
                ((0,   90, 60), (10, 255, 255)),   # rojos bajos
                ((170, 90, 60), (179, 255, 255)),  # rojos altos
            ]
        elif "azul" in color:
            team_ranges[nombre_equipo] = [
                ((95,  80, 60), (125, 255, 255)),
            ]
        elif "negro" in color or "arbitro" in color:
             team_ranges[nombre_equipo] = [
                ((0,   0, 0),   (179, 80, 80)),
            ]
        else:
            print(f"Advertencia: No hay rangos HSV definidos para el color '{color}' del equipo '{nombre_equipo}'.")

    return team_ranges

# --- TUS FUNCIONES DE CLASIFICACIÓN DE COLOR ---
def coverage_for_ranges(hsv_roi, ranges):
    """Calcula la cobertura de color dentro de un rango HSV dado."""
    total_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        low = np.array(low, dtype=np.uint8)
        high = np.array(high, dtype=np.uint8)
        mask = cv.inRange(hsv_roi, low, high)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, KERNEL, iterations=1)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, KERNEL, iterations=1)
        total_mask = cv.bitwise_or(total_mask, mask)
    return (total_mask > 0).mean()

# --- FUNCIÓN PRINCIPAL ---
def main():
    # Cargar el modelo YOLOv8 pre-entrenado
    yolo_model = YOLO("yolov8n-seg.pt") 
    
    # Cargar los rangos de colores de los equipos desde el archivo
    TEAM_RANGES = cargar_rangos_equipos()
    if not TEAM_RANGES:
        return

    # --- CONFIGURACIÓN DE CÁMARA ---
    CAM_INDEX = 0 
    cap = cv.VideoCapture(CAM_INDEX)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara con índice {CAM_INDEX}.")
        return

    print(f"Cámara con índice {CAM_INDEX} abierta correctamente.")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: Frame no disponible. Saliendo.")
            break
        
        results = yolo_model(frame, classes=[0], verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                w_p, h_p = x2 - x1, y2 - y1
                torso_x1 = int(x1 + w_p * 0.3)
                torso_x2 = int(x1 + w_p * 0.7)
                torso_y1 = int(y1 + h_p * 0.35)
                torso_y2 = int(y1 + h_p * 0.75)

                torso_x1 = max(0, torso_x1)
                torso_y1 = max(0, torso_y1)
                torso_x2 = min(frame.shape[1], torso_x2)
                torso_y2 = min(frame.shape[0], torso_y2)

                roi_bgr = frame[torso_y1:torso_y2, torso_x1:torso_x2]
                
                if roi_bgr.size == 0:
                    continue
                
                roi_bgr = cv.GaussianBlur(roi_bgr, (5,5), 0)
                hsv = cv.cvtColor(roi_bgr, cv.COLOR_BGR2HSV)
                
                coberturas = {}
                for team, rangos in TEAM_RANGES.items():
                    cov = coverage_for_ranges(hsv, rangos)
                    coberturas[team] = cov

                best_team, best_cov = max(coberturas.items(), key=lambda kv: kv[1]) if coberturas else ("Desconocido", 0.0)

                color_caja = (255, 255, 255)
                if best_cov >= UMBRAL_COBERTURA:
                    texto = f"{best_team}"
                    if "rojo" in best_team.lower():
                        color_caja = (0, 0, 255)
                    elif "azul" in best_team.lower():
                        color_caja = (255, 0, 0)
                    elif "negro" in best_team.lower() or "arbitro" in best_team.lower():
                        color_caja = (0, 255, 255)
                else:
                    texto = "Sin certeza"
                    color_caja = (0, 0, 0)

                cv.rectangle(frame, (x1, y1), (x2, y2), color_caja, 2)
                cv.putText(frame, texto, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color_caja, 2, cv.LINE_AA)
        
        cv.imshow("Clasificador de Equipos de Futbol", frame)

        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()