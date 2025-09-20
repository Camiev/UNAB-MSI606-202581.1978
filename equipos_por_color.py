import cv2 as cv
import numpy as np
from ultralytics import YOLO
import json
import os

# ========= CONFIGURACIÓN =========
EQUIPOS_FILE = "equipos.json"
UMBRAL_COBERTURA = 0.08
KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))

# Rangos HSV solo para rojo, azul y negro
RANGOS_BASE = {
    "rojo": [
        ((0,   90, 60), (10, 255, 255)),
        ((170, 90, 60), (179, 255, 255)),
    ],
    "azul": [
        ((95,  80, 60), (125, 255, 255)),
    ],
    "negro": [
        ((0,   0, 0),   (179, 80, 80)),
    ],
}

def cargar_rangos_equipos():
    if not os.path.exists(EQUIPOS_FILE):
        print("Error: no existe equipos.json")
        return {}

    with open(EQUIPOS_FILE, "r", encoding="utf-8") as f:
        equipos_data = json.load(f)

    team_ranges = {}
    for nombre_equipo, datos in equipos_data.items():
        color = datos.get("color", "").lower()
        if color in RANGOS_BASE:
            team_ranges[nombre_equipo] = RANGOS_BASE[color]
        else:
            print(f"Advertencia: color '{color}' no mapeado en '{nombre_equipo}'")
    return team_ranges

def coverage_for_ranges(hsv_roi, ranges):
    total_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        low = np.array(low, dtype=np.uint8)
        high = np.array(high, dtype=np.uint8)
        mask = cv.inRange(hsv_roi, low, high)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, KERNEL, iterations=1)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, KERNEL, iterations=1)
        total_mask = cv.bitwise_or(total_mask, mask)
    return (total_mask > 0).mean()

def main():
    cap = cv.VideoCapture(0)  # cámbialo por "video.mp4" si quieres archivo
    if not cap.isOpened():
        print("Error: no se pudo abrir cámara/video")
        return

    yolo_model = YOLO("yolov8n-seg.pt")
    TEAM_RANGES = cargar_rangos_equipos()
    if not TEAM_RANGES:
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = yolo_model(frame, classes=[0], verbose=False)
        for r in results:
            if r.masks is None:
                continue

            boxes = r.boxes
            masks = r.masks.data.cpu().numpy()
            H, W = frame.shape[:2]

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                mask_i = (masks[i] > 0.5).astype(np.uint8) * 255
                if mask_i.shape[0] != H or mask_i.shape[1] != W:
                    mask_i = cv.resize(mask_i, (W, H), interpolation=cv.INTER_NEAREST)

                # ROI = torso 35%-75% de la caja
                h_box = y2 - y1
                ty1 = int(y1 + 0.35 * h_box)
                ty2 = int(y1 + 0.75 * h_box)
                tx1 = int(x1 + 0.3 * (x2 - x1))
                tx2 = int(x1 + 0.7 * (x2 - x1))
                roi_mask = mask_i[ty1:ty2, tx1:tx2]
                roi_bgr = frame[ty1:ty2, tx1:tx2]
                if roi_bgr.size == 0:
                    continue

                roi_bgr = cv.bitwise_and(roi_bgr, roi_bgr, mask=roi_mask)
                hsv = cv.cvtColor(cv.GaussianBlur(roi_bgr, (5,5), 0), cv.COLOR_BGR2HSV)

                coberturas = {}
                for team, rangos in TEAM_RANGES.items():
                    cov = coverage_for_ranges(hsv, rangos)
                    coberturas[team] = cov

                best_team, best_cov = max(coberturas.items(), key=lambda kv: kv[1]) if coberturas else ("Desconocido", 0)
                if best_cov < UMBRAL_COBERTURA:
                    best_team = "Sin certeza"

                color_caja = (255,255,255)
                if "rojo" in best_team.lower():
                    color_caja = (0,0,255)
                elif "azul" in best_team.lower():
                    color_caja = (255,0,0)
                elif "arbitro" in best_team.lower() or "negro" in best_team.lower():
                    color_caja = (0,0,0)

                cv.rectangle(frame, (x1,y1), (x2,y2), color_caja, 2)
                cv.putText(frame, f"{best_team}", (x1,y1-10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, color_caja if color_caja!=(0,0,0) else (255,255,255), 2)

        cv.imshow("Clasificador 3 equipos", frame)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
