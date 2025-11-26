import os, time, cv2, numpy as np, requests
from collections import deque
from ultralytics import YOLO
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
SHEET_KEY = os.getenv("SHEET_KEY")
ROW_ID = int(os.getenv("ROW_ID","5"))
TOTAL = int(os.getenv("TOTAL_SPOTS","16"))
S = int(os.getenv("FRAME_EVERY_SECONDS","30"))         # cada cuántos segundos medir
UPDATE_MIN = int(os.getenv("UPDATE_MIN_SECONDS","30"))  # mínimo entre escrituras a Sheets
SOURCE = os.getenv("SOURCE","video.MOV")

# Google Sheets
scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
gc = gspread.authorize(creds)
sheet = gc.open_by_key(SHEET_KEY).sheet1

# YOLO (modelo más preciso)
model = YOLO("yolov8x.pt")

# cargar ROIs guardados
import numpy as np
rois = np.load("rois.npy", allow_pickle=True)

def apply_rois(frame):
    import cv2, numpy as np
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for poly in rois:
        pts = np.array(poly, np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)


def count_cars(img):
    res = model(img, imgsz=1536, conf=0.10, iou=0.5, augment=True)[0]
    # SOLO autos (car=2)
    return sum(1 for cid in res.boxes.cls if int(cid.item()) == 2)

def update_sheet_if_needed(libres, last_posted, last_time):
    now = time.time()

    # MODO DEBUG: escribir SIEMPRE que cambie el número de libres
    if last_posted is None or (libres != last_posted):
        try:
            ids = sheet.col_values(1)
            row_index = None
            for i, val in enumerate(ids, start=1):
                if str(val).strip() == str(ROW_ID):
                    row_index = i
                    break

            if row_index is None:
                # si no existe la fila para ese id, la creamos al final
                sheet.append_row([ROW_ID, "", "", libres, TOTAL, "TEST Düsseldorf", ""])
                print(f"[Sheets] Append nueva fila para id={ROW_ID}, libres={libres}")
            else:
                sheet.update_cell(row_index, 4, libres)  # D availableSpots
                sheet.update_cell(row_index, 5, TOTAL)   # E totalSpots
                try:
                    sheet.update_cell(
                        row_index,
                        8,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                except Exception as e:
                    print("[Sheets] No pude actualizar la columna H:", e)

                print(f"[Sheets] Update fila {row_index} id={ROW_ID}, libres={libres}")

            return libres, now

        except Exception as e:
            print("[Sheets] ERROR al escribir:", e)
            return last_posted, last_time

    else:
        # Para ver cuándo decide NO escribir
        print(f"[Sheets] No escribo (libres={libres}, last_posted={last_posted})")
        return last_posted, last_time

print("Iniciando ParQIA POC (video + ROI) … (Ctrl+C para salir)")

# Si es imagen: procesa 1 vez
if SOURCE.lower().endswith((".jpg",".jpeg",".png")):
    if SOURCE.startswith("http"):
        r = requests.get(SOURCE, timeout=10)
        arr = np.frombuffer(r.content, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        frame = cv2.imread(SOURCE)
    h, w = frame.shape[:2]
    y1 = int(h*0.15); y2 = h       # ROI: mitad inferior
    x1 = 0; x2 = w
    roi = frame[y1:y2, x1:x2]
    autos = count_cars(apply_rois(frame)) if roi is not None else 0
    libres = max(0, TOTAL - min(autos, TOTAL))
    print(datetime.now().strftime("%H:%M:%S"), f"Autos(ROI:{autos}) → Libres:{libres}")
    update_sheet_if_needed(libres, None, 0)
    raise SystemExit(0)

# Video / RTSP
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("No pude abrir la fuente de video:", SOURCE); raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_interval = max(1, int(round(fps * S)))
from collections import deque
hist = deque(maxlen=9)
last_posted, last_time = None, 0.0

while True:
    grabbed = True
    for _ in range(frame_interval-1):
        grabbed = cap.grab()
        if not grabbed: break
    if not grabbed:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        time.sleep(S); continue

    ok, frame = cap.retrieve()
    if not ok or frame is None:
        time.sleep(0.2); continue

    # --- recorte ROI: mitad inferior (ajustable) ---
    h, w = frame.shape[:2]
    y1 = int(h*0.15); y2 = h      # si aún cuenta tránsito, subí 0.5 a 0.6 o 0.7
    x1 = 0; x2 = w
    roi = frame[y1:y2, x1:x2]

    autos = count_cars(apply_rois(frame))
    hist.append(autos)
    autos_sm = int(np.median(hist))
    ocupadas = min(autos_sm, TOTAL)
    libres = max(0, TOTAL - ocupadas)
    print(datetime.now().strftime("%H:%M:%S"), f"Autos(ROI raw:{autos}→med:{autos_sm}) → Libres:{libres} (cada {S}s)")

    last_posted, last_time = update_sheet_if_needed(libres, last_posted, last_time)
    time.sleep(S)
