import os, cv2, numpy as np

SRC = os.getenv("SOURCE","video.MOV")
cap = cv2.VideoCapture(SRC)
ok, frame = cap.read(); cap.release()
if not ok:
    raise SystemExit("No pude leer un frame del video para dibujar ROIs.")

win = "ROIs: click puntos | ENTER cierra polígono | n nuevo | s GUARDAR | q salir SIN guardar"
cv2.namedWindow(win)
rois, current = [], []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current.append((x,y))

cv2.setMouseCallback(win, click)

while True:
    vis = frame.copy()

    # dibuja cerrados (verde)
    for poly in rois:
        pts = np.array(poly, np.int32).reshape((-1,1,2))
        cv2.polylines(vis, [pts], True, (0,255,0), 2)

    # dibuja actual (amarillo)
    for i in range(1, len(current)):
        cv2.line(vis, current[i-1], current[i], (0,200,255), 2)
    for (x,y) in current:
        cv2.circle(vis, (x,y), 3, (0,140,255), -1)

    # instrucciones
    cv2.rectangle(vis, (0,0), (vis.shape[1], 32), (0,0,0), -1)
    cv2.putText(vis, "CLICK puntos | ENTER=cerrar poligono | n=nuevo | s=GUARDAR | q=salir SIN guardar",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow(win, vis)
    k = cv2.waitKey(20) & 0xFF

    if k == 13:  # ENTER -> cerrar polígono actual
        if len(current) >= 3:
            rois.append(current.copy())
            current = []
        else:
            # dibuja aviso mínimo 3 puntos
            pass
    elif k == ord('n'):  # empezar otro polígono (descarta el actual sin cerrar)
        current = []
    elif k == ord('s'):  # GUARDAR a rois.npy (solo si hay al menos 1 polígono cerrado)
        if len(rois) == 0:
            print("No hay polígonos cerrados. Cerrá con ENTER y luego presioná 's'.")
        else:
            path = os.path.join(os.getcwd(), "rois.npy")
            np.save(path, np.array(rois, dtype=object))
            print(f"Guardado {len(rois)} polígono(s) en {path}")
            break
    elif k == ord('q'):  # salir SIN guardar
        print("Saliste sin guardar (q). No se creó/actualizó rois.npy.")
        break

cv2.destroyAllWindows()
