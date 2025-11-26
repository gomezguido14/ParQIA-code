import argparse
import json
import time

import cv2

from detector import ParkingDetector


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_video(config_path: str) -> None:
    cfg = load_config(config_path)

    source = cfg["source"]           # ruta al video o URL de cámara
    model_path = cfg.get("model_path", "yolov8n.pt")
    conf = cfg.get("conf", 0.4)
    frame_interval_sec = float(cfg.get("frame_interval_sec", 2))
    classes = cfg.get("classes", [2, 3, 5, 7])
    parking_slots = cfg.get("parking_slots", [])

    print(f"[ParQIA] Cargando modelo desde: {model_path}")
    detector = ParkingDetector(
        model_path=model_path,
        conf=conf,
        classes=classes,
    )

    print(f"[ParQIA] Abriendo fuente de video: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video/cámara: {source}")

    last_process_time = 0.0

    frame_index = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ParQIA] Fin del video o pérdida de stream.")
                break

            now = time.time()
            elapsed_since_last = now - last_process_time

            # Procesar un frame cada X segundos aprox
            if elapsed_since_last < frame_interval_sec:
                continue

            last_process_time = now
            frame_index += 1

            detections = detector.detect(frame)
            num_vehicles = len(detections)

            occupancy = None
            if parking_slots:
                occupancy = detector.compute_slot_occupancy(
                    detections=detections,
                    parking_slots=parking_slots,
                )

            print("=" * 60)
            print(f"Frame #{frame_index}")
            print(f"Vehículos detectados: {num_vehicles}")

            if occupancy:
                print(
                    f"Slots totales: {occupancy['total_slots']} | "
                    f"Ocupados: {occupancy['occupied_slots']} | "
                    f"Libres: {occupancy['free_slots']}"
                )
                print(f"IDs ocupados: {occupancy['occupied_slot_ids']}")

    finally:
        cap.release()
        total_time = time.time() - start_time
        print(f"[ParQIA] Procesamiento terminado. Tiempo total: {total_time:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="POC ParQIA - Análisis de video con YOLO"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta al archivo de configuración JSON",
    )

    args = parser.parse_args()
    process_video(args.config)


if __name__ == "__main__":
    main()