from ultralytics import YOLO
import numpy as np


class ParkingDetector:
    """
    Wrapper simple alrededor de YOLO para:
      - Detectar vehículos en un frame
      - (Opcional) calcular ocupación de slots de estacionamiento
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.4,
        device: str | None = None,
        classes: list[int] | None = None,
    ) -> None:
        """
        :param model_path: ruta al modelo YOLO (yolov8n.pt o tu best.pt)
        :param conf: umbral de confianza
        :param device: "cpu", "cuda" o None (auto)
        :param classes: lista de IDs de clase a filtrar (COCO: 2=coche, 3=moto, 5=bus, 7=camión)
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.classes = classes

    def detect(self, frame) -> list[dict]:
        """
        Corre YOLO sobre un frame (imagen BGR de OpenCV).

        Devuelve una lista de dicts:
        {
            "cls_id": int,
            "cls_name": str,
            "conf": float,
            "bbox": [x1, y1, x2, y2]
        }
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            device=self.device,
            verbose=False
        )

        detections: list[dict] = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if self.classes is not None and cls_id not in self.classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].item())
                cls_name = self.model.names[cls_id]

                detections.append(
                    {
                        "cls_id": cls_id,
                        "cls_name": cls_name,
                        "conf": conf,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        return detections

    @staticmethod
    def _point_in_bbox(cx: float, cy: float, bbox: list[float]) -> bool:
        x1, y1, x2, y2 = bbox
        return (x1 <= cx <= x2) and (y1 <= cy <= y2)

    def compute_slot_occupancy(
        self,
        detections: list[dict],
        parking_slots: list[dict],
    ) -> dict:
        """
        Calcula qué slots están ocupados, asumiendo que cada slot es un bbox.

        parking_slots: lista de dicts con:
            {
              "id": 1,
              "x1": ...,
              "y1": ...,
              "x2": ...,
              "y2": ...
            }
        """

        occupied_ids: set[int] = set()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            for slot in parking_slots:
                slot_bbox = [slot["x1"], slot["y1"], slot["x2"], slot["y2"]]
                if self._point_in_bbox(cx, cy, slot_bbox):
                    occupied_ids.add(slot["id"])

        total_slots = len(parking_slots)
        occupied_slots = len(occupied_ids)
        free_slots = max(0, total_slots - occupied_slots)

        return {
            "total_slots": total_slots,
            "occupied_slots": occupied_slots,
            "free_slots": free_slots,
            "occupied_slot_ids": sorted(list(occupied_ids)),
        }