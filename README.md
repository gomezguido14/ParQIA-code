# [ParQIA] – Smart Parking Detection Prototype 🚗📍

ParQIA es un sistema de detección de lugares de estacionamiento basado en analisis de imagenes con Inteligencia Artificial.  
Este repositorio contiene el **prototipo funcional** utilizado para detectar espacios libres y ocupados usando **YOLOv8**, zonas ROI y procesamiento en tiempo real.

## 📌 Características principales

- 🔍 Detección de autos con YOLOv8 (modelos livianos incluidos)
- 📦 Sistema modular: detector, ROIs, stream de video
- 🎥 Compatible con cámaras IP y videos locales
- 🧠 ROI mapping para identificar lugares individuales
- ⚡ Actualización en tiempo real del estado de cada espacio
- 🗂️ Código limpio y organizado en `/src`

---

## 📁 Estructura del proyecto

## 🌐 Live demo (Lovable)

👉 Podés ver el prototipo web en funcionamiento acá:  
https://parkia-map-finder.lovable.app/

```
ParQIA-code/
│
├── models/                 # Modelos YOLO (.pt)
│   ├── yolov8m.pt
│   └── yolov8n.pt
│
├── src/
│   ├── detector.py         # Motor YOLO + detecciones
│   ├── parqia_yolo_stream.py # Lógica del streaming en tiempo real
│   ├── draw_roi.py         # Dibuja ROIs y estado de spots
│   └── main.py             # Script principal de ejecución
│
├── rois.npy                # Archivo de zonas de estacionamiento
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Este archivo
```

---

## ⚙️ Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/gomezguido14/ParQIA-code.git
cd ParQIA-code
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## ▶️ Cómo correr el prototipo

Para ejecutar la detección:

```bash
python src/main.py
```

Si querés usar una cámara IP, editá `parqia_yolo_stream.py` y reemplazá la URL del video.

---

## 🧪 Datos necesarios

El sistema requiere:

- `rois.npy` → posiciones de los lugares
- Un modelo YOLO (`yolov8n.pt` o `yolov8m.pt`)
- Video o stream de cámara

Ya están incluidos en este repo.

---

## 📡 Conexión futura (MVP)

Próximas integraciones previstas:

- 🟢 Webhook hacia base de datos (Supabase / Firestore)
- 🟢 SSE para enviar datos en tiempo real a apps web
- 🟢 Dashboard en navegador para ver el mapa

---

## 🤝 Colaboradores

Si querés contribuir, hacé un fork o creá un Pull Request.  
Para invitaciones directas: **Settings → Collaborators**.

---

## 📄 Licencia

Proyecto privado para validación tecnológica.  
© 2025 – [ParQIA]
