#!/usr/bin/env python3
"""Diagnose YOLO cone detection on live camera."""

import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, "/usr/lib/python3/dist-packages")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO
from picamera2 import MappedArray, Picamera2

OUT = Path(__file__).resolve().parent / "debug"
OUT.mkdir(exist_ok=True)

MODEL = Path(__file__).resolve().parent / "models" / "best.pt"
model = YOLO(str(MODEL))
print("Model classes:", model.names)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}, buffer_count=4))
picam2.start()
time.sleep(1.5)

request = picam2.capture_request()
with MappedArray(request, "main") as m:
    frame_rgb = m.array.copy()
request.release()
picam2.stop()
picam2.close()

cv2.imwrite(str(OUT / "yolo_live_frame.jpg"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
print(f"Saved {OUT}/yolo_live_frame.jpg")

# Test multiple confidence / size settings
for imgsz in [320, 416, 640]:
    for conf in [0.5, 0.35, 0.25, 0.15]:
        results = model.predict(frame_rgb, imgsz=imgsz, conf=conf, verbose=False, device="cpu")[0]
        n = len(results.boxes) if results.boxes is not None else 0
        details = []
        if n:
            for box in results.boxes:
                c = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                details.append(f"conf={c:.2f} box=({x1},{y1},{x2},{y2})")
        print(f"imgsz={imgsz} conf={conf}: {n} detections {details}")

# Best-effort run at low conf for visualization
results = model.predict(frame_rgb, imgsz=640, conf=0.15, verbose=False, device="cpu")[0]
vis = frame_rgb.copy()
if results.boxes is not None:
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        c = float(box.conf[0])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{c:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.imwrite(str(OUT / "yolo_live_result.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print(f"Saved {OUT}/yolo_live_result.jpg")

# Also test BGR input (common mistake)
frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
results_bgr = model.predict(frame_bgr, imgsz=640, conf=0.25, verbose=False, device="cpu")[0]
n_bgr = len(results_bgr.boxes) if results_bgr.boxes is not None else 0
print(f"BGR input at conf=0.25: {n_bgr} detections")
