#!/usr/bin/env python3
"""
Flask MJPEG stream with custom YOLO11n traffic cone detection.

Run: python traffic_cone.py
Stream: http://<pi_ip>:5000
"""

from __future__ import annotations

import atexit
import signal
import sys
from pathlib import Path

if sys.prefix != sys.base_prefix:
    sys.path.insert(0, "/usr/lib/python3/dist-packages")

import cv2
from flask import Flask, Response
from picamera2 import MappedArray, Picamera2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "traffic_cone.pt"
TARGET_CLASS = "traffic_cone"
CONFIDENCE = 0.15
FRAME_SIZE = (640, 480)
INFERENCE_SIZE = 640
JPEG_QUALITY = 85
PORT = 5000

app = Flask(__name__)
picam2: Picamera2 | None = None
model: YOLO | None = None
target_class_id: int | None = None


def load_model() -> None:
    global model, target_class_id

    print(f"Loading YOLO model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Place traffic_cone.pt in the models/ folder."
        )

    model = YOLO(str(MODEL_PATH))
    target_class_id = None
    for class_id, name in model.names.items():
        if name == TARGET_CLASS:
            target_class_id = int(class_id)
            break

    if target_class_id is None:
        raise ValueError(f"Class '{TARGET_CLASS}' not found in model labels: {model.names}")

    print(f"YOLO ready — detecting '{TARGET_CLASS}' (class {target_class_id})")


def start_camera() -> None:
    global picam2

    print("Starting IMX500 camera...")
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"size": FRAME_SIZE},
            buffer_count=12,
        )
    )
    picam2.start()
    print(f"Camera running at {FRAME_SIZE[0]}x{FRAME_SIZE[1]}")


def stop_camera() -> None:
    global picam2
    if picam2 is not None:
        try:
            picam2.stop()
            picam2.close()
        except Exception:
            pass
        picam2 = None
        print("Camera stopped.")


def shutdown(*_args) -> None:
    stop_camera()


def capture_rgb_frame():
    request = picam2.capture_request()
    try:
        with MappedArray(request, "main") as m:
            return m.array.copy()
    finally:
        request.release()


def detect_objects(frame_rgb) -> list[tuple[int, int, int, int, float]]:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(
        frame_bgr,
        imgsz=INFERENCE_SIZE,
        conf=CONFIDENCE,
        verbose=False,
        device="cpu",
    )[0]

    detections: list[tuple[int, int, int, int, float]] = []
    for box in results.boxes:
        if int(box.cls[0]) != target_class_id:
            continue
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((x1, y1, x2, y2, confidence))
    return detections


def draw_detections(frame, detections) -> None:
    for x1, y1, x2, y2, confidence in detections:
        label = f"{TARGET_CLASS} {confidence * 100:.0f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        text_y = max(y1 - 8, text_height + 4)
        cv2.rectangle(
            frame,
            (x1, text_y - text_height - 4),
            (x1 + text_width + 4, text_y + baseline),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(
            frame, label, (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2,
        )


def generate():
    while True:
        frame = capture_rgb_frame()
        detections = detect_objects(frame)
        draw_detections(frame, detections)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, jpeg = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )


@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mars Rover — Traffic Cone Detection</title>
        <style>
            body { font-family: sans-serif; text-align: center; background: #111; color: #eee; }
            h1 { margin-top: 1rem; }
            p { color: #aaa; max-width: 680px; margin: 0.5rem auto 1rem; line-height: 1.4; }
            img { border: 2px solid #333; max-width: 100%; }
        </style>
    </head>
    <body>
        <h1>Mars Rover — Traffic Cone Detection</h1>
        <p>Live YOLO detection for <strong>traffic_cone</strong></p>
        <img src="/video" width="640">
    </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    atexit.register(shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        load_model()
        start_camera()
        print(f"Web stream: http://0.0.0.0:{PORT}")
        app.run(host="0.0.0.0", port=PORT, threaded=True)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        shutdown()
        sys.exit(1)
