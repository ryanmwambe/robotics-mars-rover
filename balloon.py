#!/usr/bin/env python3
"""
Flask MJPEG stream with custom YOLO11n multi-color balloon detection.

Detects up to one balloon per color: red, white, yellow, blue, black.

Run: python balloon.py
Stream: http://<pi_ip>:5003
"""

from __future__ import annotations

import atexit
import math
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
MODEL_PATH = BASE_DIR / "models" / "balloon.pt"
CONFIDENCE = 0.15
FRAME_SIZE = (640, 480)
INFERENCE_SIZE = 640
JPEG_QUALITY = 85
PORT = 5003

# One detection per color — keep the highest-confidence box for each.
BALLOON_COLORS = ("red", "white", "yellow", "blue", "black")

# BGR draw colors and label text colors for each balloon color.
COLOR_STYLES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "red": ((0, 0, 255), (255, 255, 255)),
    "white": ((220, 220, 220), (20, 20, 20)),
    "yellow": ((0, 220, 255), (20, 20, 20)),
    "blue": ((255, 120, 0), (255, 255, 255)),
    "black": ((60, 60, 60), (255, 255, 255)),
}

# Balloons are round — reject elongated false positives.
MIN_ASPECT = 0.50
MAX_ASPECT = 2.00
MIN_BOX_SIZE = 25
MAX_BOX_SIZE = 200
MIN_BOX_AREA = 500
MAX_BOX_AREA = 25000

app = Flask(__name__)
picam2: Picamera2 | None = None
model: YOLO | None = None
# Maps normalized color name -> set of YOLO class ids.
color_class_ids: dict[str, set[int]] = {}


def _normalize_label(name: str) -> str:
    return name.lower().replace("_", "-").replace(" ", "-")


def _color_from_label(name: str) -> str | None:
    """Match model class names like 'red', 'red-balloon', 'balloon_red', etc."""
    normalized = _normalize_label(name)
    for color in BALLOON_COLORS:
        variants = {
            color,
            f"{color}-balloon",
            f"balloon-{color}",
            f"{color}_balloon",
            f"balloon_{color}",
        }
        if normalized in variants:
            return color
    return None


def load_model() -> None:
    global model, color_class_ids

    print(f"Loading YOLO model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Place balloon.pt in the models/ folder."
        )

    model = YOLO(str(MODEL_PATH))
    color_class_ids = {color: set() for color in BALLOON_COLORS}

    for class_id, name in model.names.items():
        color = _color_from_label(name)
        if color is not None:
            color_class_ids[color].add(int(class_id))

    found = {color: ids for color, ids in color_class_ids.items() if ids}
    if not found:
        raise ValueError(
            f"No balloon color classes found in model labels: {model.names}. "
            f"Expected classes named like: {', '.join(BALLOON_COLORS)}"
        )

    print("YOLO ready — balloon colors detected:")
    for color in BALLOON_COLORS:
        ids = sorted(color_class_ids[color])
        if ids:
            print(f"  {color}: class ids {ids}")


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


def _class_id_to_color(class_id: int) -> str | None:
    for color, ids in color_class_ids.items():
        if class_id in ids:
            return color
    return None


def _passes_shape_filter(
    x1: int, y1: int, x2: int, y2: int,
) -> bool:
    box_w = x2 - x1
    box_h = y2 - y1
    aspect = box_w / max(box_h, 1)
    area = box_w * box_h
    size = max(box_w, box_h)

    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return False
    if size < MIN_BOX_SIZE or size > MAX_BOX_SIZE:
        return False
    if area < MIN_BOX_AREA or area > MAX_BOX_AREA:
        return False
    return True


def _pick_best_per_color(
    candidates: list[tuple[str, int, int, int, int, float]],
    frame_width: int,
    frame_height: int,
) -> list[tuple[str, int, int, int, int, float]]:
    """Keep at most one balloon per color — prefer round, confident detections."""
    best_by_color: dict[str, tuple[str, int, int, int, int, float]] = {}
    focus_x = frame_width / 2
    focus_y = frame_height * 0.55

    for detection in candidates:
        color, x1, y1, x2, y2, confidence = detection
        if not _passes_shape_filter(x1, y1, x2, y2):
            continue

        box_w = x2 - x1
        box_h = y2 - y1
        aspect = box_w / max(box_h, 1)
        area = box_w * box_h
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        roundness = 1.0 - abs(1.0 - aspect)
        distance = math.hypot(center_x - focus_x, center_y - focus_y)
        score = confidence * 100 + roundness * 15 + area / 1000 - distance / 15

        current = best_by_color.get(color)
        if current is None or score > current[-1]:
            best_by_color[color] = (color, x1, y1, x2, y2, confidence, score)

    return [
        (color, x1, y1, x2, y2, confidence)
        for color, x1, y1, x2, y2, confidence, _score in best_by_color.values()
    ]


def detect_objects(frame_rgb) -> list[tuple[str, int, int, int, int, float]]:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    frame_height, frame_width = frame_bgr.shape[:2]

    results = model.predict(
        frame_bgr,
        imgsz=INFERENCE_SIZE,
        conf=CONFIDENCE,
        verbose=False,
        device="cpu",
    )[0]

    candidates: list[tuple[str, int, int, int, int, float]] = []
    if results.boxes is not None:
        for box in results.boxes:
            class_id = int(box.cls[0])
            color = _class_id_to_color(class_id)
            if color is None:
                continue
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            candidates.append((color, x1, y1, x2, y2, confidence))

    return _pick_best_per_color(candidates, frame_width, frame_height)


def draw_detections(frame, detections) -> None:
    for color, x1, y1, x2, y2, confidence in detections:
        box_color, text_color = COLOR_STYLES[color]
        label = f"{color} {confidence * 100:.0f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
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
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2,
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
    colors = ", ".join(BALLOON_COLORS)
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mars Rover — Balloon Detection</title>
        <style>
            body {{ font-family: sans-serif; text-align: center; background: #111; color: #eee; }}
            h1 {{ margin-top: 1rem; }}
            p {{ color: #aaa; max-width: 680px; margin: 0.5rem auto 1rem; line-height: 1.4; }}
            img {{ border: 2px solid #333; max-width: 100%; }}
        </style>
    </head>
    <body>
        <h1>Mars Rover — Balloon Detection</h1>
        <p>Live YOLO detection for <strong>{colors}</strong> balloons</p>
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
