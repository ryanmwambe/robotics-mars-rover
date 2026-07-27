#!/usr/bin/env python3
"""
Flask MJPEG stream with HSV color-based balloon detection.

Finds round blobs in red, white, yellow, blue, and black — no ML model needed.

Run: python balloon.py
Stream: http://<pi_ip>:5003
"""

from __future__ import annotations

import atexit
import math
import signal
import sys

if sys.prefix != sys.base_prefix:
    sys.path.insert(0, "/usr/lib/python3/dist-packages")

import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import MappedArray, Picamera2

FRAME_SIZE = (640, 480)
JPEG_QUALITY = 85
PORT = 5003

# HSV ranges: (lower, upper) — OpenCV uses H 0-179.
COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [
        ((0, 120, 70), (10, 255, 255)),
        ((170, 120, 70), (179, 255, 255)),
    ],
    "yellow": [
        ((18, 100, 100), (35, 255, 255)),
    ],
    "blue": [
        ((95, 100, 70), (130, 255, 255)),
    ],
    "white": [
        ((0, 0, 180), (179, 45, 255)),
    ],
    "black": [
        ((0, 0, 0), (179, 255, 55)),
    ],
}

COLOR_STYLES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "red": ((0, 0, 255), (255, 255, 255)),
    "white": ((220, 220, 220), (20, 20, 20)),
    "yellow": ((0, 220, 255), (20, 20, 20)),
    "blue": ((255, 120, 0), (255, 255, 255)),
    "black": ((60, 60, 60), (255, 255, 255)),
}

MIN_AREA = 800
MAX_AREA = 40000
MIN_CIRCULARITY = 0.45
MIN_ASPECT = 0.55
MAX_ASPECT = 1.80

app = Flask(__name__)
picam2: Picamera2 | None = None


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


def _color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in COLOR_RANGES[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _find_best_blob(
    mask: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int, float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[int, int, int, int, float] | None = None
    best_score = -1.0
    focus_x = frame_width / 2
    focus_y = frame_height * 0.55

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        center_x = x + w / 2
        center_y = y + h / 2
        distance = math.hypot(center_x - focus_x, center_y - focus_y)
        score = circularity * 50 + area / 500 - distance / 20

        if score > best_score:
            best_score = score
            confidence = min(0.99, 0.5 + circularity * 0.4 + area / MAX_AREA * 0.1)
            best = (x, y, x + w, y + h, confidence)

    return best


def detect_objects(frame_rgb) -> list[tuple[str, int, int, int, int, float]]:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    frame_height, frame_width = hsv.shape[:2]

    detections: list[tuple[str, int, int, int, int, float]] = []
    for color in COLOR_RANGES:
        mask = _color_mask(hsv, color)
        blob = _find_best_blob(mask, frame_width, frame_height)
        if blob is not None:
            x1, y1, x2, y2, confidence = blob
            detections.append((color, x1, y1, x2, y2, confidence))

    return detections


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
    colors = ", ".join(COLOR_RANGES)
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
        <p>Color-based detection for <strong>{colors}</strong> balloons (no model file needed)</p>
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
        start_camera()
        print(f"Balloon color detection ready — stream: http://0.0.0.0:{PORT}")
        app.run(host="0.0.0.0", port=PORT, threaded=True)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        shutdown()
        sys.exit(1)
