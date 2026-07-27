#!/usr/bin/env python3
"""
Flask MJPEG stream with hybrid balloon detection.

YOLO finds balloon shapes anywhere in the camera view.
HSV classifies the color inside each detected balloon.

Designed for the Mars Rover SA challenge — balloons on the ground
in front of the rover, one colour at a time.

Run: python balloon.py
Stream: http://<pi_ip>:5003
"""

from __future__ import annotations

import atexit
import math
import os
import signal
import sys
import threading
from pathlib import Path

if sys.prefix != sys.base_prefix:
    sys.path.insert(0, "/usr/lib/python3/dist-packages")

import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import MappedArray, Picamera2
from ultralytics import YOLO
from werkzeug.serving import make_server

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / "ballons" / "models"

FRAME_SIZE = (640, 480)
INFERENCE_SIZE = 640
SHAPE_CONFIDENCE = 0.02  # models score ~0.02 on real balloons
MIN_COLOR_VOTE = 0.35
JPEG_QUALITY = 85
PORT = 5003

# Accept balloons anywhere in frame (ground, close-up testing, etc.).
ROI_MARGIN = 8
MIN_BOX_AREA = 800
MAX_BOX_AREA = 120000
MIN_ASPECT = 0.35
MAX_ASPECT = 2.5
NMS_IOU = 0.45
MAX_DETECTIONS = 3  # report the best balloon(s) currently in view

SHAPE_MODELS = (
    "light_blue_balloon.pt",
    "pink_balloon.pt",
    "yellow_balloon.pt",
    "white_balloon.pt",
    "navy_blue_balloon.pt",
)

BALLOON_COLORS: dict[str, dict] = {
    "yellow": {
        "label": "yellow balloon",
        "box_color": (0, 220, 255),
        "text_color": (20, 20, 20),
    },
    "pink": {
        "label": "pink balloon",
        "box_color": (180, 105, 255),
        "text_color": (20, 20, 20),
    },
    "light_blue": {
        "label": "light blue balloon",
        "box_color": (255, 200, 80),
        "text_color": (20, 20, 20),
    },
    "black": {
        "label": "black balloon",
        "box_color": (180, 80, 40),
        "text_color": (255, 255, 255),
    },
    "white": {
        "label": "white balloon",
        "box_color": (220, 220, 220),
        "text_color": (20, 20, 20),
    },
}

app = Flask(__name__)
picam2: Picamera2 | None = None
shape_models: dict[str, YOLO] = {}
_running = True
_server = None


def load_models() -> None:
    global shape_models

    if not MODELS_DIR.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODELS_DIR}")

    for model_name in SHAPE_MODELS:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        print(f"Loading shape model {model_name}...")
        shape_models[model_name] = YOLO(str(model_path))

    print("Hybrid detection ready — full-frame YOLO shape + HSV color")


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
    global _running, _server
    if not _running:
        os._exit(0)

    _running = False
    stop_camera()
    print("\nStopped. Press Ctrl+C again to force quit.")
    if _server is not None:
        threading.Thread(target=_server.shutdown, daemon=True).start()


def capture_rgb_frame():
    request = picam2.capture_request()
    try:
        with MappedArray(request, "main") as m:
            return m.array.copy()
    finally:
        request.release()


def _box_area(x1: int, y1: int, x2: int, y2: int) -> int:
    return max(0, x2 - x1) * max(0, y2 - y1)


def _box_valid(x1: int, y1: int, x2: int, y2: int, frame_width: int, frame_height: int) -> bool:
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if cx < ROI_MARGIN or cy < ROI_MARGIN:
        return False
    if cx > frame_width - ROI_MARGIN or cy > frame_height - ROI_MARGIN:
        return False

    area = _box_area(x1, y1, x2, y2)
    if area < MIN_BOX_AREA or area > MAX_BOX_AREA:
        return False

    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    aspect = w / h
    return MIN_ASPECT <= aspect <= MAX_ASPECT


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = _box_area(ax1, ay1, ax2, ay2) + _box_area(bx1, by1, bx2, by2) - inter
    return inter / max(union, 1)


def _nms_boxes(
    boxes: list[tuple[int, int, int, int, float]],
) -> list[tuple[int, int, int, int, float]]:
    kept: list[tuple[int, int, int, int, float]] = []
    for box in sorted(boxes, key=lambda item: item[4], reverse=True):
        x1, y1, x2, y2, conf = box
        if any(_iou((x1, y1, x2, y2), kept_box[:4]) > NMS_IOU for kept_box in kept):
            continue
        kept.append(box)
    return kept


def _classify_hsv_pixel(h: int, s: int, v: int) -> str | None:
    # Allow bright specular highlights on shiny latex balloons (high V, high S).
    if 22 <= h <= 40 and s >= 80 and v >= 100:
        return "yellow"
    if 135 <= h <= 175 and s >= 40 and v >= 100:
        return "pink"
    if 105 <= h <= 125 and s >= 120 and v <= 180:
        return "black"
    if 68 <= h <= 118 and s >= 15 and v >= 20:
        return "light_blue"
    if v >= 180 and s <= 90:
        return "white"
    return None


def _classify_balloon_color(
    hsv: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[str | None, float]:
    h, w = hsv.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    patch = hsv[y1:y2, x1:x2]
    margin_x = max(1, int((x2 - x1) * 0.15))
    margin_y = max(1, int((y2 - y1) * 0.15))
    center = patch[margin_y:-margin_y or None, margin_x:-margin_x or None]
    if center.size == 0:
        center = patch

    pixels = center.reshape(-1, 3)
    votes: dict[str, int] = {}
    for px in pixels:
        color_key = _classify_hsv_pixel(int(px[0]), int(px[1]), int(px[2]))
        if color_key is not None:
            votes[color_key] = votes.get(color_key, 0) + 1

    if not votes:
        return None, 0.0

    best_key = max(votes, key=votes.get)
    classified = sum(votes.values())
    vote_ratio = votes[best_key] / max(classified, 1)
    return best_key, vote_ratio


def _prominence_score(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    shape_conf: float,
    color_conf: float,
    frame_width: int,
    frame_height: int,
) -> float:
    """Prefer large, centred, confident detections — the balloon in front."""
    area = _box_area(x1, y1, x2, y2)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    focus_x = frame_width / 2
    focus_y = frame_height * 0.55
    distance = math.hypot(cx - focus_x, cy - focus_y)
    max_distance = math.hypot(frame_width, frame_height)
    centrality = 1.0 - distance / max_distance
    size_score = min(1.0, area / 15000)
    return shape_conf * 0.35 + color_conf * 0.45 + centrality * 0.10 + size_score * 0.10


def _find_balloon_boxes(frame_bgr) -> list[tuple[int, int, int, int, float]]:
    height, width = frame_bgr.shape[:2]
    boxes: list[tuple[int, int, int, int, float]] = []

    for model in shape_models.values():
        results = model.predict(
            frame_bgr,
            imgsz=INFERENCE_SIZE,
            conf=SHAPE_CONFIDENCE,
            verbose=False,
            device="cpu",
        )[0]

        if results.boxes is None:
            continue

        for box in results.boxes:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if _box_valid(x1, y1, x2, y2, width, height):
                boxes.append((x1, y1, x2, y2, confidence))

    return _nms_boxes(boxes)


def detect_objects(frame_rgb) -> list[tuple[str, int, int, int, int, float]]:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    frame_height, frame_width = frame_bgr.shape[:2]
    balloon_boxes = _find_balloon_boxes(frame_bgr)

    candidates: list[tuple[str, int, int, int, int, float, float]] = []
    for x1, y1, x2, y2, shape_conf in balloon_boxes:
        color_key, color_conf = _classify_balloon_color(hsv, x1, y1, x2, y2)
        if color_key is None or color_conf < MIN_COLOR_VOTE:
            continue

        label = BALLOON_COLORS[color_key]["label"]
        score = _prominence_score(
            x1, y1, x2, y2, shape_conf, color_conf, frame_width, frame_height
        )
        display_conf = min(0.99, score)
        candidates.append((label, x1, y1, x2, y2, display_conf, score))

    candidates.sort(key=lambda item: item[6], reverse=True)

    used_labels: set[str] = set()
    detections: list[tuple[str, int, int, int, int, float]] = []
    for label, x1, y1, x2, y2, confidence, _score in candidates:
        if label in used_labels:
            continue
        used_labels.add(label)
        detections.append((label, x1, y1, x2, y2, confidence))
        if len(detections) >= MAX_DETECTIONS:
            break

    return detections


def draw_detections(frame, detections) -> None:
    label_to_style = {cfg["label"]: cfg for cfg in BALLOON_COLORS.values()}

    for label, x1, y1, x2, y2, confidence in detections:
        style = label_to_style[label]
        box_color = style["box_color"]
        text_color = style["text_color"]
        display = f"{label} {confidence * 100:.0f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            display, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
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
            frame, display, (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2,
        )


def generate():
    while _running:
        if picam2 is None:
            break

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
    colors = ", ".join(cfg["label"] for cfg in BALLOON_COLORS.values())
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
            button {{
                margin: 1rem;
                padding: 0.6rem 1.2rem;
                font-size: 1rem;
                background: #c0392b;
                color: #fff;
                border: none;
                border-radius: 6px;
                cursor: pointer;
            }}
            button:hover {{ background: #e74c3c; }}
        </style>
    </head>
    <body>
        <h1>Mars Rover — Balloon Detection</h1>
        <p>Detects whatever balloon is in front of the camera —
           <strong>{colors}</strong></p>
        <img src="/video" width="640">
        <br>
        <button onclick="fetch('/stop', {{method: 'POST'}}).then(() => window.close())">
            Stop detection
        </button>
    </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stop", methods=["POST"])
def stop():
    threading.Thread(target=shutdown, daemon=True).start()
    return "Stopping detection server."


if __name__ == "__main__":
    atexit.register(shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        load_models()
        start_camera()
        print(f"Balloon detection ready — stream: http://0.0.0.0:{PORT}")
        print("Press Ctrl+C to stop, or click 'Stop detection' in the browser.")
        _server = make_server("0.0.0.0", PORT, app, threaded=True)
        _server.serve_forever()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        shutdown()
        sys.exit(1)
