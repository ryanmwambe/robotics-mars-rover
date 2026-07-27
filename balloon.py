#!/usr/bin/env python3
"""
Flask MJPEG stream with hybrid balloon detection.

YOLO finds balloon shapes (ignores non-balloon objects).
HSV classifies the color inside each detected balloon.

Run: python balloon.py
Stream: http://<pi_ip>:5003
"""

from __future__ import annotations

import atexit
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
CROP_INFERENCE_SIZE = 320
SHAPE_CONFIDENCE = 0.08
MIN_COLOR_VOTE = 0.18
JPEG_QUALITY = 85
PORT = 5003

# Search windows: (cx, cy, radius, shape_model_file)
SEARCH_WINDOWS: list[tuple[int, int, int, str]] = [
    (228, 265, 75, "light_blue_balloon.pt"),
    (350, 275, 80, "pink_balloon.pt"),
    (408, 275, 75, "yellow_balloon.pt"),
    (483, 265, 80, "white_balloon.pt"),
    (554, 265, 80, "navy_blue_balloon.pt"),
]

# Table region — ignore ceiling, floor, and background.
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 120, 170, 625, 345
MIN_BOX_AREA = 1200
MAX_BOX_AREA = 40000
MIN_ASPECT = 0.45
MAX_ASPECT = 2.0
NMS_IOU = 0.45

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

    needed = {entry[3] for entry in SEARCH_WINDOWS}
    for model_name in sorted(needed):
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        print(f"Loading shape model {model_name}...")
        shape_models[model_name] = YOLO(str(model_path))

    print("Hybrid detection ready — YOLO shape + HSV color")


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


def _box_in_roi(x1: int, y1: int, x2: int, y2: int) -> bool:
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if cx < ROI_X1 or cx > ROI_X2 or cy < ROI_Y1 or cy > ROI_Y2:
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
    if 25 <= h <= 38 and s >= 130 and v >= 140:
        return "yellow"
    if 138 <= h <= 172 and v >= 160 and s <= 175:
        return "pink"
    if 108 <= h <= 118 and s >= 185 and 55 <= v <= 175:
        return "black"
    if 80 <= h <= 112 and 20 <= s <= 200 and 25 <= v <= 210:
        return "light_blue"
    if v >= 220 and s <= 85:
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
    margin_x = max(1, int((x2 - x1) * 0.18))
    margin_y = max(1, int((y2 - y1) * 0.18))
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
    vote_ratio = votes[best_key] / len(pixels)
    return best_key, vote_ratio


def _crop_region(
    frame_bgr,
    center_x: int,
    center_y: int,
    radius: int,
) -> tuple[object, int, int]:
    height, width = frame_bgr.shape[:2]
    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(width, center_x + radius)
    y2 = min(height, center_y + radius)
    return frame_bgr[y1:y2, x1:x2], x1, y1


def _find_balloon_boxes(frame_bgr) -> list[tuple[int, int, int, int, float]]:
    boxes: list[tuple[int, int, int, int, float]] = []

    for center_x, center_y, radius, model_name in SEARCH_WINDOWS:
        crop, offset_x, offset_y = _crop_region(frame_bgr, center_x, center_y, radius)
        if crop.size == 0:
            continue

        model = shape_models[model_name]
        results = model.predict(
            crop,
            imgsz=CROP_INFERENCE_SIZE,
            conf=SHAPE_CONFIDENCE,
            verbose=False,
            device="cpu",
        )[0]

        if results.boxes is None:
            continue

        best_conf = -1.0
        best_box: tuple[int, int, int, int, float] | None = None
        for box in results.boxes:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            abs_box = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
            if _box_in_roi(*abs_box) and confidence > best_conf:
                best_conf = confidence
                best_box = (*abs_box, confidence)

        if best_box is not None:
            boxes.append(best_box)

    return _nms_boxes(boxes)


def detect_objects(frame_rgb) -> list[tuple[str, int, int, int, int, float]]:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    balloon_boxes = _find_balloon_boxes(frame_bgr)

    candidates: list[tuple[str, int, int, int, int, float]] = []
    for x1, y1, x2, y2, shape_conf in balloon_boxes:
        color_key, color_conf = _classify_balloon_color(hsv, x1, y1, x2, y2)
        if color_key is None or color_conf < MIN_COLOR_VOTE:
            continue

        label = BALLOON_COLORS[color_key]["label"]
        combined_conf = min(0.99, shape_conf * 0.55 + color_conf * 0.45)
        candidates.append((label, x1, y1, x2, y2, combined_conf))

    candidates.sort(key=lambda item: item[5], reverse=True)

    used_labels: set[str] = set()
    detections: list[tuple[str, int, int, int, int, float]] = []
    for label, x1, y1, x2, y2, confidence in candidates:
        if label in used_labels:
            continue
        used_labels.add(label)
        detections.append((label, x1, y1, x2, y2, confidence))

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
        <p>Hybrid YOLO shape + HSV color detection for <strong>{colors}</strong></p>
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
