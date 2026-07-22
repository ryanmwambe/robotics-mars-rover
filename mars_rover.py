#!/usr/bin/env python3
"""
Mars Rover Challenge — precise detection for competition props:
  - light green tennis ball
  - red traffic cone with white reflector band near the top
  - claw hammer (metal head + narrow wooden handle)

Uses strict multi-cue OpenCV rules and rejects skin / furniture false positives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2

PORT = 5001
FRAME_SIZE = (640, 480)
MIN_DISPLAY_CONFIDENCE = 0.68
MAX_PER_CLASS = 1

app = Flask(__name__)


def start_camera():
    global picam2
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"size": FRAME_SIZE, "format": "RGB888"},
            buffer_count=4,
        )
    )
    picam2.start()
    print("Mars Rover detector running (strict OpenCV)")


@dataclass
class Detection:
    name: str
    box: tuple[int, int, int, int]
    confidence: float


def _skin_mask(hsv: np.ndarray) -> np.ndarray:
    lower1 = cv2.inRange(hsv, (0, 25, 60), (25, 180, 255))
    lower2 = cv2.inRange(hsv, (160, 30, 60), (180, 180, 255))
    return lower1 | lower2


def _overlap_ratio(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    roi = mask[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / roi.size


def _merge_overlapping(
    detections: list[Detection], iou_threshold: float = 0.4
) -> list[Detection]:
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []

    def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        return inter / (aw * ah + bw * bh - inter)

    for det in detections:
        if det.confidence < MIN_DISPLAY_CONFIDENCE:
            continue
        if all(iou(det.box, k.box) < iou_threshold or det.name != k.name for k in kept):
            kept.append(det)

    limited: list[Detection] = []
    counts: dict[str, int] = {}
    for det in kept:
        counts[det.name] = counts.get(det.name, 0)
        if counts[det.name] >= MAX_PER_CLASS:
            continue
        counts[det.name] += 1
        limited.append(det)
    return limited


def detect_tennis_balls(frame: np.ndarray, skin: np.ndarray) -> list[Detection]:
    """Light green, compact, highly circular blob — not skin, not furniture."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # light green tennis ball (hue ~50-60 in OpenCV HSV)
    mask = cv2.inRange(hsv, (42, 70, 110), (64, 220, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    detections: list[Detection] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500 or area > 12000:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.72:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        if not 0.8 < aspect < 1.25:
            continue

        box = (x, y, w, h)
        if _overlap_ratio(skin, box) > 0.08:
            continue

        roi_hsv = hsv[y : y + h, x : x + w]
        roi_mask = mask[y : y + h, x : x + w]
        green_fill = float(np.count_nonzero(roi_mask)) / max(roi_mask.size, 1)
        if green_fill < 0.45:
            continue

        hue_vals = roi_hsv[:, :, 0][roi_mask > 0]
        sat_vals = roi_hsv[:, :, 1][roi_mask > 0]
        if hue_vals.size < 40:
            continue
        if float(np.std(hue_vals)) > 8.0 or float(np.std(sat_vals)) > 55.0:
            continue

        (_, _), radius = cv2.minEnclosingCircle(contour)
        if not 14 <= radius <= 75:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1)
        if solidity < 0.9:
            continue

        score = min(
            0.99,
            0.55 + circularity * 0.25 + green_fill * 0.15 + solidity * 0.08,
        )
        detections.append(Detection("tennis ball", box, score))

    return detections


def detect_traffic_cones(frame: np.ndarray, skin: np.ndarray) -> list[Detection]:
    """Red tapered body with a white reflective band just above the red section."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    red = cv2.inRange(hsv, (0, 110, 70), (10, 255, 255))
    red |= cv2.inRange(hsv, (170, 110, 70), (180, 255, 255))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    white = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))

    detections: list[Detection] = []
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1500 or area > 90000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h < 45 or w < 15:
            continue
        aspect = h / max(w, 1)
        if aspect < 1.1:
            continue

        box = (x, y, w, h)
        if _overlap_ratio(skin, box) > 0.12:
            continue

        red_roi = red[y : y + h, x : x + w]
        red_fill = float(np.count_nonzero(red_roi)) / max(red_roi.size, 1)
        if red_fill < 0.25:
            continue

        # White reflector sits above the red body, not inside the red mask.
        stripe_h = max(int(h * 0.45), 18)
        stripe_y1 = max(y - stripe_h, 0)
        stripe_y2 = y + min(int(h * 0.12), 20)
        white_band = white[stripe_y1:stripe_y2, x : x + w]
        white_ratio = float(np.count_nonzero(white_band)) / max(white_band.size, 1)
        if white_ratio < 0.05:
            continue
        if white_ratio > 0.55:
            continue

        def row_width(mask_roi: np.ndarray, row: int) -> int:
            if row < 0 or row >= mask_roi.shape[0]:
                return 0
            cols = np.where(mask_roi[row, :] > 0)[0]
            return int(cols[-1] - cols[0]) if cols.size else 0

        top_w = row_width(red_roi, int(h * 0.15))
        bottom_w = row_width(red_roi, int(h * 0.8))
        if bottom_w <= 0 or top_w <= 0:
            continue
        if bottom_w < top_w * 1.04:
            continue

        full_y1 = stripe_y1
        full_y2 = y + h
        full_box = (x, full_y1, w, full_y2 - full_y1)
        if _overlap_ratio(skin, full_box) > 0.12:
            continue

        score = min(
            0.99,
            0.42
            + red_fill * 0.2
            + white_ratio * 0.28
            + min((bottom_w / top_w - 1.0), 0.8) * 0.12
            + min(aspect / 3.0, 0.3) * 0.1,
        )
        detections.append(Detection("traffic cone", full_box, score))

    return detections


def detect_hammers(frame: np.ndarray, skin: np.ndarray) -> list[Detection]:
    """Claw hammer: horizontal dark metal head attached to a narrow wooden handle."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    metal = cv2.inRange(hsv, (0, 0, 35), (180, 75, 150))
    wood = cv2.inRange(hsv, (5, 35, 35), (32, 220, 220))
    metal = cv2.morphologyEx(metal, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    wood = cv2.morphologyEx(wood, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    detections: list[Detection] = []
    wood_contours, _ = cv2.findContours(wood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for w_contour in wood_contours:
        w_area = cv2.contourArea(w_contour)
        if w_area < 180 or w_area > 8000:
            continue

        wx, wy, ww, wh = cv2.boundingRect(w_contour)
        if ww < 6 or ww > 34 or wh < 45:
            continue
        handle_aspect = wh / max(ww, 1)
        if handle_aspect < 2.2:
            continue

        head_y1 = max(wy - int(wh * 0.55), 0)
        head_y2 = wy + max(int(wh * 0.08), 6)
        head_x1 = max(wx - int(ww * 3.0), 0)
        head_x2 = min(wx + ww + int(ww * 3.0), frame.shape[1])
        head_region_metal = metal[head_y1:head_y2, head_x1:head_x2]
        if head_region_metal.size == 0:
            continue
        metal_fill = float(np.count_nonzero(head_region_metal)) / head_region_metal.size
        if metal_fill < 0.12:
            continue

        ys, xs = np.where(head_region_metal > 0)
        if xs.size < 40:
            continue
        hx = int(xs.min())
        hy = int(ys.min())
        hw = int(xs.max() - xs.min())
        hh = int(ys.max() - ys.min())
        if hw < 14 or hh < 5:
            continue
        head_aspect = hw / max(hh, 1)
        if head_aspect < 1.15:
            continue

        head_gray = gray[
            head_y1 + hy : head_y1 + hy + hh,
            head_x1 + hx : head_x1 + hx + hw,
        ]
        darkness = 1.0 - float(np.mean(head_gray)) / 255.0
        best_head_score = darkness * 0.5 + min(head_aspect / 4.0, 0.3) + metal_fill * 0.15
        if best_head_score < 0.42:
            continue
        best_head = (head_x1 + hx, head_y1 + hy, hw, hh)

        hx, hy, hw, hh = best_head
        if hw < ww * 1.1:
            continue

        cx_handle = wx + ww // 2
        cx_head = hx + hw // 2
        if abs(cx_handle - cx_head) > max(ww * 2.2, hw * 0.55):
            continue

        x1 = min(wx, hx)
        y1 = min(hy, wy)
        x2 = max(wx + ww, hx + hw)
        y2 = max(wy + wh, hy + hh)
        box = (x1, y1, x2 - x1, y2 - y1)

        if box[2] > 220 or box[3] > 220:
            continue
        if _overlap_ratio(skin, box) > 0.1:
            continue

        score = min(0.98, 0.5 + best_head_score * 0.35 + min(handle_aspect / 5.0, 0.2))
        detections.append(Detection("hammer", box, score))

    return detections


def detect_all(frame: np.ndarray) -> list[Detection]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    skin = _skin_mask(hsv)

    detections: list[Detection] = []
    detections.extend(detect_tennis_balls(frame, skin))
    detections.extend(detect_traffic_cones(frame, skin))
    detections.extend(detect_hammers(frame, skin))
    return _merge_overlapping(detections)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    colors = {
        "tennis ball": (120, 255, 120),
        "traffic cone": (60, 60, 255),
        "hammer": (255, 170, 60),
    }
    for det in detections:
        x, y, w, h = det.box
        color = colors.get(det.name, (0, 255, 0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{det.name} {det.confidence * 100:.0f}%"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(y - 8, th + 4)
        cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 4, ty + baseline), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    return frame


def generate():
    while True:
        frame = picam2.capture_array()
        detections = detect_all(frame)
        frame = draw_detections(frame, detections)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, jpeg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
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
        <title>Mars Rover Challenge</title>
        <style>
            body { font-family: sans-serif; text-align: center; background: #111; color: #eee; }
            h1 { margin-top: 1rem; }
            p { color: #aaa; max-width: 680px; margin: 0.5rem auto 1rem; line-height: 1.4; }
            img { border: 2px solid #333; max-width: 100%; }
        </style>
    </head>
    <body>
        <h1>Mars Rover Challenge</h1>
        <p>Strict detection: light green tennis ball, red cone with white reflector, claw hammer.</p>
        <img src="/video" width="640">
    </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    start_camera()
    app.run(host="0.0.0.0", port=PORT, threaded=True)
