#!/usr/bin/env python3
"""
Mars Rover Challenge — precise detection for competition props:
  - light green tennis ball
  - red traffic cone with white reflector band
  - claw hammer (metal head + narrow wooden handle)

Uses strict multi-cue OpenCV rules and rejects skin / furniture false positives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import MappedArray, Picamera2

PORT = 5001
FRAME_SIZE = (640, 480)
MIN_DISPLAY_CONFIDENCE = 0.68
MAX_PER_CLASS = 1
CONE_ONLY = True  # set False when tennis ball / hammer detectors are ready

app = Flask(__name__)


def start_camera():
    global picam2
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"size": FRAME_SIZE},
            buffer_count=12,
        )
    )
    picam2.start()
    print("Mars Rover detector running (strict OpenCV)")


def capture_rgb_frame() -> np.ndarray:
    """Capture one ISP-processed RGB frame (same pipeline quality as app.py)."""
    request = picam2.capture_request()
    try:
        with MappedArray(request, "main") as m:
            return m.array.copy()
    finally:
        request.release()


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
    """
    Red traffic cone with a white reflective band.

    Finds the white stripe in the centre column, then locates red cone body
    above (tip) and below (base), tolerating the gap between stripe and base.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    white = cv2.inRange(hsv, (0, 0, 140), (180, 75, 255))
    # Include orange-red hues — cone red spans H≈0-20 and H≈160-180
    red = cv2.inRange(hsv, (0, 35, 40), (20, 255, 255))
    red |= cv2.inRange(hsv, (160, 35, 40), (180, 255, 255))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    combined = red | white

    cx1, cx2 = int(w * 0.36), int(w * 0.64)
    col_w = cx2 - cx1
    row_white = np.count_nonzero(white[:, cx1:cx2], axis=1) / max(col_w, 1)
    row_red = np.count_nonzero(red[:, cx1:cx2], axis=1) / max(col_w, 1)

    scan_y1, scan_y2 = int(h * 0.12), int(h * 0.58)
    white_seg = row_white[scan_y1:scan_y2]
    if white_seg.size == 0 or float(np.max(white_seg)) < 0.35:
        return []

    stripe_y = scan_y1 + int(np.argmax(white_seg))
    band_ratio = float(row_white[stripe_y])

    top_lo = max(int(h * 0.08), stripe_y - 150)
    above = np.where(row_red[top_lo : stripe_y - 3] >= 0.10)[0]
    top = top_lo + int(above[0]) if above.size else max(stripe_y - 80, top_lo)

    below_lo = stripe_y + 5
    below_hi = min(h - 1, stripe_y + 185)
    below = np.where(row_red[below_lo:below_hi] >= 0.12)[0]
    if below.size:
        bottom = below_lo + int(below[-1])
    else:
        below_loose = np.where(row_red[below_lo:below_hi] >= 0.06)[0]
        bottom = below_lo + int(below_loose[-1]) if below_loose.size else stripe_y + 30

    if bottom - top < 45:
        return []

    sub = combined[top : bottom + 1, cx1:cx2]
    col_fill = np.count_nonzero(sub, axis=0) / max(sub.shape[0], 1)
    cols = np.where(col_fill >= 0.08)[0]
    if cols.size < 4:
        return []

    pad = 10
    x1 = max(cx1 + int(cols[0]) - pad, 0)
    x2 = min(cx1 + int(cols[-1]) + pad, w)
    y1 = max(top - pad, 0)
    y2 = min(bottom + pad, h)
    bw, bh = x2 - x1, y2 - y1
    if bw < 25 or bh < 50:
        return []

    box = (x1, y1, bw, bh)

    body = red[y1:y2, x1:x2]
    body_fill = float(np.count_nonzero(body)) / max(body.size, 1)
    aspect = bh / max(bw, 1)
    centre_score = 1.0 - abs((x1 + bw / 2) - w / 2) / (w * 0.25)
    score = min(
        0.99,
        0.38 + body_fill * 0.22 + band_ratio * 0.22 + centre_score * 0.10 + min(aspect, 2.0) * 0.08,
    )
    return [Detection("traffic cone", box, score)]


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
    detections.extend(detect_traffic_cones(frame, skin))
    if not CONE_ONLY:
        detections.extend(detect_tennis_balls(frame, skin))
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
        frame = capture_rgb_frame()
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
        <p>Traffic cone detection: red cone with white reflector band.</p>
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
