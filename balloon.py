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
import time
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
INFERENCE_SIZE = 480  # slightly larger helps distant/small balloons
SHAPE_CONFIDENCE = 0.01  # models score ~0.01–0.03 on real balloons
MIN_COLOR_VOTE = 0.35
MIN_WHITE_VOTE = 0.45  # white balloons are low-saturation — separate threshold
MIN_RGB_WHITE_RATIO = 0.42  # reject white-ish walls / faces that aren't balanced RGB
MIN_PALE_WHITE_RATIO = 0.40  # bright pale surface (white balloon under blue room lights)
MIN_ROUNDNESS = 0.58
MIN_SATURATION_VOTE = 50  # ignore white glare when classifying vivid colours
MAX_WHITE_SATURATION = 100
MIN_WHITE_VALUE = 140
MIN_RGB_WHITE_SUM = 360
RGB_WHITE_BALANCE = 50
MIN_BLOB_FILL = 0.38  # balloon surface fills the box; stool legs do not
MAX_BLOB_PARTS = 2  # legs / clutter split into many blobs
EDGE_MARGIN = 28  # reject partial objects / whiteboard strips at frame edge
MIN_HOUGH_ROUNDNESS = 0.52  # grayscale circle threshold (separate from colour blob)
MIN_HOUGH_FILL = 0.22  # ignore tiny circles in the corner of a box
JPEG_QUALITY = 75
PORT = 5003

# One shape model is enough — HSV names the colour for any balloon.
SHAPE_MODEL = "light_blue_balloon.pt"

# Accept balloons anywhere in frame (ground, close-up testing, etc.).
ROI_MARGIN = 8
MIN_BOX_AREA = 350  # allow smaller distant balloons
MAX_BOX_AREA = 80000
MIN_ROUND_ASPECT = 0.72  # balloons are round — reject tall/wide jacket panels
MAX_ROUND_ASPECT = 1.38
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
        "label": "black balloon",  # competition navy-blue balloon
        "box_color": (180, 80, 40),
        "text_color": (20, 20, 20),
    },
    "white": {
        "label": "white balloon",
        "box_color": (220, 220, 220),
        "text_color": (20, 20, 20),
    },
}

app = Flask(__name__)
picam2: Picamera2 | None = None
shape_model: YOLO | None = None
_running = True
_server = None
_frame_lock = threading.Lock()
_detection_lock = threading.Lock()
_pending_frame = None
_latest_detections: list[tuple[str, int, int, int, int, float]] = []


def load_models() -> None:
    global shape_model

    if not MODELS_DIR.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODELS_DIR}")

    model_path = MODELS_DIR / SHAPE_MODEL
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading shape model {SHAPE_MODEL}...")
    shape_model = YOLO(str(model_path))
    print("Hybrid detection ready — YOLO shape + HSV color (async)")


def start_camera() -> None:
    global picam2

    print("Starting IMX500 camera...")
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"size": FRAME_SIZE},
            buffer_count=4,
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


def _box_clipped_by_edge(
    x1: int, y1: int, x2: int, y2: int, frame_width: int, frame_height: int
) -> bool:
    """Partial objects at the frame edge are usually background, not the held balloon."""
    return (
        x1 <= EDGE_MARGIN
        or y1 <= EDGE_MARGIN
        or x2 >= frame_width - EDGE_MARGIN
        or y2 >= frame_height - EDGE_MARGIN
    )


def _box_valid(x1: int, y1: int, x2: int, y2: int, frame_width: int, frame_height: int) -> bool:
    if _box_clipped_by_edge(x1, y1, x2, y2, frame_width, frame_height):
        return False

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
    if min(w, h) < 28:  # stool-leg slivers are tall/thin but tiny
        return False
    aspect = w / h if w >= h else h / w
    return MIN_ROUND_ASPECT <= aspect <= MAX_ROUND_ASPECT


def _color_masks(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    # Bright cyan competition balloon.
    light_blue = (h >= 82) & (h <= 102) & (s >= 120) & (v >= 160)
    # Navy competition balloon (labelled "black") — dark blue, including dull cyan-navy.
    navy_hue = (h >= 103) & (h <= 128) & (s >= 55) & (v >= 90) & (v <= 200)
    dark_cyan = (h >= 82) & (h <= 102) & (s >= 40) & (s <= 130) & (v >= 70) & (v <= 170)
    return {
        "yellow": (h >= 22) & (h <= 40) & (s >= 80) & (v >= 100),
        "pink": (h >= 135) & (h <= 175) & (s >= 40) & (v >= 100),
        "black": navy_hue | dark_cyan,
        "light_blue": light_blue,
        "white": (v >= MIN_WHITE_VALUE) & (s <= MAX_WHITE_SATURATION),
    }


def _pale_balloon_mask(
    h: np.ndarray,
    s: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Bright pale surface — white balloon under coloured lighting looks cyan, not RGB-white."""
    light_blue = (h >= 82) & (h <= 102) & (s >= 120) & (v >= 160)
    return (v >= 155) & (s <= 115) & ~light_blue


def _rgb_white_mask(
    b: np.ndarray,
    g: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Bright, balanced RGB — catches white balloons under coloured room lighting."""
    brightness = r.astype(np.int16) + g.astype(np.int16) + b.astype(np.int16)
    balance = (
        (np.abs(r.astype(np.int16) - g) <= RGB_WHITE_BALANCE)
        & (np.abs(g.astype(np.int16) - b) <= RGB_WHITE_BALANCE)
    )
    return (
        (brightness >= MIN_RGB_WHITE_SUM)
        & balance
        & (r >= 110)
        & (g >= 110)
        & (b >= 110)
    )


def _rgb_white_confidence(bgr_patch: np.ndarray) -> float:
    b, g, r = cv2.split(bgr_patch)
    mask = _rgb_white_mask(b, g, r)
    total = bgr_patch.shape[0] * bgr_patch.shape[1]
    return float(mask.sum()) / max(total, 1)


def _white_color_confidence(hsv_patch: np.ndarray, bgr_patch: np.ndarray) -> float:
    """White balloon — pale + bright in HSV; RGB balance is optional (room lights tint cyan)."""
    pixels = hsv_patch.reshape(-1, 3)
    if len(pixels) < 20:
        return 0.0

    h, s, v = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    hsv_score = float(_pale_balloon_mask(h, s, v).sum()) / len(pixels)
    rgb_score = _rgb_white_confidence(bgr_patch)

    if hsv_score >= MIN_PALE_WHITE_RATIO:
        return min(0.99, hsv_score * 0.82 + rgb_score * 0.18)
    if rgb_score >= MIN_RGB_WHITE_RATIO:
        return hsv_score * 0.40 + rgb_score * 0.60
    return hsv_score * 0.35


def _contour_roundness(contour) -> float:
    area = cv2.contourArea(contour)
    if area < 80:
        return 0.0

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0

    circularity = 4 * math.pi * area / (perimeter * perimeter)
    _, (rect_w, rect_h), _ = cv2.minAreaRect(contour)
    if rect_w <= 0 or rect_h <= 0:
        return 0.0

    rect_aspect = min(rect_w, rect_h) / max(rect_w, rect_h)
    _, _, bw, bh = cv2.boundingRect(contour)
    extent = area / (bw * bh) if bw * bh else 0.0
    extent_score = min(extent / 0.785, 1.0)
    return circularity * 0.45 + rect_aspect * 0.35 + extent_score * 0.20


def _roundness_in_box(
    hsv: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color_key: str,
) -> float:
    """Measure how round the coloured blob inside a box is (0 = jacket, 1 = balloon)."""
    h, w = hsv.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    patch = hsv[y1:y2, x1:x2]
    px_h = patch[:, :, 0]
    px_s = patch[:, :, 1]
    px_v = patch[:, :, 2]
    mask = _color_masks(px_h, px_s, px_v)[color_key].astype(np.uint8) * 255
    if not mask.any():
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    return _contour_roundness(max(contours, key=cv2.contourArea))


def _hough_roundness(
    frame_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> float:
    """Grayscale circle fit — works for navy balloons with white glare."""
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    patch = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    ph, pw = gray.shape
    if ph < 40 or pw < 40:
        return 0.0

    min_r = int(min(pw, ph) * 0.22)
    max_r = int(min(pw, ph) * 0.56)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(pw, ph) // 2,
        param1=80,
        param2=25,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return 0.0

    _, _, radius = circles[0][0]
    fill = (math.pi * radius * radius) / max(pw * ph, 1)
    if fill < MIN_HOUGH_FILL:
        return 0.0
    radius_ratio = radius / (min(pw, ph) / 2)
    return min(1.0, radius_ratio * 0.85 + (fill / 0.65) * 0.15)


def _analyze_balloon_shape(
    hsv: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color_key: str,
    frame_bgr: np.ndarray | None = None,
) -> tuple[bool, float]:
    """True when the coloured region is one round blob (balloon), not legs/clutter."""
    h, w = hsv.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    patch = hsv[y1:y2, x1:x2]
    px_h = patch[:, :, 0]
    px_s = patch[:, :, 1]
    px_v = patch[:, :, 2]
    color_mask = _color_masks(px_h, px_s, px_v)[color_key]
    if color_key == "white":
        color_mask = color_mask | _pale_balloon_mask(px_h, px_s, px_v)
        if frame_bgr is not None:
            patch_bgr = frame_bgr[y1:y2, x1:x2]
            pb, pg, pr = cv2.split(patch_bgr)
            color_mask = color_mask | _rgb_white_mask(pb, pg, pr)
    mask = color_mask.astype(np.uint8) * 255
    if not mask.any():
        return False, 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parts = [c for c in contours if cv2.contourArea(c) >= 120]
    if not parts:
        return False, 0.0
    if len(parts) > MAX_BLOB_PARTS:
        return False, 0.0

    main = max(parts, key=cv2.contourArea)
    roundness = _contour_roundness(main)
    if roundness < MIN_ROUNDNESS:
        return False, roundness

    box_area = max((x2 - x1) * (y2 - y1), 1)
    blob_fill = cv2.contourArea(main) / box_area
    min_fill = 0.28 if color_key == "white" else MIN_BLOB_FILL
    if blob_fill < min_fill:
        if frame_bgr is not None and box_area >= 8000:
            hough = _hough_roundness(frame_bgr, x1, y1, x2, y2)
            if hough >= MIN_HOUGH_ROUNDNESS:
                return True, hough
        return False, roundness

    return True, roundness


def _analyze_balloon_shape_from_color_or_hough(
    hsv: np.ndarray,
    frame_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color_key: str,
) -> tuple[bool, float]:
    ok, roundness = _analyze_balloon_shape(
        hsv, x1, y1, x2, y2, color_key, frame_bgr=frame_bgr
    )
    if ok:
        return ok, roundness

    area = _box_area(x1, y1, x2, y2)
    if area < 8000:
        return False, roundness

    hough = _hough_roundness(frame_bgr, x1, y1, x2, y2)
    if hough >= MIN_HOUGH_ROUNDNESS:
        return True, hough
    return False, max(roundness, hough)


def _is_round_balloon(
    hsv: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color_key: str,
) -> bool:
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    aspect = w / h if w >= h else h / w
    if aspect < MIN_ROUND_ASPECT or aspect > MAX_ROUND_ASPECT:
        return False
    return _roundness_in_box(hsv, x1, y1, x2, y2, color_key) >= MIN_ROUNDNESS


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


def _classify_balloon_color(
    hsv: np.ndarray,
    bgr: np.ndarray,
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

    patch_hsv = hsv[y1:y2, x1:x2]
    patch_bgr = bgr[y1:y2, x1:x2]
    margin_x = max(1, int((x2 - x1) * 0.15))
    margin_y = max(1, int((y2 - y1) * 0.15))
    center_hsv = patch_hsv[margin_y:-margin_y or None, margin_x:-margin_x or None]
    center_bgr = patch_bgr[margin_y:-margin_y or None, margin_x:-margin_x or None]
    if center_hsv.size == 0:
        center_hsv = patch_hsv
        center_bgr = patch_bgr

    white_conf = _white_color_confidence(center_hsv, center_bgr)

    pixels = center_hsv.reshape(-1, 3)
    saturated = pixels[pixels[:, 1] >= MIN_SATURATION_VOTE]
    best_vivid_key: str | None = None
    vivid_conf = 0.0
    if len(saturated) >= 40:
        if len(saturated) > 2500:
            saturated = saturated[:: max(1, len(saturated) // 2500)]
        px_h = saturated[:, 0]
        px_s = saturated[:, 1]
        px_v = saturated[:, 2]
        vivid_masks = {
            name: mask
            for name, mask in _color_masks(px_h, px_s, px_v).items()
            if name != "white"
        }
        vivid_votes = {
            name: int(mask.sum()) for name, mask in vivid_masks.items() if mask.any()
        }
        if vivid_votes:
            best_vivid_key = max(vivid_votes, key=vivid_votes.get)
            vivid_conf = vivid_votes[best_vivid_key] / max(len(saturated), 1)

    if white_conf >= MIN_WHITE_VOTE:
        if best_vivid_key is None or white_conf >= vivid_conf * 0.85:
            return "white", white_conf

    if best_vivid_key is not None and vivid_conf >= MIN_COLOR_VOTE:
        return best_vivid_key, vivid_conf

    if white_conf >= MIN_WHITE_VOTE * 0.85:
        return "white", white_conf

    return None, 0.0


def _prominence_score(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color_conf: float,
    roundness: float,
    frame_width: int,
    frame_height: int,
) -> float:
    """Prefer large, centred, round, colour-certain detections — the balloon in front."""
    area = _box_area(x1, y1, x2, y2)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    focus_x = frame_width / 2
    focus_y = frame_height * 0.55
    distance = math.hypot(cx - focus_x, cy - focus_y)
    max_distance = math.hypot(frame_width, frame_height)
    centrality = 1.0 - distance / max_distance
    size_score = min(1.0, area / 8000)
    score = (
        color_conf * 0.25
        + size_score * 0.40
        + centrality * 0.25
        + roundness * 0.10
    )
    # Strong bonus for a large balloon filling the frame (held in front).
    if area >= 20000:
        score += 0.12
    # Ignore wall / ceiling colour blobs in the upper frame.
    if cy < frame_height * 0.40:
        score *= 0.25
    # Ignore small detections tucked in the bottom-right (stool legs).
    if cx > frame_width * 0.62 and cy > frame_height * 0.72 and area < 18000:
        score *= 0.15
    return score


def _adjust_white_score(
    score: float,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    roundness: float,
    frame_width: int,
    frame_height: int,
) -> float:
    """Score the white balloon on the table — not whiteboard, cone stripe, or laptop glare."""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    area = _box_area(x1, y1, x2, y2)
    nx, ny = cx / frame_width, cy / frame_height

    # Whiteboard / ceiling at the top of the frame.
    if ny < 0.32:
        score *= 0.30
    # Cone reflective stripe — small, far-right sliver (not the balloon beside it).
    elif nx > 0.80 and area < 8000:
        score *= 0.35
    # Large white wall / whiteboard beside the cone on the right.
    elif nx > 0.72 and ny < 0.52 and area > 12000:
        score *= 0.30
    # Large whiteboard spans (not a single balloon).
    elif ny < 0.48 and area > 25000:
        score *= 0.25
    # Laptop / face glare on the left.
    elif nx < 0.38 and ny < 0.50 and area < 8000:
        score *= 0.35
    elif nx < 0.32 and ny < 0.55:
        score *= 0.35
    # Small white clutter in the frame corners.
    elif area < 5000 and (nx < 0.18 or nx > 0.82):
        score *= 0.30
    # White balloon on the table (next to the traffic cone).
    elif (
        0.36 <= nx <= 0.78
        and 0.36 <= ny <= 0.58
        and 2000 <= area <= 20000
        and roundness >= 0.55
    ):
        score += 0.20

    return score


def _pick_best_candidate(
    candidates: list[tuple[str, str, int, int, int, int, float, float, float, int]],
) -> tuple[str, int, int, int, int, float]:
    """Choose the balloon in front — not background balloons or navy clothing."""
    vivid = {"light_blue", "yellow", "pink", "white"}
    ranked = sorted(candidates, key=lambda item: item[7], reverse=True)
    best = ranked[0]
    best_key = best[1]
    best_score = best[7]
    best_area = best[9]

    if best_key == "white":
        for alt in ranked[1:]:
            if (
                alt[1] == "black"
                and alt[9] >= 20000
                and alt[7] >= best_score - 0.15
                and alt[7] >= best_score * 0.85
            ):
                best = alt
                best_key = alt[1]
                best_score = alt[7]
                best_area = alt[9]
                break

        if best_key == "white":
            for alt in ranked[1:]:
                if (
                    alt[1] in vivid
                    and alt[1] != "white"
                    and alt[8] >= 0.80
                    and alt[7] >= 0.70
                    and alt[7] >= best_score * 0.72
                ):
                    best = alt
                    break

    for alt in ranked[1:]:
        alt_key = alt[1]
        alt_score = alt[7]
        alt_conf = alt[8]
        alt_area = alt[9]

        # Large held navy balloon beats a small background balloon.
        if (
            best_key in vivid
            and alt_key == "black"
            and alt_area >= 20000
            and best_area < 8000
            and alt_score >= best_score - 0.20
        ):
            best = alt
            best_key = alt_key
            best_score = alt_score
            best_area = alt_area
            continue

        # Vivid balloon in hand beats ambiguous navy clothing (not a large held balloon).
        if (
            best_key == "black"
            and alt_key in vivid
            and best_area < 20000
            and best[8] < 0.62
            and alt_conf >= 0.80
            and alt_score >= 0.70
            and alt_score >= best_score * 0.72
        ):
            best = alt
            break

    _label, _key, x1, y1, x2, y2, display_conf, _score, _conf, _area = best
    return best[0], x1, y1, x2, y2, display_conf


def _find_round_color_blobs(
    hsv: np.ndarray,
    frame_bgr: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> list[tuple[int, int, int, int, float]]:
    """Find round colour blobs YOLO may miss at distance."""
    boxes: list[tuple[int, int, int, int, float]] = []
    px_h = hsv[:, :, 0]
    px_s = hsv[:, :, 1]
    px_v = hsv[:, :, 2]

    for color_key, mask in _color_masks(px_h, px_s, px_v).items():
        binary = mask.astype(np.uint8) * 255
        ksize = 3 if color_key == "white" else 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            x2, y2 = x + bw, y + bh
            if not _box_valid(x, y, x2, y2, frame_width, frame_height):
                continue

            ok, roundness = _analyze_balloon_shape_from_color_or_hough(
                hsv, frame_bgr, x, y, x2, y2, color_key
            )
            if not ok:
                continue

            boxes.append((x, y, x2, y2, roundness * 0.04))

    return boxes


def _hough_white_in_roi(
    frame_bgr: np.ndarray,
    hsv: np.ndarray,
    rx1: int,
    ry1: int,
    rx2: int,
    ry2: int,
    frame_width: int,
    frame_height: int,
) -> list[tuple[int, int, int, int, float]]:
    """Fast circle search in a cropped region — full-frame Hough takes ~3s on Pi."""
    rx1 = max(0, min(rx1, frame_width - 1))
    ry1 = max(0, min(ry1, frame_height - 1))
    rx2 = max(rx1 + 1, min(rx2, frame_width))
    ry2 = max(ry1 + 1, min(ry2, frame_height))

    roi = frame_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    rh, rw = gray.shape
    if rh < 40 or rw < 40:
        return []

    boxes: list[tuple[int, int, int, int, float]] = []
    seen: list[tuple[int, int, int, int]] = []

    for min_r, max_r in ((12, 55), (25, 90)):
        max_r = min(max_r, min(rw, rh) // 2)
        if max_r <= min_r:
            continue
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(min_r * 2, 30),
            param1=80,
            param2=28,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is None:
            continue

        for cx, cy, radius in circles[0]:
            cx_f = float(cx) + rx1
            cy_f = float(cy) + ry1
            radius_f = float(radius)
            x1 = int(cx_f - radius_f * 1.08)
            y1 = int(cy_f - radius_f * 1.08)
            x2 = int(cx_f + radius_f * 1.08)
            y2 = int(cy_f + radius_f * 1.08)
            if not _box_valid(x1, y1, x2, y2, frame_width, frame_height):
                continue
            if any(_iou((x1, y1, x2, y2), kept) > 0.35 for kept in seen):
                continue

            color_key, color_conf = _classify_balloon_color(
                hsv, frame_bgr, x1, y1, x2, y2
            )
            if color_key != "white" or color_conf < MIN_WHITE_VOTE:
                continue
            ok, roundness = _analyze_balloon_shape_from_color_or_hough(
                hsv, frame_bgr, x1, y1, x2, y2, "white"
            )
            if not ok:
                continue

            seen.append((x1, y1, x2, y2))
            boxes.append((x1, y1, x2, y2, roundness * 0.04))

    return boxes


def _find_hough_balloon_boxes(
    frame_bgr: np.ndarray,
    hsv: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> list[tuple[int, int, int, int, float]]:
    """White balloon beside the traffic cone — search table zone only (not full frame)."""
    tx1 = int(frame_width * 0.34)
    ty1 = int(frame_height * 0.24)
    tx2 = int(frame_width * 0.84)
    ty2 = int(frame_height * 0.60)
    return _hough_white_in_roi(
        frame_bgr, hsv, tx1, ty1, tx2, ty2, frame_width, frame_height
    )


def _find_balloon_boxes(
    frame_bgr,
) -> list[tuple[int, int, int, int, float]]:
    height, width = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    boxes: list[tuple[int, int, int, int, float]] = []

    results = shape_model.predict(
        frame_bgr,
        imgsz=INFERENCE_SIZE,
        conf=SHAPE_CONFIDENCE,
        verbose=False,
        device="cpu",
    )[0]

    if results.boxes is not None:
        for box in results.boxes:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if _box_valid(x1, y1, x2, y2, width, height):
                boxes.append((x1, y1, x2, y2, confidence))

    boxes.extend(_find_round_color_blobs(hsv, frame_bgr, width, height))
    boxes.extend(_find_hough_balloon_boxes(frame_bgr, hsv, width, height))

    deduped: list[tuple[int, int, int, int, float]] = []
    for box in sorted(boxes, key=lambda item: item[4], reverse=True):
        x1, y1, x2, y2, conf = box
        if any(_iou((x1, y1, x2, y2), kept[:4]) > NMS_IOU for kept in deduped):
            continue
        deduped.append(box)
    return deduped


def detect_objects(frame_rgb) -> list[tuple[str, int, int, int, int, float]]:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    frame_height, frame_width = frame_bgr.shape[:2]
    balloon_boxes = _find_balloon_boxes(frame_bgr)

    candidates: list[tuple[str, str, int, int, int, int, float, float, float, int]] = []
    for x1, y1, x2, y2, _shape_conf in balloon_boxes:
        color_key, color_conf = _classify_balloon_color(
            hsv, frame_bgr, x1, y1, x2, y2
        )
        if color_key is None or color_conf < MIN_COLOR_VOTE:
            continue
        if color_key == "white" and color_conf < MIN_WHITE_VOTE:
            continue
        # Navy/black needs clearer colour proof than other balloons.
        if color_key == "black" and color_conf < 0.45:
            continue

        ok, roundness = _analyze_balloon_shape_from_color_or_hough(
            hsv, frame_bgr, x1, y1, x2, y2, color_key
        )
        if not ok:
            continue

        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        aspect = w / h if w >= h else h / w
        if aspect < MIN_ROUND_ASPECT or aspect > MAX_ROUND_ASPECT:
            continue

        label = BALLOON_COLORS[color_key]["label"]
        score = _prominence_score(
            x1, y1, x2, y2, color_conf, roundness, frame_width, frame_height
        )
        if color_key == "white":
            score = _adjust_white_score(
                score, x1, y1, x2, y2, roundness, frame_width, frame_height
            )
        display_conf = min(0.99, score)
        if display_conf < 0.55:
            continue
        area = _box_area(x1, y1, x2, y2)
        candidates.append(
            (label, color_key, x1, y1, x2, y2, display_conf, score, color_conf, area)
        )

    if not candidates:
        return []

    label, x1, y1, x2, y2, confidence = _pick_best_candidate(candidates)
    return [(label, x1, y1, x2, y2, confidence)]


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


def _detection_worker() -> None:
    """Run YOLO on the latest frame without blocking the video stream."""
    global _pending_frame, _latest_detections

    while _running:
        frame = None
        with _frame_lock:
            while _pending_frame is not None:
                frame = _pending_frame
                _pending_frame = None

        if frame is None:
            time.sleep(0.002)
            continue

        detections = detect_objects(frame)
        with _detection_lock:
            _latest_detections = detections


def generate():
    global _pending_frame

    while _running:
        if picam2 is None:
            break

        frame = capture_rgb_frame()

        with _frame_lock:
            _pending_frame = frame.copy()

        with _detection_lock:
            detections = list(_latest_detections)

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
        threading.Thread(target=_detection_worker, daemon=True).start()
        print(f"Balloon detection ready — stream: http://0.0.0.0:{PORT}")
        print("Press Ctrl+C to stop, or click 'Stop detection' in the browser.")
        _server = make_server("0.0.0.0", PORT, app, threaded=True)
        _server.serve_forever()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        shutdown()
        sys.exit(1)
