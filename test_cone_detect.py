#!/usr/bin/env python3
"""Iterate cone detection algorithms against captured frame."""

import math
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/pi/Desktop/mars")

FRAME_PATH = "/home/pi/Desktop/mars/debug/frame.jpg"
OUT = "/home/pi/Desktop/mars/debug"


def load_frame():
    bgr = cv2.imread(FRAME_PATH)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return rgb, hsv


def detect_cone_v2(frame: np.ndarray, hsv: np.ndarray) -> list[tuple]:
    """
    Stripe-anchored cone detection for blue/purple cones with white reflector band.
    """
    h, w = frame.shape[:2]

    # Purple/blue cone body (competition cone is blue-violet, not orange)
    cone_color = cv2.inRange(hsv, (95, 45, 40), (135, 255, 220))
    # White / light reflector stripe
    white = cv2.inRange(hsv, (0, 0, 140), (180, 80, 255))

    cone_color = cv2.morphologyEx(cone_color, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cone_color = cv2.morphologyEx(cone_color, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    detections = []
    contours, _ = cv2.findContours(cone_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1200 or area > h * w * 0.55:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        if bh < 40 or bw < 20:
            continue

        cx = x + bw // 2
        cy = y + bh // 2
        # Cone should be in lower 75% of frame, roughly centered horizontally
        if cy < h * 0.15:
            continue
        if cx < w * 0.12 or cx > w * 0.88:
            continue

        aspect = bh / max(bw, 1)
        if aspect < 0.45:  # reject flat table-like blobs
            continue

        roi_cone = cone_color[y : y + bh, x : x + bw]
        color_fill = float(np.count_nonzero(roi_cone)) / max(roi_cone.size, 1)
        if color_fill < 0.12:
            continue

        # White stripe: search upper 55% of bounding box
        stripe_y1 = y
        stripe_y2 = y + int(bh * 0.55)
        stripe = white[stripe_y1:stripe_y2, x : x + bw]
        white_ratio = float(np.count_nonzero(stripe)) / max(stripe.size, 1)

        # Also accept bright low-saturation band (overexposed reflector)
        bright = cv2.inRange(hsv[stripe_y1:stripe_y2, x : x + bw], (0, 0, 170), (180, 100, 255))
        bright_ratio = float(np.count_nonzero(bright)) / max(bright.size, 1)
        band_ratio = max(white_ratio, bright_ratio)
        if band_ratio < 0.04:
            continue

        def row_width(mask_roi, row):
            if row < 0 or row >= mask_roi.shape[0]:
                return 0
            cols = np.where(mask_roi[row, :] > 0)[0]
            return int(cols[-1] - cols[0]) if cols.size >= 2 else 0

        top_w = row_width(roi_cone, int(bh * 0.1))
        mid_w = row_width(roi_cone, int(bh * 0.5))
        bot_w = row_width(roi_cone, int(bh * 0.85))
        if bot_w <= 0:
            continue
        # Cone tapers: bottom wider than top (relaxed for partial occlusion)
        taper = bot_w / max(top_w, 1)
        if taper < 0.95 and top_w > 5:
            continue

        # Expand box upward to include stripe above color mask
        pad_top = max(int(bh * 0.08), 4)
        box = (x, max(y - pad_top, 0), bw, min(y + bh, h) - max(y - pad_top, 0))

        score = min(
            0.99,
            0.35 + color_fill * 0.25 + band_ratio * 0.25 + min(taper / 3, 0.15) + min(aspect, 1.5) * 0.08,
        )
        detections.append((box, score, taper, band_ratio, color_fill, aspect))

    detections.sort(key=lambda d: d[1], reverse=True)
    return detections


def detect_cone_v3(frame: np.ndarray, hsv: np.ndarray) -> list[tuple]:
    """Find white horizontal bands, grow to full cone."""
    h, w = frame.shape[:2]
    white = cv2.inRange(hsv, (0, 0, 150), (180, 90, 255))
    cone_color = cv2.inRange(hsv, (90, 40, 35), (140, 255, 230))

    detections = []
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200 or area > 25000:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 30 or bh > bw * 0.45:  # want horizontal stripe
            continue
        if y < h * 0.05 or y > h * 0.75:
            continue

        # Grow downward and upward to include cone body
        x1 = max(x - int(bw * 0.3), 0)
        x2 = min(x + bw + int(bw * 0.3), w)
        y1 = max(y - int(bh * 4), 0)
        y2 = min(y + int(bh * 5), h)

        region = cone_color[y1:y2, x1:x2]
        body_fill = float(np.count_nonzero(region)) / max(region.size, 1)
        if body_fill < 0.06:
            continue

        full_w = x2 - x1
        full_h = y2 - y1
        aspect = full_h / max(full_w, 1)
        if aspect < 0.5:
            continue

        cx = (x1 + x2) // 2
        if cx < w * 0.1 or cx > w * 0.9:
            continue

        score = min(0.99, 0.4 + body_fill * 0.3 + min(aspect, 2.0) * 0.12 + min(bw / 80, 0.2))
        detections.append(((x1, y1, full_w, full_h), score, body_fill, aspect))

    detections.sort(key=lambda d: d[1], reverse=True)
    return detections


def main():
    frame, hsv = load_frame()
    h, w = frame.shape[:2]

    for name, fn in [("v2_color_first", detect_cone_v2), ("v3_stripe_first", detect_cone_v3)]:
        dets = fn(frame, hsv)
        vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print(f"\n=== {name}: {len(dets)} detections ===")
        for i, det in enumerate(dets[:5]):
            box, score, *rest = det
            x, y, bw, bh = box
            print(f"  #{i} score={score:.3f} box=({x},{y},{bw},{bh}) extra={rest}")
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
            cv2.putText(vis, f"{score:.2f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(f"{OUT}/result_{name}.jpg", vis)

    # Expected cone roughly centered: x~200-440, y~80-380
    print("\n=== Ground truth region (approx): center box 180,70,280,320 ===")


if __name__ == "__main__":
    main()
