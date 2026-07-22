#!/usr/bin/env python3
"""Refine cone box by trimming empty rows."""

import cv2
import numpy as np

bgr = cv2.imread("/home/pi/Desktop/mars/debug/frame.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

white = cv2.inRange(hsv, (0, 0, 150), (180, 65, 255))
purple = cv2.inRange(hsv, (90, 35, 30), (125, 255, 185))
purple = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
# Include white in cone silhouette
cone_mask = purple | white


def trim_box_to_content(mask, x1, y1, x2, y2, min_row_fill=0.02):
    """Shrink box to rows/cols that actually contain cone pixels."""
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return x1, y1, x2 - x1, y2 - y1

    row_fill = np.count_nonzero(roi, axis=1) / max(roi.shape[1], 1)
    col_fill = np.count_nonzero(roi, axis=0) / max(roi.shape[0], 1)

    rows = np.where(row_fill >= min_row_fill)[0]
    cols = np.where(col_fill >= min_row_fill)[0]
    if rows.size == 0 or cols.size == 0:
        return x1, y1, x2 - x1, y2 - y1

    ty, by = int(rows[0]), int(rows[-1])
    lx, rx = int(cols[0]), int(cols[-1])
    pad = 8
    nx1 = max(x1 + lx - pad, 0)
    ny1 = max(y1 + ty - pad, 0)
    nx2 = min(x1 + rx + pad, w)
    ny2 = min(y1 + by + pad, h)
    return nx1, ny1, nx2 - nx1, ny2 - ny1


def detect_traffic_cone(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    white = cv2.inRange(hsv, (0, 0, 150), (180, 65, 255))
    purple = cv2.inRange(hsv, (90, 35, 30), (125, 255, 185))
    purple = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cone_mask = purple | white

    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 400 or area > 8000:
            continue
        sx, sy, sw, sh = cv2.boundingRect(c)
        cx = sx + sw // 2
        if cx < w * 0.25 or cx > w * 0.72:
            continue
        if sy < h * 0.12 or sy > h * 0.65:
            continue
        if sw < 25:
            continue

        pad_x = max(int(sw * 1.2), 40)
        pad_up = max(int(sh * 5), 80)
        pad_down = max(int(sh * 4), 60)
        x1 = max(sx - pad_x, 0)
        x2 = min(sx + sw + pad_x, w)
        y1 = max(sy - pad_up, 0)
        y2 = min(sy + sh + pad_down, h)

        body = purple[y1:y2, x1:x2]
        body_fill = float(np.count_nonzero(body)) / max(body.size, 1)
        if body_fill < 0.05:
            continue

        # Trim to actual cone content
        tx, ty, tw, th = trim_box_to_content(cone_mask, x1, y1, x2, y2, min_row_fill=0.03)
        if th < 50 or tw < 25:
            continue

        aspect = th / max(tw, 1)
        centre_score = 1.0 - abs((tx + tw / 2) - w / 2) / (w * 0.25)
        score = min(0.99, 0.45 + body_fill * 0.25 + centre_score * 0.2 + min(aspect, 2.0) * 0.08)
        detections.append(((tx, ty, tw, th), score))

    detections.sort(key=lambda d: d[1], reverse=True)
    return detections


dets = detect_traffic_cone(rgb)
vis = bgr.copy()
for box, score in dets[:1]:
    x, y, bw, bh = box
    print(f"score={score:.3f} box=({x},{y},{bw},{bh})")
    cv2.rectangle(vis, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
    cv2.putText(vis, f"traffic cone {score*100:.0f}%", (x, y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

cv2.imwrite("/home/pi/Desktop/mars/debug/result_v5.jpg", vis)

expected = (180, 70, 280, 320)
if dets:
    bx, by, bw, bh = dets[0][0]
    ex, ey, ew, eh = expected
    x1, y1 = max(ex, bx), max(ey, by)
    x2, y2 = min(ex+ew, bx+bw), min(ey+eh, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    iou = inter / (ew*eh + bw*bh - inter)
    print(f"IoU={iou:.3f}")
