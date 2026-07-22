#!/usr/bin/env python3
"""Stripe-anchored cone detector v4."""

import cv2
import numpy as np

bgr = cv2.imread("/home/pi/Desktop/mars/debug/frame.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

white = cv2.inRange(hsv, (0, 0, 150), (180, 65, 255))
purple = cv2.inRange(hsv, (90, 35, 30), (125, 255, 185))
purple = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

detections = []
for c in contours:
    area = cv2.contourArea(c)
    if area < 400 or area > 8000:
        continue
    sx, sy, sw, sh = cv2.boundingRect(c)
    cx = sx + sw // 2

    # Stripe must be in central area, not floor/table edge
    if cx < w * 0.25 or cx > w * 0.72:
        continue
    if sy < h * 0.12 or sy > h * 0.65:
        continue
    if sw < 25:
        continue

    # Grow cone box around stripe
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

    bw, bh = x2 - x1, y2 - y1
    aspect = bh / max(bw, 1)
    if aspect < 0.55:
        continue

    centre_score = 1.0 - abs(cx - w / 2) / (w * 0.25)
    score = min(0.99, 0.45 + body_fill * 0.25 + centre_score * 0.2 + min(aspect, 2.0) * 0.08)
    detections.append(((x1, y1, bw, bh), score, (sx, sy, sw, sh)))

detections.sort(key=lambda d: d[1], reverse=True)
vis = bgr.copy()
print(f"Found {len(detections)}")
for box, score, stripe in detections[:3]:
    x, y, bw, bh = box
    print(f"  score={score:.3f} box=({x},{y},{bw},{bh}) stripe={stripe}")
    cv2.rectangle(vis, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
    sx, sy, sw, sh = stripe
    cv2.rectangle(vis, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)

cv2.imwrite("/home/pi/Desktop/mars/debug/result_v4.jpg", vis)

# IoU
expected = (180, 70, 280, 320)
if detections:
    bx, by, bw, bh = detections[0][0]
    ex, ey, ew, eh = expected
    x1, y1 = max(ex, bx), max(ey, by)
    x2, y2 = min(ex+ew, bx+bw), min(ey+eh, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    iou = inter / (ew*eh + bw*bh - inter)
    print(f"IoU={iou:.3f} {'PASS' if iou > 0.2 else 'FAIL'}")
