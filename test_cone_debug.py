#!/usr/bin/env python3
"""Debug each filter stage."""

import cv2
import numpy as np

bgr = cv2.imread("/home/pi/Desktop/mars/debug/frame.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

purple = cv2.inRange(hsv, (95, 42, 35), (118, 255, 175))
white = cv2.inRange(hsv, (0, 0, 155), (180, 55, 255))
bright = cv2.inRange(hsv, (0, 0, 175), (180, 35, 255))
purple = cv2.morphologyEx(purple, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
purple = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

rx1, rx2 = int(w * 0.15), int(w * 0.85)
ry1, ry2 = int(h * 0.05), int(h * 0.95)
roi_purple = purple[ry1:ry2, rx1:rx2]
contours, _ = cv2.findContours(roi_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

frame_cx = w / 2
print(f"Total contours in ROI: {len(contours)}")

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    x, y, bw, bh = cv2.boundingRect(contour)
    x += rx1
    y += ry1
    cx = x + bw / 2
    reasons = []

    if area < 2500:
        reasons.append(f"area={area:.0f}<2500")
    if area > h * w * 0.45:
        reasons.append(f"area={area:.0f} too big")
    if bh < 50:
        reasons.append(f"bh={bh}<50")
    if bw < 25:
        reasons.append(f"bw={bw}<25")
    if abs(cx - frame_cx) > w * 0.22:
        reasons.append(f"off-center cx={cx:.0f}")

    roi_p = purple[y : y + bh, x : x + bw]
    purple_fill = float(np.count_nonzero(roi_p)) / max(roi_p.size, 1)
    if purple_fill < 0.08:
        reasons.append(f"purple_fill={purple_fill:.3f}")

    stripe_y2 = y + int(bh * 0.65)
    stripe_region = white[y:stripe_y2, x : x + bw] | bright[y:stripe_y2, x : x + bw]
    band_ratio = float(np.count_nonzero(stripe_region)) / max(stripe_region.size, 1)
    if band_ratio < 0.025:
        reasons.append(f"band_ratio={band_ratio:.3f}")

    status = "PASS" if not reasons else "FAIL"
    if area > 500:
        print(f"  c{i}: area={area:.0f} box=({x},{y},{bw},{bh}) cx={cx:.0f} fill={purple_fill:.3f} band={band_ratio:.3f} -> {status} {reasons}")
