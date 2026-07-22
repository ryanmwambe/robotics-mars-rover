#!/usr/bin/env python3
"""Capture live frame and diagnose red cone detection."""

import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/home/pi/Desktop/mars")
from mars_rover import start_camera, capture_rgb_frame, detect_all, draw_detections, MIN_DISPLAY_CONFIDENCE

OUT = "/home/pi/Desktop/mars/debug"

start_camera()
time.sleep(1.5)
frame = capture_rgb_frame()
hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
h, w = frame.shape[:2]

cv2.imwrite(f"{OUT}/cone_frame.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Sample center column
x = w // 2
print(f"Vertical scan at x={x}:")
for y in range(40, 420, 20):
    p = hsv[y, x]
    r = frame[y, x]
    print(f"  y={y:3d} RGB=({r[0]:3d},{r[1]:3d},{r[2]:3d}) HSV=({p[0]:3d},{p[1]:3d},{p[2]:3d})")

# Red mask variants
for name, lo1, hi1, lo2, hi2 in [
    ("strict", (0, 70, 50), (12, 255, 255), (168, 70, 50), (180, 255, 255)),
    ("loose", (0, 45, 40), (15, 255, 255), (165, 45, 40), (180, 255, 255)),
    ("orange", (0, 35, 40), (20, 255, 255), (160, 35, 40), (180, 255, 255)),
]:
    mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)
    pct = 100 * np.count_nonzero(mask) / mask.size
    cv2.imwrite(f"{OUT}/mask_red_{name}.jpg", mask)
    print(f"red_{name}: {pct:.1f}%")

white = cv2.inRange(hsv, (0, 0, 145), (180, 70, 255))
cv2.imwrite(f"{OUT}/mask_white.jpg", white)

# Center column analysis
cx1, cx2 = int(w * 0.33), int(w * 0.67)
col_w = cx2 - cx1
row_white = np.count_nonzero(white[:, cx1:cx2], axis=1) / col_w
red = cv2.inRange(hsv, (0, 45, 40), (15, 255, 255)) | cv2.inRange(hsv, (165, 45, 40), (180, 255, 255))
row_red = np.count_nonzero(red[:, cx1:cx2], axis=1) / col_w

for sy1, sy2 in [(int(h*0.15), int(h*0.55)), (int(h*0.12), int(h*0.65))]:
    seg = row_white[sy1:sy2]
    peak = sy1 + int(np.argmax(seg))
    print(f"white scan {sy1}-{sy2}: peak y={peak} val={float(np.max(seg)):.3f}")

seg_r = row_red[int(h*0.08):int(h*0.78)]
rows = np.where(seg_r >= 0.06)[0]
if rows.size:
    print(f"red rows in center: {int(h*0.08)+rows[0]} to {int(h*0.08)+rows[-1]}")

dets = detect_all(frame)
print(f"\ndetect_all: {[(d.name, round(d.confidence,3), d.box) for d in dets]}")

if dets:
    vis = draw_detections(frame.copy(), dets)
    cv2.imwrite(f"{OUT}/cone_detect.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
else:
    print("FAIL - debugging each filter stage...")
    import mars_rover as mr
    skin = mr._skin_mask(hsv)
    # manual step through
    scan_y1, scan_y2 = int(h * 0.15), int(h * 0.55)
    white_window = row_white[scan_y1:scan_y2]
    print(f"  white max in scan: {float(np.max(white_window)):.3f}")
    if float(np.max(white_window)) >= 0.04:
        stripe_y = scan_y1 + int(np.argmax(white_window))
        bottom = stripe_y
        low_rows = 0
        red_strict = cv2.inRange(hsv, (0, 70, 50), (12, 255, 255)) | cv2.inRange(hsv, (168, 70, 50), (180, 255, 255))
        row_red_s = np.count_nonzero(red_strict[:, cx1:cx2], axis=1) / col_w
        while bottom < min(h - 1, stripe_y + int(h * 0.35)):
            if row_red_s[bottom] < 0.30:
                low_rows += 1
            else:
                low_rows = 0
            if low_rows >= 4:
                break
            bottom += 1
        bottom = max(bottom - 4, stripe_y + 10)
        below_h = bottom - stripe_y
        print(f"  stripe_y={stripe_y} bottom={bottom} below_h={below_h}")
        print(f"  row_red at stripe={row_red_s[stripe_y]:.3f} at bottom={row_red_s[min(bottom,h-1)]:.3f}")
