#!/usr/bin/env python3
"""Trim using center-column density only."""

import cv2
import numpy as np

bgr = cv2.imread("/home/pi/Desktop/mars/debug/frame.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

purple = cv2.inRange(hsv, (90, 35, 30), (125, 255, 185))
white = cv2.inRange(hsv, (0, 0, 150), (180, 65, 255))
purple = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

# Stripe at (304, 146, 92, 111) from v4
sx, sy, sw, sh = 304, 146, 92, 111
pad_x, pad_up, pad_down = 50, 90, 100
x1, x2 = sx - pad_x, sx + sw + pad_x
y1, y2 = sy - pad_up, sy + sh + pad_down

# Use center 60% of x range for row density
bw = x2 - x1
cx1 = x1 + int(bw * 0.2)
cx2 = x1 + int(bw * 0.8)

combined = purple | white
roi = combined[y1:y2, cx1:cx2]
row_fill = np.count_nonzero(roi, axis=1) / roi.shape[1]

# Find contiguous block of high-density rows around stripe
threshold = 0.08
in_run = False
best_run = (0, 0, 0)
current = (0, 0)
for i, rf in enumerate(row_fill):
    if rf >= threshold:
        if not in_run:
            current = (i, i)
            in_run = True
        else:
            current = (current[0], i)
    else:
        if in_run:
            length = current[1] - current[0]
            if length > best_run[2]:
                best_run = (current[0], current[1], length)
            in_run = False
if in_run:
    length = current[1] - current[0]
    if length > best_run[2]:
        best_run = (current[0], current[1], length)

print(f"Best row run: {best_run} (relative to y1={y1})")
ty = y1 + best_run[0]
by = y1 + best_run[1]
print(f"Trimmed y: {ty} to {by}, height={by-ty}")

# Column trim on that row range
sub = combined[ty:by, x1:x2]
col_fill = np.count_nonzero(sub, axis=0) / max(sub.shape[0], 1)
cols = np.where(col_fill >= 0.08)[0]
if cols.size:
    lx, rx = int(cols[0]), int(cols[-1])
    fx1, fx2 = x1 + lx, x1 + rx
    print(f"Trimmed x: {fx1} to {fx2}, width={fx2-fx1}")

vis = bgr.copy()
cv2.rectangle(vis, (fx1, ty), (fx2, by), (0, 255, 0), 3)
cv2.imwrite("/home/pi/Desktop/mars/debug/result_v6.jpg", vis)

expected = (180, 70, 280, 320)
bx, by2, bw, bh = fx1, ty, fx2-fx1, by-ty
ex, ey, ew, eh = expected
xi, yi = max(ex, bx), max(ey, by2)
xa, ya = min(ex+ew, bx+bw), min(ey+eh, by2+bh)
inter = max(0, xa-xi) * max(0, ya-yi)
iou = inter / (ew*eh + bw*bh - inter)
print(f"IoU={iou:.3f}")
