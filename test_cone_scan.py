#!/usr/bin/env python3
"""Vertical scan through cone center to find color profile."""

import cv2
import numpy as np

bgr = cv2.imread("/home/pi/Desktop/mars/debug/frame.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
h, w = rgb.shape[:2]

# Scan vertical line at x=320
x = 320
print(f"Vertical scan at x={x}:")
for y in range(60, 420, 20):
    p = hsv[y, x]
    r = rgb[y, x]
    print(f"  y={y:3d} RGB=({r[0]:3d},{r[1]:3d},{r[2]:3d}) HSV=({p[0]:3d},{p[1]:3d},{p[2]:3d})")

# Try tighter cone-only mask: purple with moderate sat/value
masks = {
    "cone_tight": cv2.inRange(hsv, (98, 50, 35), (125, 255, 180)),
    "cone_mid": cv2.inRange(hsv, (90, 40, 30), (130, 255, 200)),
}
OUT = "/home/pi/Desktop/mars/debug"
for name, mask in masks.items():
    cv2.imwrite(f"{OUT}/mask_{name}.jpg", mask)
    pct = 100 * np.count_nonzero(mask) / mask.size
    print(f"\n{name}: {pct:.1f}%")

# Center ROI only (where cone lives)
roi = hsv[int(h*0.08):int(h*0.85), int(w*0.25):int(w*0.75)]
print(f"\nCenter ROI HSV percentiles:")
for ch, name in enumerate("HSV"):
    vals = roi[:, :, ch].ravel()
    print(f"  {name}: p10={np.percentile(vals,10):.0f} p50={np.percentile(vals,50):.0f} p90={np.percentile(vals,90):.0f}")

# Mask within center ROI and find contours
y1, y2 = int(h * 0.08), int(h * 0.85)
x1, x2 = int(w * 0.2), int(w * 0.8)
center_hsv = hsv[y1:y2, x1:x2]
mask = cv2.inRange(center_hsv, (98, 50, 35), (125, 255, 180))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
vis = bgr.copy()
for c in contours:
    area = cv2.contourArea(c)
    if area < 500:
        continue
    rx, ry, rw, rh = cv2.boundingRect(c)
    print(f"  center contour: area={area:.0f} box=({x1+rx},{y1+ry},{rw},{rh}) aspect={rh/max(rw,1):.2f}")
    cv2.rectangle(vis, (x1+rx, y1+ry), (x1+rx+rw, y1+ry+rh), (0, 255, 0), 2)
cv2.imwrite(f"{OUT}/center_tight_contours.jpg", vis)
