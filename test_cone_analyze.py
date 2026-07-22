#!/usr/bin/env python3
"""Analyze blue cone HSV and test detection candidates."""

import cv2
import numpy as np

frame = cv2.imread("/home/pi/Desktop/mars/debug/frame.jpg")
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

OUT = "/home/pi/Desktop/mars/debug"

# Blue/purple hue ranges in OpenCV (H: 0-180)
blue_ranges = [
    ("blue_90_130", (90, 40, 40), (130, 255, 255)),
    ("blue_100_140", (100, 50, 50), (140, 255, 255)),
    ("blue_95_135_lo", (95, 30, 30), (135, 255, 255)),
    ("purple_110_150", (110, 40, 40), (150, 255, 255)),
]

for name, lo, hi in blue_ranges:
    mask = cv2.inRange(hsv, lo, hi)
    pct = 100.0 * np.count_nonzero(mask) / mask.size
    cv2.imwrite(f"{OUT}/mask_{name}.jpg", mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) > 500]
    print(f"{name}: {pct:.1f}% pixels, {len(big)} contours >500px")

# Sample pixels from known cone regions (approximate from image)
# Top purple tip ~ center-top of cone
# White band ~ middle
h, w = frame_rgb.shape[:2]
samples = {
    "cone_tip": (320, 120),
    "white_band": (320, 200),
    "cone_body": (320, 280),
    "table": (320, 400),
}
print("\nPixel HSV samples:")
for label, (px, py) in samples.items():
    if 0 <= py < h and 0 <= px < w:
        p = hsv[py, px]
        print(f"  {label} ({px},{py}): H={p[0]} S={p[1]} V={p[2]}")

# Combined blue + white cone detection test
blue = cv2.inRange(hsv, (95, 35, 35), (145, 255, 255))
white = cv2.inRange(hsv, (0, 0, 150), (180, 70, 255))
blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

vis = frame.copy()
for c in contours:
    area = cv2.contourArea(c)
    if area < 800:
        continue
    x, y, bw, bh = cv2.boundingRect(c)
    asp = bh / max(bw, 1)
    print(f"  contour area={area:.0f} box=({x},{y},{bw},{bh}) aspect={asp:.2f}")
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

cv2.imwrite(f"{OUT}/test_blue_contours.jpg", vis)
print(f"\nSaved {OUT}/test_blue_contours.jpg")
