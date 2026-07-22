#!/usr/bin/env python3
"""Capture frame and diagnose traffic cone visibility."""

import cv2
import numpy as np
from picamera2 import Picamera2

OUT_DIR = "/home/pi/Desktop/mars/debug"
FRAME_SIZE = (640, 480)

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": FRAME_SIZE, "format": "RGB888"},
        buffer_count=4,
    )
)
picam2.start()
import time
time.sleep(1.5)
frame = picam2.capture_array()
picam2.stop()
picam2.close()

import os
os.makedirs(OUT_DIR, exist_ok=True)
cv2.imwrite(f"{OUT_DIR}/frame.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

# Try several red ranges
ranges = {
    "red_strict": ((0, 110, 70), (10, 255, 255), (170, 110, 70), (180, 255, 255)),
    "red_loose": ((0, 80, 50), (15, 255, 255), (165, 80, 50), (180, 255, 255)),
    "red_very_loose": ((0, 40, 40), (20, 255, 255), (160, 40, 40), (180, 255, 255)),
}

for name, *pairs in [(k, v[0], v[1], v[2], v[3]) for k, v in ranges.items()]:
    lo1, hi1, lo2, hi2 = pairs[0], pairs[1], pairs[2], pairs[3]
    mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)
    pct = 100.0 * np.count_nonzero(mask) / mask.size
    cv2.imwrite(f"{OUT_DIR}/mask_{name}.jpg", mask)
    print(f"{name}: {pct:.1f}% red pixels")

# White ranges
for sat_max in [60, 80, 100]:
    white = cv2.inRange(hsv, (0, 0, 160), (180, sat_max, 255))
    pct = 100.0 * np.count_nonzero(white) / white.size
    cv2.imwrite(f"{OUT_DIR}/mask_white_sat{sat_max}.jpg", white)
    print(f"white sat<={sat_max}: {pct:.1f}%")

# Center region stats (cone likely centered)
h, w = frame.shape[:2]
cx, cy = w // 2, h // 2
roi = hsv[cy - 120 : cy + 120, cx - 120 : cx + 120]
print(f"\nCenter ROI HSV mean: H={roi[:,:,0].mean():.1f} S={roi[:,:,1].mean():.1f} V={roi[:,:,2].mean():.1f}")
print(f"Center ROI HSV std:  H={roi[:,:,0].std():.1f} S={roi[:,:,1].std():.1f} V={roi[:,:,2].std():.1f}")

# Save center crop
center = frame[cy - 120 : cy + 120, cx - 120 : cx + 120]
cv2.imwrite(f"{OUT_DIR}/center_crop.jpg", cv2.cvtColor(center, cv2.COLOR_RGB2BGR))
print(f"\nSaved debug images to {OUT_DIR}/")
