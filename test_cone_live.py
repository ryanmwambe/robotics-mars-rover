#!/usr/bin/env python3
"""Live test: capture frames and verify traffic cone detection."""

import sys
import time

import cv2

sys.path.insert(0, "/home/pi/Desktop/mars")
from mars_rover import (
    start_camera,
    capture_rgb_frame,
    detect_all,
    draw_detections,
    MIN_DISPLAY_CONFIDENCE,
)

NUM_FRAMES = 15
OUT = "/home/pi/Desktop/mars/debug"

start_camera()
time.sleep(1.5)

hits = 0
best_score = 0.0
best_frame = None
best_dets = []

print(f"Testing {NUM_FRAMES} live frames (threshold={MIN_DISPLAY_CONFIDENCE})...")
for i in range(NUM_FRAMES):
    frame = capture_rgb_frame()
    dets = detect_all(frame)
    cone = [d for d in dets if d.name == "traffic cone"]
    if cone:
        hits += 1
        score = cone[0].confidence
        if score > best_score:
            best_score = score
            best_frame = frame.copy()
            best_dets = cone
        print(f"  frame {i+1:2d}: DETECTED score={score:.3f} box={cone[0].box}")
    else:
        print(f"  frame {i+1:2d}: miss")
    time.sleep(0.15)

print(f"\nResult: {hits}/{NUM_FRAMES} frames detected cone")
if best_frame is not None:
    vis = draw_detections(best_frame, best_dets)
    cv2.imwrite(f"{OUT}/live_best.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Best score: {best_score:.3f} — saved {OUT}/live_best.jpg")

sys.exit(0 if hits >= NUM_FRAMES * 0.6 else 1)
