#!/usr/bin/env python3
"""Capture a frame and debug hybrid balloon detection."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.prefix != sys.base_prefix:
    sys.path.insert(0, "/usr/lib/python3/dist-packages")

import cv2
from picamera2 import MappedArray, Picamera2

import balloon

OUT = Path(__file__).resolve().parent / "debug_balloons"
FRAME_SIZE = (640, 480)


def capture_frame():
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(main={"size": FRAME_SIZE}, buffer_count=4)
    )
    picam2.start()
    for _ in range(5):
        picam2.capture_array("main")
    request = picam2.capture_request()
    try:
        with MappedArray(request, "main") as m:
            frame = m.array.copy()
    finally:
        request.release()
    picam2.stop()
    picam2.close()
    return frame


def main() -> None:
    OUT.mkdir(exist_ok=True)
    balloon.load_models()
    frame = capture_frame()
    cv2.imwrite(str(OUT / "frame.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    detections = balloon.detect_objects(frame)
    annotated = frame.copy()
    balloon.draw_detections(annotated, detections)
    cv2.imwrite(str(OUT / "detections.jpg"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    print(f"Detected {len(detections)} balloon(s):")
    for label, x1, y1, x2, y2, conf in detections:
        print(f"  {label}: ({x1},{y1})-({x2},{y2}) conf={conf:.2f}")

    if not detections:
        print("No balloon detected — hold a balloon in front of the camera and retry.")
        sys.exit(1)

    print(f"Saved debug images to {OUT}")


if __name__ == "__main__":
    main()
