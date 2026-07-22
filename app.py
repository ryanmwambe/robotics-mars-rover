#!/usr/bin/env python3
"""
Flask MJPEG stream with IMX500 object detection.

Uses the same detection pipeline as the official Raspberry Pi demo:
  ~/picamera2/examples/imx500/imx500_object_detection_demo.py
"""

import sys
from functools import lru_cache
from pathlib import Path

import cv2
from flask import Flask, Response

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
COCO_LABELS = Path("/home/pi/picamera2/examples/imx500/assets/coco_labels.txt")

THRESHOLD = 0.55
IOU = 0.65
MAX_DETECTIONS = 10

last_detections = []

app = Flask(__name__)

print("Loading IMX500...")
imx500 = IMX500(MODEL)

intrinsics = imx500.network_intrinsics
if not intrinsics:
    intrinsics = NetworkIntrinsics()
    intrinsics.task = "object detection"
elif intrinsics.task != "object detection":
    print("Network is not an object detection task", file=sys.stderr)
    sys.exit(1)

if intrinsics.labels is None:
    with open(COCO_LABELS, "r") as f:
        intrinsics.labels = f.read().splitlines()
intrinsics.update_with_defaults()

picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(
    controls={"FrameRate": intrinsics.inference_rate},
    buffer_count=12,
)

imx500.show_network_fw_progress_bar()
picam2.start(config)

if intrinsics.preserve_aspect_ratio:
    imx500.set_auto_aspect_ratio()

print("Camera running with IMX500 object detection")


class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0], conf=THRESHOLD, iou_thres=IOU, max_out_dets=MAX_DETECTIONS
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes

        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > THRESHOLD
    ]
    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def draw_detections_on_request(request, detections):
    """Draw detections using MappedArray, matching the official pre_callback path."""
    labels = get_labels()
    with MappedArray(request, "main") as m:
        for detection in detections:
            x, y, w, h = detection.box
            label = f"{labels[int(detection.category)]} {detection.conf * 100:.1f}%"

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            text_x = x + 5
            text_y = y + 15

            overlay = m.array.copy()
            cv2.rectangle(
                overlay,
                (text_x, text_y - text_height),
                (text_x + text_width, text_y + baseline),
                (255, 255, 255),
                cv2.FILLED,
            )

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            cv2.putText(
                m.array,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)
            cv2.putText(
                m.array,
                "ROI",
                (b_x + 5, b_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), color, thickness=2)

        return m.array.copy()


def generate():
    """Yield MJPEG frames with live IMX500 detections drawn on each frame."""
    while True:
        request = picam2.capture_request()
        try:
            metadata = request.get_metadata()
            detections = parse_detections(metadata)
            frame = draw_detections_on_request(request, detections)
        finally:
            request.release()

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, jpeg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )


@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IMX500 Object Detection</title>
        <style>
            body { font-family: sans-serif; text-align: center; background: #111; color: #eee; }
            h1 { margin-top: 1rem; }
            img { border: 2px solid #333; max-width: 100%; }
        </style>
    </head>
    <body>
        <h1>IMX500 Object Detection</h1>
        <img src="/video" width="640">
    </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
