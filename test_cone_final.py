#!/usr/bin/env python3
"""Test final cone detector design."""

import math
import cv2
import numpy as np

FRAME = "/home/pi/Desktop/mars/debug/frame.jpg"
OUT = "/home/pi/Desktop/mars/debug"


def detect_traffic_cone(frame: np.ndarray) -> list[tuple]:
    """
    Detect blue/purple traffic cones with a white reflective band.
    Competition cone is blue-violet (NOT orange/red).
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # --- colour masks ---
    # Purple/violet cone body (tuned from live camera samples)
    purple = cv2.inRange(hsv, (95, 42, 35), (118, 255, 175))
    # White / silver reflector stripe
    white = cv2.inRange(hsv, (0, 0, 155), (180, 55, 255))
    # Also catch over-exposed stripe (very low saturation, high value)
    bright = cv2.inRange(hsv, (0, 0, 175), (180, 35, 255))

    purple = cv2.morphologyEx(purple, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    purple = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

    # Search region: centre 70 % of frame, skip top 5 %
    rx1, rx2 = int(w * 0.15), int(w * 0.85)
    ry1, ry2 = int(h * 0.05), int(h * 0.95)
    roi_purple = purple[ry1:ry2, rx1:rx2]

    contours, _ = cv2.findContours(roi_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def row_width(mask_roi: np.ndarray, row: int) -> int:
        if row < 0 or row >= mask_roi.shape[0]:
            return 0
        cols = np.where(mask_roi[row, :] > 0)[0]
        return int(cols[-1] - cols[0]) if cols.size >= 2 else 0

    candidates = []
    frame_cx = w / 2

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2500 or area > h * w * 0.45:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        # Map back to full frame coords
        x += rx1
        y += ry1

        if bh < 50 or bw < 25:
            continue

        cx = x + bw / 2
        # Must be roughly centred
        if abs(cx - frame_cx) > w * 0.22:
            continue

        aspect = bh / max(bw, 1)

        roi_p = purple[y : y + bh, x : x + bw]
        purple_fill = float(np.count_nonzero(roi_p)) / max(roi_p.size, 1)
        if purple_fill < 0.08:
            continue

        # White stripe in upper 65 % of bounding box
        stripe_y2 = y + int(bh * 0.65)
        stripe_region = white[y:stripe_y2, x : x + bw] | bright[y:stripe_y2, x : x + bw]
        band_ratio = float(np.count_nonzero(stripe_region)) / max(stripe_region.size, 1)
        if band_ratio < 0.025:
            continue

        # Taper check: cone wider at bottom
        top_w = row_width(roi_p, int(bh * 0.08))
        bot_w = row_width(roi_p, int(bh * 0.88))
        taper = bot_w / max(top_w, 1) if top_w > 3 else 2.0

        # Centre preference (closer to middle = better)
        centre_score = 1.0 - abs(cx - frame_cx) / (w * 0.22)
        # Lower in frame = closer to camera on table
        lower_score = min((y + bh) / h, 1.0)

        score = min(
            0.99,
            0.30
            + purple_fill * 0.22
            + band_ratio * 0.22
            + centre_score * 0.12
            + lower_score * 0.08
            + min(taper / 4.0, 0.08),
        )

        # Expand box to include stripe and tip
        pad_x = max(int(bw * 0.05), 2)
        pad_top = max(int(bh * 0.05), 4)
        pad_bot = max(int(bh * 0.05), 4)
        box = (
            max(x - pad_x, 0),
            max(y - pad_top, 0),
            min(x + bw + pad_x, w) - max(x - pad_x, 0),
            min(y + bh + pad_bot, h) - max(y - pad_top, 0),
        )
        candidates.append((box, score, area, taper, band_ratio))

    candidates.sort(key=lambda c: c[1], reverse=True)
    return candidates


def main():
    bgr = cv2.imread(FRAME)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    dets = detect_traffic_cone(rgb)

    vis = bgr.copy()
    print(f"Detections: {len(dets)}")
    for i, (box, score, area, taper, band) in enumerate(dets[:3]):
        x, y, bw, bh = box
        print(f"  #{i} score={score:.3f} box=({x},{y},{bw},{bh}) area={area:.0f} taper={taper:.2f} band={band:.3f}")
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
        cv2.putText(vis, f"cone {score*100:.0f}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(f"{OUT}/result_final.jpg", vis)

    # IoU against expected cone region
    expected = (180, 70, 280, 320)
    if dets:
        best = dets[0][0]
        ex, ey, ew, eh = expected
        bx, by, bw, bh = best
        x1, y1 = max(ex, bx), max(ey, by)
        x2, y2 = min(ex + ew, bx + bw), min(ey + eh, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = ew * eh + bw * bh - inter
        iou = inter / union if union else 0
        print(f"IoU vs expected cone box: {iou:.3f}")
        print("PASS" if iou > 0.25 and dets[0][1] >= 0.68 else "FAIL - tuning needed")


if __name__ == "__main__":
    main()
