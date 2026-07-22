#!/usr/bin/env python3
"""Test improved red cone detector before patching mars_rover."""

import cv2
import numpy as np

bgr = cv2.imread("/home/pi/Desktop/mars/debug/cone_frame.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def detect_red_cone(frame: np.ndarray) -> tuple[tuple, float] | None:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    white = cv2.inRange(hsv, (0, 0, 140), (180, 75, 255))
    red = cv2.inRange(hsv, (0, 35, 40), (20, 255, 255))
    red |= cv2.inRange(hsv, (160, 35, 40), (180, 255, 255))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    combined = red | white

    cx1, cx2 = int(w * 0.30), int(w * 0.70)
    col_w = cx2 - cx1
    row_white = np.count_nonzero(white[:, cx1:cx2], axis=1) / col_w
    row_red = np.count_nonzero(red[:, cx1:cx2], axis=1) / col_w

    scan_y1, scan_y2 = int(h * 0.12), int(h * 0.58)
    white_seg = row_white[scan_y1:scan_y2]
    if white_seg.size == 0 or float(np.max(white_seg)) < 0.35:
        return None

    stripe_y = scan_y1 + int(np.argmax(white_seg))
    band_ratio = float(row_white[stripe_y])

    # Scan upward through red tip (allow short gaps)
    top = stripe_y
    gap = 0
    while top > int(h * 0.05):
        top -= 1
        if row_red[top] >= 0.10 or (row_white[top] >= 0.15 and top < stripe_y - 5):
            gap = 0
        else:
            gap += 1
            if gap >= 8:
                top += gap
                break
    top = max(top, int(h * 0.05))

    # Scan downward — tolerate gap between stripe and lower red base
    bottom = stripe_y
    best_bottom = stripe_y
    gap = 0
    limit = min(h - 1, stripe_y + int(h * 0.45))
    while bottom < limit:
        bottom += 1
        if row_red[bottom] >= 0.12:
            best_bottom = bottom
            gap = 0
        else:
            gap += 1
            if gap >= 40:
                break
    bottom = best_bottom

    if bottom - top < 45:
        return None

    sub = combined[top : bottom + 1, cx1:cx2]
    col_fill = np.count_nonzero(sub, axis=0) / max(sub.shape[0], 1)
    cols = np.where(col_fill >= 0.08)[0]
    if cols.size < 4:
        return None

    pad = 10
    x1 = max(cx1 + int(cols[0]) - pad, 0)
    x2 = min(cx1 + int(cols[-1]) + pad, w)
    y1 = max(top - pad, 0)
    y2 = min(bottom + pad, h)
    bw, bh = x2 - x1, y2 - y1
    if bw < 25 or bh < 50:
        return None

    body = red[y1:y2, x1:x2]
    body_fill = float(np.count_nonzero(body)) / max(body.size, 1)
    aspect = bh / max(bw, 1)
    centre = 1.0 - abs((x1 + bw / 2) - w / 2) / (w * 0.25)
    score = min(0.99, 0.38 + body_fill * 0.22 + band_ratio * 0.22 + centre * 0.10 + min(aspect, 2.0) * 0.08)
    return (x1, y1, bw, bh), score


result = detect_red_cone(rgb)
vis = bgr.copy()
if result:
    box, score = result
    x, y, bw, bh = box
    print(f"DETECTED score={score:.3f} box={box}")
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
    cv2.putText(vis, f"cone {score*100:.0f}%", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    ex, ey, ew, eh = 200, 60, 250, 340
    xi, yi = max(ex, x), max(ey, y)
    xa, ya = min(ex + ew, x + bw), min(ey + eh, y + bh)
    inter = max(0, xa - xi) * max(0, ya - yi)
    iou = inter / (ew * eh + bw * bh - inter)
    print(f"IoU={iou:.3f}")
else:
    print("FAIL")

cv2.imwrite("/home/pi/Desktop/mars/debug/cone_result_test.jpg", vis)
