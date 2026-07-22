#!/usr/bin/env python3
import cv2
import numpy as np

bgr = cv2.imread("/home/pi/Desktop/mars/debug/cone_frame.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def detect_red_cone(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    white = cv2.inRange(hsv, (0, 0, 140), (180, 75, 255))
    red = cv2.inRange(hsv, (0, 35, 40), (20, 255, 255))
    red |= cv2.inRange(hsv, (160, 35, 40), (180, 255, 255))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    combined = red | white

    cx1, cx2 = int(w * 0.36), int(w * 0.64)
    col_w = cx2 - cx1
    row_white = np.count_nonzero(white[:, cx1:cx2], axis=1) / col_w
    row_red = np.count_nonzero(red[:, cx1:cx2], axis=1) / col_w

    scan_y1, scan_y2 = int(h * 0.12), int(h * 0.58)
    white_seg = row_white[scan_y1:scan_y2]
    if float(np.max(white_seg)) < 0.35:
        return None
    stripe_y = scan_y1 + int(np.argmax(white_seg))
    band_ratio = float(row_white[stripe_y])

    # Red above stripe → cone tip
    top_lo = max(int(h * 0.08), stripe_y - 150)
    above = np.where(row_red[top_lo : stripe_y - 3] >= 0.10)[0]
    top = top_lo + int(above[0]) if above.size else max(stripe_y - 80, top_lo)

    # Red below stripe → cone base (limited extent to avoid background)
    below_lo = stripe_y + 5
    below_hi = min(h - 1, stripe_y + 185)
    below = np.where(row_red[below_lo:below_hi] >= 0.12)[0]
    if below.size:
        bottom = below_lo + int(below[-1])
    else:
        # cone body may continue right below stripe through orange region
        below2 = np.where(row_red[below_lo:below_hi] >= 0.06)[0]
        bottom = below_lo + int(below2[-1]) if below2.size else stripe_y + 30

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
    return (x1, y1, bw, bh), score, stripe_y, top, bottom


r = detect_red_cone(rgb)
vis = bgr.copy()
if r:
    box, score, sy, top, bot = r
    print(f"score={score:.3f} box={box} stripe={sy} top={top} bottom={bot}")
    x, y, bw, bh = box
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
    cv2.imwrite("/home/pi/Desktop/mars/debug/cone_result_v3.jpg", vis)
else:
    print("fail")
