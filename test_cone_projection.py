#!/usr/bin/env python3
"""Row-projection cone detector — works when white stripe merges with background."""

import cv2
import numpy as np

bgr = cv2.imread("/home/pi/Desktop/mars/debug/frame.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

white = cv2.inRange(hsv, (0, 0, 145), (180, 70, 255))
purple = cv2.inRange(hsv, (90, 35, 30), (125, 255, 190))
purple = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

cx1, cx2 = int(w * 0.32), int(w * 0.68)
col_white = white[:, cx1:cx2]
col_purple = purple[:, cx1:cx2]
col_w = cx2 - cx1

row_white = np.count_nonzero(col_white, axis=1) / col_w
row_purple = np.count_nonzero(col_purple, axis=1) / col_w

# Find strongest white-band row in middle of frame
y_lo, y_hi = int(h * 0.15), int(h * 0.70)
segment = row_white[y_lo:y_hi]
peak_idx = int(np.argmax(segment))
stripe_y = y_lo + peak_idx
peak_val = segment[peak_idx]
print(f"White row peak at y={stripe_y}, fill={peak_val:.3f}")

# Find purple extent above/below stripe in center column
purple_seg = row_purple[y_lo:y_hi]
purple_rows = np.where(purple_seg >= 0.06)[0]
if purple_rows.size:
    p_top = y_lo + int(purple_rows[0])
    p_bot = y_lo + int(purple_rows[-1])
    print(f"Purple rows: {p_top} to {p_bot}")

# Build box from combined mask in center
combined = purple | white
center = combined[:, cx1:cx2]
row_combined = np.count_nonzero(center, axis=1) / col_w
rows = np.where(row_combined >= 0.05)[0]
if rows.size:
    ty, by = int(rows[0]), int(rows[-1])
    sub = center[ty:by+1, :]
    col_fill = np.count_nonzero(sub, axis=0) / max(sub.shape[0], 1)
    cols = np.where(col_fill >= 0.05)[0]
    if cols.size:
        lx, rx = int(cols[0]), int(cols[-1])
        box = (cx1 + lx, ty, rx - lx, by - ty)
        print(f"Center combined box: {box}")

        vis = bgr.copy()
        x, y, bw, bh = box
        cv2.rectangle(vis, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
        cv2.line(vis, (cx1, stripe_y), (cx2, stripe_y), (255, 0, 0), 2)
        cv2.imwrite("/home/pi/Desktop/mars/debug/result_projection.jpg", vis)

        expected = (180, 70, 280, 320)
        ex, ey, ew, eh = expected
        xi, yi = max(ex, x), max(ey, y)
        xa, ya = min(ex+ew, x+bw), min(ey+eh, y+bh)
        inter = max(0, xa-xi) * max(0, ya-yi)
        iou = inter / (ew*eh + bw*bh - inter)
        print(f"IoU={iou:.3f}")
