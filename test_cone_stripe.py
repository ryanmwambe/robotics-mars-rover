#!/usr/bin/env python3
import cv2
import numpy as np

bgr = cv2.imread("/home/pi/Desktop/mars/debug/frame.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
h, w = rgb.shape[:2]

white = cv2.inRange(hsv, (0, 0, 155), (180, 60, 255))
bright = cv2.inRange(hsv, (0, 0, 170), (180, 40, 255))
stripe_mask = white | bright
cv2.imwrite("/home/pi/Desktop/mars/debug/stripe_mask.jpg", stripe_mask)

contours, _ = cv2.findContours(stripe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("White/bright contours:")
for c in contours:
    area = cv2.contourArea(c)
    if area < 100:
        continue
    x, y, bw, bh = cv2.boundingRect(c)
    if bw < 20:
        continue
    horiz = bw / max(bh, 1)
    print(f"  area={area:.0f} box=({x},{y},{bw},{bh}) horiz={horiz:.1f}")

# Purple around stripe at y~280
purple = cv2.inRange(hsv, (95, 42, 35), (118, 255, 175))
# Wider purple for body
purple2 = cv2.inRange(hsv, (90, 35, 30), (130, 255, 200))

# Combine: dilate purple vertically
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 21))
purple_v = cv2.morphologyEx(purple2, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("/home/pi/Desktop/mars/debug/purple_v.jpg", purple_v)

# Center column focus
cx1, cx2 = int(w*0.28), int(w*0.72)
col_mask = np.zeros((h, w), np.uint8)
col_mask[:, cx1:cx2] = purple_v[:, cx1:cx2]
contours, _ = cv2.findContours(col_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
vis = bgr.copy()
print("\nCenter column purple contours:")
for c in contours:
    area = cv2.contourArea(c)
    if area < 800:
        continue
    x, y, bw, bh = cv2.boundingRect(c)
    print(f"  area={area:.0f} box=({x},{y},{bw},{bh}) aspect={bh/max(bw,1):.2f}")
    cv2.rectangle(vis, (x,y), (x+bw,y+bh), (0,255,0), 2)
cv2.imwrite("/home/pi/Desktop/mars/debug/center_col_contours.jpg", vis)
