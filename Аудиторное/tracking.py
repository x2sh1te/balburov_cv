import cv2
import numpy as np
from math import dist
import time
from pathlib import Path
import json

save_path = Path(__file__).parent
config_path = save_path / "config.json"

cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_GUI_NORMAL)

position = (0, 0)
clicked = False
lower = None
upper = None
positions = []
prev_time = time.time()

d = 6.36

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global position, clicked
        position = (x, y)
        clicked = True

cv2.setMouseCallback("Image", on_click)
cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    if clicked:
        clicked = False
        color = hsv[position[1]][position[0]]
        lower = np.clip(color * 0.9, 0, 255).astype(np.uint8)
        upper = np.clip(color * 1.1, 0, 255).astype(np.uint8)

    if lower is not None and upper is not None:
        inr = cv2.inRange(hsv, lower, upper)
        inr = cv2.morphologyEx(inr, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
        inr = cv2.GaussianBlur(inr, (7, 7), 0)
        cv2.imshow("Mask", inr)

        contours, _ = cv2.findContours(inr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(max_contour)
            if radius > 10:
                x = int(x)
                y = int(y)
                radius = int(radius)
                cv2.circle(frame, (x, y), radius, (0, 255, 255), 4)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                positions.append((x, y))
                if len(positions) > 20:
                    positions.pop(0)

                for i, pos in enumerate(positions[:-1]):
                    color_val = int(20 + 155 * i / len(positions))
                    cv2.circle(frame, pos, 5, (0, 0, color_val), -1)

                curr_time = time.time()
                delta = curr_time - prev_time
                if len(positions) >= 2 and delta > 0:
                    dst = dist(positions[-1], positions[-2])
                    pxl_per_cm = d / (2 * radius)
                    pxl_per_m = pxl_per_cm / 100
                    speed = (dst / delta) * pxl_per_m
                    cv2.putText(frame, f"speed: {speed:.2f} m/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
                prev_time = curr_time

    cv2.imshow("Image", frame)

cam.release()
cv2.destroyAllWindows()

with config_path.open("w", encoding="utf-8") as f:
    json.dump({
        "lower": None if lower is None else lower.tolist(),
        "upper": None if upper is None else upper.tolist()
    }, f, indent=2)