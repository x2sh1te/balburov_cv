import cv2
import numpy as np
import random


NUM_BALLS = 3
ball_colors_hsv = []
secret_order = []
calibrated = False
clicked_pos = None


def on_click(event, x, y, flags, param):
    global clicked_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pos = (x, y)


cv2.namedWindow("Game")
cv2.setMouseCallback("Game", on_click)
cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    if clicked_pos and len(ball_colors_hsv) < NUM_BALLS:
        color = hsv[clicked_pos[1], clicked_pos[0]]
        ball_colors_hsv.append((
            np.clip(color * 0.8, 0, 255).astype(np.uint8),
            np.clip(color * 1.2, 0, 255).astype(np.uint8)
        ))
        clicked_pos = None

        if len(ball_colors_hsv) == NUM_BALLS:
            secret_order = list(range(NUM_BALLS))
            random.shuffle(secret_order)
            calibrated = True

    found_balls = []
    for i, (low, upp) in enumerate(ball_colors_hsv):
        mask = cv2.inRange(hsv, low, upp)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_cnt) > 800:
                (x, y), r = cv2.minEnclosingCircle(max_cnt)
                found_balls.append({'id': i, 'x': int(x)})
                cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 255), 2)

    if calibrated:
        if len(found_balls) == NUM_BALLS:
            found_balls.sort(key=lambda b: b['x'])
            if [b['id'] for b in found_balls] == secret_order:
                cv2.putText(frame, "МУЖИК", (150, 250), 0, 2, (0, 255, 0), 5)
    else:
        msg = f"Click on ball {len(ball_colors_hsv) + 1}"
        cv2.putText(frame, msg, (20, 50), 0, 0.8, (255, 255, 0), 2)

    cv2.imshow("Game", frame)
    if cv2.waitKey(1) == ord('q'): break

cam.release()
cv2.destroyAllWindows()