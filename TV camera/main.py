import cv2
import numpy as np
import time

tv = cv2.imread("news.jpg")

pts2 = np.array([[18, 25], [432, 53], [435, 270], [39, 294]], dtype="float32")

cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
ret, first_frame = cam.read()
if not ret:
    print("Камера не работает")
    cam.release()
    exit()

rows, cols, _ = first_frame.shape
pts1 = np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]], dtype="float32")
M = cv2.getPerspectiveTransform(pts1, pts2)

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break

    transformed = cv2.warpPerspective(frame, M, (tv.shape[1], tv.shape[0]))

    mask = np.zeros(tv.shape[:2], dtype="uint8")
    cv2.fillConvexPoly(mask, pts2.astype(int), 255)

    fg = cv2.bitwise_and(transformed, transformed, mask=mask)
    bg = cv2.bitwise_and(tv, tv, mask=cv2.bitwise_not(mask))

    result = cv2.add(bg, fg)
    cv2.imshow("TV Result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()