import cv2

image =cv2.imread("gears.png")

cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow("mask", cv2.WINDOW_GUI_NORMAL)