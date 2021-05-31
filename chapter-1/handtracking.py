import mediapipe as mp
import cv2

import time

camera = cv2.VideoCapture(0)
window_name = "image"

cv2.namedWindow(window_name)
cv2.startWindowThread()

while cv2.getWindowProperty(window_name, 0) >= 0:
    success, img = camera.read()
    cv2.imshow(window_name, img)

    keyCode = cv2.waitKey(50)
    # cv2.waitKey(1)