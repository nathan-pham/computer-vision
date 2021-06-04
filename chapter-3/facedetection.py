import mediapipe as mp
import cv2

import time

capture = cv2.VideoCapture(0)

previous_time = 0
current_time = 0

while True:
    _, frame = capture.read()


    current_time = time.time()
    fps = int(1 / (current_time - previous_time))
    previous_time = current_time


    cv2.putText(frame, "fps: " + str(fps), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
    cv2.imshow("face detection", frame)

    if cv2.waitKey(50) == 27:
        break

capture.release()
cv2.destroyAllWindows()