import mediapipe as mp
import cv2

import time

capture = cv2.VideoCapture(0)

previous_time = 0
current_time = 0

draw = mp.solutions.drawing_utils
model = mp.solutions.face_detection
face_detection = model.FaceDetection()


while True:
    _, frame = capture.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_results = face_detection.process(frame_rgb)

    for id, detection in enumerate(frame_results.detections or []):
        draw.draw_detection(frame, detection)        

    current_time = time.time()
    fps = int(1 / (current_time - previous_time))
    previous_time = current_time


    cv2.putText(frame, "fps: " + str(fps), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
    cv2.imshow("face detection", frame)

    if cv2.waitKey(50) == 27:
        break

capture.release()
cv2.destroyAllWindows()