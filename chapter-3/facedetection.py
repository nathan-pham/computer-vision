import mediapipe as mp
import cv2

import time

capture = cv2.VideoCapture(0)

previous_time = 0
current_time = 0

draw = mp.solutions.drawing_utils
model = mp.solutions.face_detection
face_detection = model.FaceDetection(0.75)

while True:
    _, frame = capture.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_results = face_detection.process(frame_rgb)

    for id, detection in enumerate(frame_results.detections or []):
        # draw.draw_detection(frame, detection)

        _bbox = detection.location_data.relative_bounding_box
        height, width = frame.shape[:2]
        bbox = (int(_bbox.xmin * width), int(_bbox.ymin * height), 
                int(_bbox.width * width), int(_bbox.height * height)) 

        cv2.rectangle(frame, bbox, (255, 0, 255), 2)
        cv2.putText(frame, f"detection_score: {int(detection.score[0] * 100)}%", (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

    current_time = time.time()
    fps = int(1 / (current_time - previous_time))
    previous_time = current_time


    cv2.putText(frame, f"fps: {fps}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
    cv2.imshow("face detection", frame)

    if cv2.waitKey(50) == 27:
        break

capture.release()
cv2.destroyAllWindows()