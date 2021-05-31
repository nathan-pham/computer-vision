import mediapipe as mp
import cv2

import time

camera = cv2.VideoCapture(0)
window_name = "hand tracking"

# initialize window
cv2.namedWindow(window_name)
cv2.startWindowThread()

# initialize models
draw = mp.solutions.drawing_utils
model = mp.solutions.hands
hands = model.Hands()

previous_time = 0
current_time = 0

while cv2.getWindowProperty(window_name, 0) >= 0:
    success, img = camera.read()

    # get results
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    height, width, channels = img.shape

    # traverse hands
    for hand in results.multi_hand_landmarks or []:
        draw.draw_landmarks(img, hand, model.HAND_CONNECTIONS)
        
        # traverse nodes in hand
        for id, landmark in enumerate(hand.landmark):
            cx = int(landmark.x * width)
            cy = int(landmark.y * height)

            if id == 0:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

    # get & render fps
    current_time = time.time()
    fps = int(1 / (current_time - previous_time))
    previous_time = current_time
    cv2.putText(img, "fps: " + str(fps), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

    cv2.imshow(window_name, img)
    key_code = cv2.waitKey(50)