import mediapipe as mp
import cv2

import time

camera = cv2.VideoCapture("videos/girl-dancing.mp4")
window_name = "pose estimation"

# initialize window
cv2.namedWindow(window_name)
cv2.startWindowThread()

# initialize models
draw = mp.solutions.drawing_utils
model = mp.solutions.pose
pose = model.Pose()

previous_time = 0
current_time = 0

# resize image given width or height
def resize_aspect(img, width=None, height=None):
    (o_height, o_width) = img.shape[:2]
    dimensions = [0, 0]

    if width:
        ratio = width / float(o_width)
        dimensions = (width, (int(o_height * ratio)))
    elif height:
        ratio = height / float(o_height)
        dimensions = (int(o_width * ratio), height)

    # resize image after calculating new dimensions
    return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)

while cv2.getWindowProperty(window_name, 0) >= 0:
    success, img = camera.read()
    resized_img = resize_aspect(img=img, height=400)

    # get results
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        draw.draw_landmarks(resized_img, results.pose_landmarks, model.POSE_CONNECTIONS)

    # get & render fps
    current_time = time.time()
    fps = int(1 / (current_time - previous_time))
    previous_time = current_time
    cv2.putText(resized_img, "fps: " + str(fps), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

    cv2.imshow(window_name, resized_img)
    key_code = cv2.waitKey(50)