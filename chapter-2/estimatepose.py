import mediapipe as mp
import cv2

camera = cv2.VideoCapture("videos/girl-scooter.mp4")
window_name = "pose estimation"

cv2.namedWindow(window_name)
cv2.startWindowThread()

def resize_aspect(img, width=None, height=None):
    (o_height, o_width) = img.shape[:2]
    dimensions = [0, 0]
    
    if width:
        ratio = width / float(o_width)
        dimensions = (width, (int(o_height * ratio)))
    elif height:
        ratio = height / float(o_height)
        dimensions = (int(o_width * ratio), height)

    return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)

while cv2.getWindowProperty(window_name, 0) >= 0:
    success, img = camera.read()
    resized_img = resize_aspect(img=img, height=600)

    cv2.imshow(window_name, resized_img)
    key_code = cv2.waitKey(50)