import mediapipe as mp
import cv2

import time

class PoseDetector():
    def __init__(self, mode=False, upper_body=False, smooth_landmarks=True, detection_confidence=0.5, tracking_confidence=0.5):
        self.draw = mp.solutions.drawing_utils
        self.model = mp.solutions.pose
        self.pose = model.Pose(mode, upper_body, smooth_landmarks, detection_confidence, tracking_confidence)
        self.started = False

    def create_capture(self, source):
        capture = cv2.VideoCapture(source)

        if not self.started:
            cv2.startWindowThread()
            self.started = True

        return capture

    def resize_source(self, img, width=None, height=None):
        (h, w) = img.shape[:2]
        dimensions = (0, 0)
        
        if width:
            ratio = width / float(w)
            dimensions = (width, (int(h * ratio)))
        elif height:
            ratio = height / float(h)
            dimensions = (int(w * ratio), height)

        return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)

    def find_pose(self, img, draw=True):
        resized_img = self.resize_source(img, height=400)
        height, width = resized_img.shape[:2]

        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        landmarks = results.pose_landmarks or []
        
        if len(landmarks) > 0 and draw:
            self.draw.draw_landmarks(resized_img, landmarks, self.model.POSE_CONNECTIONS)

        return landmarks
        
    def find_position():
        pass
def main():
    pose_detector = PoseDetector()
    capture = pose_detector.create_capture("pose detection", 0)
    
    previous_time = 0
    current_time = 0

    while True:
    if results.pose_landmarks:
        draw.draw_landmarks(resized_img, results.pose_landmarks, model.POSE_CONNECTIONS)

        # for id, landmark in enumerate(results.pose_landmarks.landmark):
        #     cx, cy = int(landmark.x * width), int(landmark.y * height)
        #     if id == 0:
        #         cv2.circle(resized_img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

    # get & render fps
    current_time = time.time()
    fps = int(1 / (current_time - previous_time))
    previous_time = current_time
    cv2.putText(resized_img, "fps: " + str(fps), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

    cv2.imshow(window_name, resized_img)
    key_code = cv2.waitKey(50)

if __name__ == "__main__":
    main()