import mediapipe as mp
import cv2

import time

class PoseDetector():
    def __init__(self, mode=False, upper_body=False, smooth_landmarks=True, detection_confidence=0.5, tracking_confidence=0.5):
        self.draw = mp.solutions.drawing_utils
        self.model = mp.solutions.pose
        self.pose = self.model.Pose(mode, upper_body, smooth_landmarks, detection_confidence, tracking_confidence)
        self.started = False

    def create_capture(self, source):
        capture = cv2.VideoCapture(source)

        if not self.started:
            cv2.startWindowThread()
            self.started = True

        return capture

    def flip_source(self, img, code):
        return cv2.flip(img, code)

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
        landmarks = results.pose_landmarks
        
        if landmarks and draw:
            self.draw.draw_landmarks(resized_img, landmarks, self.model.POSE_CONNECTIONS)

        return resized_img, landmarks

    def find_position(self, img, pose, draw=True):
        height, width = img.shape[:2]
        landmark_list = []

        if pose:
            for id, landmark in enumerate(pose.landmark):
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return img, landmark_list

def main():
    pose_detector = PoseDetector()
    capture = pose_detector.create_capture(0)
    
    previous_time = 0
    current_time = 0

    while True:
        _, img = capture.read()
        
        img, pose = pose_detector.find_pose(img)
        img, landmark_list = pose_detector.find_position(img, pose)

        current_time = time.time()
        fps = int(1 / (current_time - previous_time))
        previous_time = current_time

        img = pose_detector.flip_source(img, 1)
        cv2.putText(img, "fps: " + str(fps), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
        cv2.imshow("pose detection", img)
        
        key_code = cv2.waitKey(30)
        if key_code == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()