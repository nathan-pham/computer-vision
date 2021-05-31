import mediapipe as mp
import cv2

import time
    
class HandTracker():
    def __init__(self, window_name="hand tracking", mode=False, max_hands=2, detection_confidence=0.5, track_confidence=0.5):
        self.window_name = window_name
        self.draw = mp.solutions.drawing_utils
        self.model = mp.solutions.hands
        self.hands = self.model.Hands(mode, max_hands, detection_confidence, track_confidence)

    def create_window(self):
        self.camera = cv2.VideoCapture(0)

        cv2.namedWindow(self.window_name)
        cv2.startWindowThread()

    def find_hands(self, img, draw=True, fps=0):
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        hands = results.multi_hand_landmarks or []

        if draw:
            for hand in hands:
                self.draw.draw_landmarks(img, hand, self.model.HAND_CONNECTIONS)

            if fps > 0:
                cv2.putText(img, "fps: " + str(fps), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

        return hands

    def find_position(self, img, hand, draw=True):
        height, width, channels = img.shape
        landmark_list = []

        for id, landmark in enumerate(hand.landmark):
            cx = int(landmark.x * width)
            cy = int(landmark.y * height)
            landmark_list.append([id, cx, cy])
            
            if draw:
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
    
        return landmark_list

def main():
    hand_tracker = HandTracker()
    previous_time = 0
    current_time = 0

    hand_tracker.create_window()

    while cv2.getWindowProperty(hand_tracker.window_name, 0) >= 0:
        success, img = hand_tracker.camera.read()

        current_time = time.time()
        fps = int(1 / (current_time - previous_time))
        previous_time = current_time

        hands = hand_tracker.find_hands(img, True, fps)
        
        if len(hands) > 0:
            landmark_list = hand_tracker.find_position(img, hands[0])

        cv2.imshow(hand_tracker.window_name, img)
        cv2.waitKey(50)
    
if __name__ == "__main__":
    main()