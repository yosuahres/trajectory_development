import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_points = [] # To store index finger tip coordinates

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        h, w, c = frame.shape # Get frame dimensions
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get index finger tip coordinates (landmark 8)
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                self.drawing_points.append((cx, cy))
                
                # Draw a circle at the index finger tip
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Draw lines connecting the stored points
        if len(self.drawing_points) > 1:
            for i in range(1, len(self.drawing_points)):
                cv2.line(frame, self.drawing_points[i-1], self.drawing_points[i], (255, 0, 255), 2)

        return frame

    def release(self):
        self.hands.close()
