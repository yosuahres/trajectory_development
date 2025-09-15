import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.7, roll_change_threshold=0.02, smoothing_window_size=5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_points = []
        self.previous_roll_norm = None
        self.roll_change_threshold = roll_change_threshold
        self.roll_history = [] 
        self.smoothing_window_size = 5

    def _get_hand_orientation(self, hand_landmarks, w, h):
        points = np.asarray([
            [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z],
            [hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z],
            [hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z]
        ])
        normal = np.cross(points[2] - points[0], points[1] - points[2])
        normal = normal / np.linalg.norm(normal) 

        ref_up = np.array([0, -1, 0])

        pitch = np.arctan2(normal[1], normal[2])
        roll = np.arctan2(normal[0], normal[1])
        
        yaw = np.arctan2(normal[0], normal[2]) 
        
        hand_x_axis = points[1] - points[0]
        hand_x_axis = hand_x_axis / np.linalg.norm(hand_x_axis)        
        hand_z_axis = normal
        hand_y_axis = np.cross(hand_z_axis, hand_x_axis)
        hand_y_axis = hand_y_axis / np.linalg.norm(hand_y_axis) 
        
        rotation_matrix = np.array([hand_x_axis, hand_y_axis, hand_z_axis]).T
        rvec, _ = cv2.Rodrigues(rotation_matrix)
        return rvec, points[0], roll, pitch, yaw

    def _normalize_angle(self, angle, min_val, max_val):
        normalized = (angle - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)

    def _blend_colors(self, roll_norm):
        b = int(255 * (1 - roll_norm)) 
        r = int(255 * roll_norm)       
        g = 0                          
        return (b, g, r)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        h, w, c = frame.shape
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                rvec, wrist_3d_coords, roll, pitch, yaw = self._get_hand_orientation(hand_landmarks, w, h)

                tvec = np.array([wrist_3d_coords[0] * w, wrist_3d_coords[1] * h, wrist_3d_coords[2] * w], dtype="double")

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"roll: {np.degrees(roll):.2f}", (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)   
                cv2.putText(frame, f"pitch: {np.degrees(pitch):.2f}", (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"yaw: {np.degrees(yaw):.2f}", (10, 110), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                self.drawing_points.append((cx, cy, roll, pitch, yaw))

                offset = 4 
                for i in range(1, len(self.drawing_points)):
                    x0, y0, roll0, pitch0, yaw0 = self.drawing_points[i-1]
                    x1, y1, roll1, pitch1, yaw1 = self.drawing_points[i]
                  
                    cv2.line(frame, (x0 - offset, y0 - offset), (x1 - offset, y1 - offset), (0, 0, 255), 2)
                    cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                  
                    cv2.line(frame, (x0 + offset, y0 + offset), (x1 + offset, y1 + offset), (255, 0, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        return frame

    def clear_trajectory(self):
        self.drawing_points = []
        self.previous_roll_norm = None 

    def release(self):
        self.hands.close()


