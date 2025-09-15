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
        self.smoothing_window_size = smoothing_window_size

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
        return rvec, points[0]

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
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                rvec, _ = self._get_hand_orientation(hand_landmarks, w, h)
                
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                tvec = np.array([index_finger_tip.x * w, index_finger_tip.y * h, 0.0], dtype="double") 
                
                focal_length = w 
                center_x, center_y = w / 2, h / 2
                camera_matrix = np.array([[focal_length, 0, center_x],
                                          [0, focal_length, center_y],
                                          [0, 0, 1]], dtype="double")
                dist_coeffs = np.zeros((4, 1), dtype="double") 
                
                gizmo_size = 50 
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, gizmo_size)
        return frame

    def clear_trajectory(self):
        self.drawing_points = []
        self.previous_roll_norm = None 

    def release(self):
        self.hands.close()
