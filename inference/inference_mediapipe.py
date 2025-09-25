import cv2
import mediapipe as mp
import numpy as np
from includes.filter import AverageFilter

class HandTracker:
    def __init__(self, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.9, roll_change_threshold=0.02, smoothing_window_size=5, frame_skip=1): #!CHANGETHIS
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
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.neutral_rotation_matrix = None 

        # init filter
        filter_window_size = smoothing_window_size 
        self.roll_filter = AverageFilter(filter_window_size)
        self.pitch_filter = AverageFilter(filter_window_size)
        self.yaw_filter = AverageFilter(filter_window_size)
        self.x_filter = AverageFilter(filter_window_size)
        self.y_filter = AverageFilter(filter_window_size)

    def calibrate_neutral_pose(self, hand_landmarks):
        # Capture the current hand orientation as the neutral pose
        wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
        mcp_index = np.array([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z])
        mcp_pinky = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z])
        middle_mcp = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z])
        
        hand_forward = middle_mcp - wrist
        hand_forward = hand_forward / np.linalg.norm(hand_forward)
        
        hand_right = mcp_index - mcp_pinky
        hand_right = hand_right / np.linalg.norm(hand_right)
        
        hand_up = np.cross(hand_forward, hand_right)
        hand_up = hand_up / np.linalg.norm(hand_up)
        
        hand_right = np.cross(hand_up, hand_forward)
        hand_right = hand_right / np.linalg.norm(hand_right)
        
        self.neutral_rotation_matrix = np.array([hand_forward, hand_right, hand_up]).T
        print("Neutral pose calibrated.")

    def _get_hand_orientation(self, hand_landmarks, w, h):
        wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
        mcp_index = np.array([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z])
        mcp_pinky = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z])
        middle_mcp = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z])
        
        # X-axis: from wrist to middle finger MCP (forward direction)
        hand_forward = middle_mcp - wrist
        hand_forward = hand_forward / np.linalg.norm(hand_forward)
        
        # Y-axis: from pinky MCP to index MCP (left-right direction)
        hand_right = mcp_index - mcp_pinky
        hand_right = hand_right / np.linalg.norm(hand_right)
        
        # Z-axis: perpendicular to palm (up-down direction)
        hand_up = np.cross(hand_forward, hand_right)
        hand_up = hand_up / np.linalg.norm(hand_up)
        
        hand_right = np.cross(hand_up, hand_forward)
        hand_right = hand_right / np.linalg.norm(hand_right)
        
        rotation_matrix = np.array([hand_forward, hand_right, hand_up]).T
        # Extract Euler angles (roll, pitch, yaw) from the rotation matrix (ZYX order)
        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        # R = [[cos(yaw)cos(pitch), cos(yaw)sin(pitch)sin(roll) - sin(yaw)cos(roll), cos(yaw)sin(pitch)cos(roll) + sin(yaw)sin(roll)],
        #      [sin(yaw)cos(pitch), sin(yaw)sin(pitch)sin(roll) + cos(yaw)cos(roll), sin(yaw)sin(pitch)cos(roll) - cos(yaw)sin(roll)],
        #      [-sin(pitch), cos(pitch)sin(roll), cos(pitch)cos(roll)]]

        pitch = -np.arcsin(rotation_matrix[2, 0])

        # Handle gimbal lock for pitch near +/- 90 degrees
        if np.abs(np.cos(pitch)) < 1e-6: # Gimbal lock
            roll = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 2])
            yaw = 0.0 
        else:
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        return rotation_matrix, wrist, roll, pitch, yaw 

    def _normalize_angle(self, angle, min_val, max_val):
        normalized = (angle - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)

    def _blend_colors(self, roll_norm):
        b = int(255 * (1 - roll_norm)) 
        r = int(255 * roll_norm)       
        g = 0                          
        return (b, g, r)

    def process_frame(self, frame):
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return frame 

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        h, w, c = frame.shape
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                current_rotation_matrix, wrist_3d_coords, raw_roll, raw_pitch, raw_yaw = self._get_hand_orientation(hand_landmarks, w, h)

                # print(f"Current Rotation Matrix:\n{current_rotation_matrix}")

                # If a neutral pose is calibrated, calculate angles relative to it
                if self.neutral_rotation_matrix is not None:
                    relative_rotation_matrix = np.linalg.inv(self.neutral_rotation_matrix) @ current_rotation_matrix
                    # print(f"Relative Rotation Matrix:\n{relative_rotation_matrix}")
                    
                    # Extract Euler angles from the relative rotation matrix (ZYX order)
                    pitch = -np.arcsin(relative_rotation_matrix[2, 0])
                    if np.abs(np.cos(pitch)) < 1e-6: # Gimbal lock
                        roll = np.arctan2(relative_rotation_matrix[0, 1], relative_rotation_matrix[0, 2])
                        yaw = 0.0
                    else:
                        roll = np.arctan2(relative_rotation_matrix[2, 1], relative_rotation_matrix[2, 2])
                        yaw = np.arctan2(relative_rotation_matrix[1, 0], relative_rotation_matrix[0, 0])
                    
                    # print(f"Relative (raw) Roll: {np.degrees(roll):.2f}, Pitch: {np.degrees(pitch):.2f}, Yaw: {np.degrees(yaw):.2f}")
                else:
                    roll = raw_roll
                    pitch = raw_pitch
                    yaw = raw_yaw
                    # print(f"Absolute (raw) Roll: {np.degrees(roll):.2f}, Pitch: {np.degrees(pitch):.2f}, Yaw: {np.degrees(yaw):.2f}")

                # Apply moving average filter
                roll = self.roll_filter.update(roll)
                pitch = self.pitch_filter.update(pitch)
                yaw = self.yaw_filter.update(yaw)

                tvec = np.array([wrist_3d_coords[0] * w, wrist_3d_coords[1] * h, wrist_3d_coords[2] * w], dtype="double")

                font = cv2.FONT_HERSHEY_SIMPLEX
                roll_desc = "palm down" if np.degrees(roll) > 10 else "palm up" if np.degrees(roll) < -10 else "neutral"
                pitch_desc = "fingers up" if np.degrees(pitch) > 10 else "fingers down" if np.degrees(pitch) < -10 else "level"
                yaw_desc = "pointing right" if np.degrees(yaw) > 10 else "pointing left" if np.degrees(yaw) < -10 else "forward"
                
                cv2.putText(frame, f"Roll: {np.degrees(roll):.1f}° ({roll_desc})", (10, 30), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)   
                cv2.putText(frame, f"Pitch: {np.degrees(pitch):.1f}° ({pitch_desc})", (10, 60), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Yaw: {np.degrees(yaw):.1f}° ({yaw_desc})", (10, 90), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                raw_cx, raw_cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # Apply moving average filter to coordinates
                cx = int(self.x_filter.update(raw_cx))
                cy = int(self.y_filter.update(raw_cy))

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
        self.roll_filter.reset()
        self.pitch_filter.reset()
        self.yaw_filter.reset()
        self.x_filter.reset()
        self.y_filter.reset()

    def release(self):
        self.hands.close()