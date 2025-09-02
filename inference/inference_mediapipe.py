import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_points = [] # To store (x, y, roll, pitch, yaw)

    def _get_hand_orientation(self, hand_landmarks, w, h):
        # Get coordinates for wrist, index finger MCP, and pinky finger MCP
        # These points define a plane for orientation calculation
        wrist = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * w,
                          hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * h,
                          hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].z * w])
        
        index_mcp = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].x * w,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y * h,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].z * w])
        
        pinky_mcp = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].x * w,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].y * h,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].z * w])

        # Calculate vectors
        vec_x = index_mcp - wrist
        vec_y = pinky_mcp - wrist
        
        # Create a normal vector to the hand plane
        normal = np.cross(vec_x, vec_y)
        normal = normal / np.linalg.norm(normal)

        # Define a reference vector (e.g., pointing upwards in screen coordinates)
        ref_up = np.array([0, -1, 0]) # Y-axis up in image coordinates

        # Calculate pitch (rotation around X-axis)
        pitch = np.arctan2(normal[1], normal[2]) # Using Y and Z components of normal
        
        # Calculate roll (rotation around Z-axis)
        roll = np.arctan2(normal[0], normal[1]) # Using X and Y components of normal

        # Calculate yaw (rotation around Y-axis) - more complex, often requires a consistent "forward" vector
        # For simplicity, we'll use the projection of the normal onto the XY plane relative to a reference
        yaw = np.arctan2(normal[0], normal[2]) # Using X and Z components of normal

        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

    def _normalize_angle(self, angle, min_val, max_val):
        # Normalize angle to a 0-1 range and clamp it
        normalized = (angle - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)

    def _blend_colors(self, roll_norm, pitch_norm, yaw_norm):
        # Initialize BGR components
        b, g, r = 0.0, 0.0, 0.0

        # Calculate contributions for Red and Green channels
        # Red component gets contribution from Roll and Pitch (Yellow)
        r_contrib = roll_norm + pitch_norm
        # Green component gets contribution from Yaw and Pitch (Yellow)
        g_contrib = yaw_norm + pitch_norm
        # Blue component is always 0 for these colors
        b_contrib = 0.0

        # Scale contributions to 0-255 range
        r = r_contrib * 255
        g = g_contrib * 255
        b = b_contrib * 255

        # Find the maximum component value to normalize
        # This ensures that the brightest component is 255, maintaining intensity
        max_val = max(r, g, b, 1.0) # Use 1.0 to avoid division by zero if all are 0

        if max_val > 255:
            scale = 255.0 / max_val
            r *= scale
            g *= scale
            b *= scale
        
        # Clamp values to 0-255 and convert to int
        b = int(np.clip(b, 0, 255))
        g = int(np.clip(g, 0, 255))
        r = int(np.clip(r, 0, 255))
        
        return (b, g, r)

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

                # Calculate hand orientation
                roll, pitch, yaw = self._get_hand_orientation(hand_landmarks, w, h)

                # Normalize roll, pitch, yaw for color blending (adjust min/max based on expected range)
                # These ranges are estimates and might need tuning
                roll_norm = self._normalize_angle(roll, -90, 90)
                pitch_norm = self._normalize_angle(pitch, -90, 90)
                yaw_norm = self._normalize_angle(yaw, -90, 90)

                # Store point with orientation data
                self.drawing_points.append((cx, cy, roll_norm, pitch_norm, yaw_norm))
                
                # Draw a circle at the index finger tip with blended color
                current_color = self._blend_colors(roll_norm, pitch_norm, yaw_norm)
                cv2.circle(frame, (cx, cy), 5, current_color, cv2.FILLED)

        # Draw lines connecting the stored points with blended colors
        if len(self.drawing_points) > 1:
            for i in range(1, len(self.drawing_points)):
                p1_coords = (self.drawing_points[i-1][0], self.drawing_points[i-1][1])
                p2_coords = (self.drawing_points[i][0], self.drawing_points[i][1])
                
                # Blend color for the line segment based on the current point's orientation
                line_color = self._blend_colors(self.drawing_points[i][2], self.drawing_points[i][3], self.drawing_points[i][4])
                cv2.line(frame, p1_coords, p2_coords, line_color, 2)

        return frame

    def release(self):
        self.hands.close()
