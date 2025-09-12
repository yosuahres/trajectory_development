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
        self.drawing_points = [] # To store (x, y, roll, pitch, yaw)
        self.previous_roll_norm = None
        self.roll_change_threshold = roll_change_threshold
        self.roll_history = [] # For smoothing roll values
        self.smoothing_window_size = smoothing_window_size

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

        # Define hand's local coordinate system (gizmo axes)
        # Z-axis: normal to the hand plane (already calculated)
        # X-axis: along the index finger MCP vector (vec_x)
        # Y-axis: orthogonal to X and Z (cross product of Z and X)
        
        # Ensure vec_x is normalized for the X-axis
        hand_x_axis = vec_x / np.linalg.norm(vec_x)
        
        # Recalculate normal to be strictly orthogonal to hand_x_axis if needed,
        # or use the existing normal as the Z-axis.
        # For simplicity, let's use the existing normal as Z-axis and derive Y.
        hand_z_axis = normal
        hand_y_axis = np.cross(hand_z_axis, hand_x_axis)
        hand_y_axis = hand_y_axis / np.linalg.norm(hand_y_axis) # Normalize Y

        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw), wrist, hand_x_axis, hand_y_axis, hand_z_axis

    def _normalize_angle(self, angle, min_val, max_val):
        # Normalize angle to a 0-1 range and clamp it
        normalized = (angle - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)

    def _blend_colors(self, roll_norm):
        # Use roll_norm (0-1) to create a color gradient, e.g., blue to red
        # If roll_norm is 0, color is blue (255, 0, 0)
        # If roll_norm is 1, color is red (0, 0, 255)
        # If roll_norm is 0.5, color is purple (127, 0, 127)

        b = int(255 * (1 - roll_norm)) # Blue decreases as roll_norm increases
        r = int(255 * roll_norm)       # Red increases as roll_norm increases
        g = 0                          # Green is always 0 for a blue-red gradient

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

                # Calculate hand orientation and get gizmo axes
                roll, pitch, yaw, wrist_coords, x_axis, y_axis, z_axis = self._get_hand_orientation(hand_landmarks, w, h)

                # Normalize roll, pitch, yaw for color blending (adjust min/max based on expected range)
                # These ranges are estimates and might need tuning
                raw_roll_norm = self._normalize_angle(roll, -90, 90)
                pitch_norm = self._normalize_angle(pitch, -90, 90)
                yaw_norm = self._normalize_angle(yaw, -90, 90)

                # Apply smoothing to roll_norm
                self.roll_history.append(raw_roll_norm)
                if len(self.roll_history) > self.smoothing_window_size:
                    self.roll_history.pop(0) # Remove oldest value

                smoothed_roll_norm = np.mean(self.roll_history)

                # Draw the orientation gizmo at the index finger tip
                # Use (cx, cy, 0) as the origin for the gizmo, assuming Z is not directly used for 2D drawing
                # The axes (x_axis, y_axis, z_axis) are still relative to the hand's orientation
                self._draw_gizmo(frame, np.array([cx, cy, 0]), x_axis, y_axis, z_axis)

        # Trajectory drawing is disabled as per user request.
        # self.drawing_points.append(...)
        # cv2.circle(...)
        # cv2.line(...)

        return frame

    def _draw_gizmo(self, frame, origin, x_axis, y_axis, z_axis, scale=50):
        # Convert 3D origin to 2D image coordinates
        origin_2d = (int(origin[0]), int(origin[1]))

        # Define axis colors (BGR)
        x_color = (0, 0, 255) # Red for X
        y_color = (0, 255, 0) # Green for Y
        z_color = (255, 0, 0) # Blue for Z

        # Calculate end points for axes
        x_end = (int(origin[0] + x_axis[0] * scale), int(origin[1] + x_axis[1] * scale))
        y_end = (int(origin[0] + y_axis[0] * scale), int(origin[1] + y_axis[1] * scale))
        z_end = (int(origin[0] + z_axis[0] * scale), int(origin[1] + z_axis[1] * scale))

        # Draw axes
        cv2.line(frame, origin_2d, x_end, x_color, 2)
        cv2.line(frame, origin_2d, y_end, y_color, 2)
        cv2.line(frame, origin_2d, z_end, z_color, 2)

    def clear_trajectory(self):
        self.drawing_points = []
        self.previous_roll_norm = None # Reset previous roll when clearing trajectory

    def release(self):
        self.hands.close()
