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

        # Construct rotation matrix from these axes
        # The columns of the rotation matrix are the basis vectors of the new coordinate system
        rotation_matrix = np.array([hand_x_axis, hand_y_axis, hand_z_axis]).T

        # Convert rotation matrix to rotation vector
        rvec, _ = cv2.Rodrigues(rotation_matrix)

        return rvec, wrist # Return rvec and wrist coordinates

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
                # Calculate hand orientation and get rvec
                rvec, _ = self._get_hand_orientation(hand_landmarks, w, h)

                # Get index finger tip coordinates (landmark 8) for translation vector
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                tvec = np.array([index_finger_tip.x * w, index_finger_tip.y * h, 0.0], dtype="double") # Z is 0 for 2D projection

                # Define dummy camera matrix and distortion coefficients
                focal_length = w # A reasonable estimate
                center_x, center_y = w / 2, h / 2
                camera_matrix = np.array([[focal_length, 0, center_x],
                                          [0, focal_length, center_y],
                                          [0, 0, 1]], dtype="double")
                dist_coeffs = np.zeros((4, 1), dtype="double") # No distortion

                # Draw the 3D axis gizmo
                gizmo_size = 50 # As discussed in the plan
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, gizmo_size)

        # Trajectory drawing is disabled as per user request.
        # self.drawing_points.append(...)
        # cv2.circle(...)
        # cv2.line(...)

        return frame

    def clear_trajectory(self):
        self.drawing_points = []
        self.previous_roll_norm = None # Reset previous roll when clearing trajectory

    def release(self):
        self.hands.close()
