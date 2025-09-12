import cv2
import mediapipe as mp
import numpy as np
import time

def match_images(runtime_image, file_image_path):
    # Load the file image
    file_image = cv2.imread(file_image_path)

    # Convert both images to grayscale for simplicity
    runtime_gray = cv2.cvtColor(runtime_image, cv2.COLOR_BGR2GRAY)
    file_gray = cv2.cvtColor(file_image, cv2.COLOR_BGR2GRAY)

    # Compare the two images
    if np.array_equal(runtime_gray, file_gray):
        print("Images match")
    else:
        print("Images not match")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

# Initialize a set to store unique movement conditions for all frames
all_frames_conditions = set()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    start = time.time()

    # Flip the image horizontally for a letter selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = hands.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color specs from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    
    # Initialize the set to store unique movement conditions for the current frame
    movement_conditions = set()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                drawing_specs, drawing_specs)

            # Define 3D object points for a simplified hand model (e.g., wrist, index finger MCP, pinky finger MCP)
            # These are arbitrary points in a local coordinate system for the hand
            # You might need to adjust these based on a more accurate hand model
            # For simplicity, let's use the wrist as origin (0,0,0) and define other points relative to it.
            # This is a conceptual representation; actual values might need calibration.
            
            # Object points (3D coordinates in a local hand coordinate system)
            # Using approximate relative positions for a generic hand
            object_points_3d = np.array([
                [0.0, 0.0, 0.0],  # WRIST
                [-0.1, -0.5, 0.0], # INDEX_FINGER_MCP (relative to wrist)
                [0.1, -0.5, 0.0],  # PINKY_MCP (relative to wrist)
                [0.0, -1.0, 0.0]   # MIDDLE_FINGER_TIP (relative to wrist, for a "forward" direction)
            ], dtype=np.float64)

            # Image points (2D coordinates from MediaPipe landmarks)
            image_points_2d = []
            for idx, lm in enumerate(hand_landmarks.landmark):
                if idx == mp_hands.HandLandmark.WRIST:
                    image_points_2d.append([lm.x * img_w, lm.y * img_h])
                elif idx == mp_hands.HandLandmark.INDEX_FINGER_MCP:
                    image_points_2d.append([lm.x * img_w, lm.y * img_h])
                elif idx == mp_hands.HandLandmark.PINKY_MCP:
                    image_points_2d.append([lm.x * img_w, lm.y * img_h])
                elif idx == mp_hands.HandLandmark.MIDDLE_FINGER_TIP:
                    image_points_2d.append([lm.x * img_w, lm.y * img_h])
            
            image_points_2d = np.array(image_points_2d, dtype=np.float64)

            # Camera matrix (assuming a simple pinhole camera model)
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]], dtype=np.float64)

            # Distance matrix (assuming no lens distortion)
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP to get rotation and translation vectors
            if len(image_points_2d) == 4: # Ensure we have enough points
                success, rot_vec, trans_vec = cv2.solvePnP(object_points_3d, image_points_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_SQPNP)

                if success:
                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles (Euler angles from rotation matrix)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Get the roll, pitch, yaw angles in degrees
                    # Note: The interpretation of these angles can vary.
                    # For consistency with the original face code, we'll use angles[0], angles[1], angles[2]
                    roll = angles[2] * 360 # Z-axis rotation
                    pitch = angles[0] * 360 # X-axis rotation
                    yaw = angles[1] * 360 # Y-axis rotation

                    # Add the condition for hand movement to the set for the current frame
                    if yaw < -15: # Adjusted thresholds for hand movements
                        movement_conditions.add("Hand Tilted Left")
                    elif yaw > 15:
                        movement_conditions.add("Hand Tilted Right")
                    elif pitch > 15:
                        movement_conditions.add("Hand Up")
                    elif pitch < -15:
                        movement_conditions.add("Hand Down")
                    else:
                        movement_conditions.add("Hand Forward")

                    # Draw 3D axis gizmo on the hand
                    # Use the wrist landmark as the origin for the gizmo
                    wrist_lm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    gizmo_origin = np.array([wrist_lm.x * img_w, wrist_lm.y * img_h, 0.0], dtype="double")
                    
                    # Draw the 3D axis gizmo
                    gizmo_size = 50
                    cv2.drawFrameAxes(image, cam_matrix, dist_matrix, rot_vec, trans_vec, gizmo_size)

                    # Display the angles
                    cv2.putText(image, "Current Frame", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(image, f"Roll: {np.round(roll, 2)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(image, f"Pitch: {np.round(pitch, 2)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(image, f"Yaw: {np.round(yaw, 2)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        end = time.time()
        totalTime = end - start

        fps = 0
        if totalTime > 0:
            fps = 1 / totalTime
            # print("FPS:", fps)
        # else:
            # print("Unable to calculate FPS. totalTime is zero.")

        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Add the unique movement conditions for the current frame to the set for all frames
        if movement_conditions: # Only add if conditions were detected
            all_frames_conditions.add(frozenset(movement_conditions))

    # Display the unique movement conditions for all frames on the image
    frame_conditions_text = "Frame Conditions: " + ", ".join(str(cond) for cond in movement_conditions)
    cv2.putText(image, frame_conditions_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Hand Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        runtime_image = image.copy()
        break

file_image_path = 'picture.png' # Placeholder, user needs to provide this
# Print the unique movement conditions for all frames
print("Unique movement conditions for all frames:")
for frame_conditions in all_frames_conditions:
    print(frame_conditions)

cap.release()
cv2.destroyAllWindows()

if len(all_frames_conditions) >= 3:
    print("Hand estimation passed")
    # match_images(runtime_image, file_image_path) # Uncomment if file_image_path is valid
else:
    print("Hand estimation failed")
