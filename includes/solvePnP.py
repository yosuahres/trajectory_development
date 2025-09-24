import cv2
import numpy as np

def get_pose_estimation(image_points, object_points, camera_matrix, dist_coeffs):
    """
    Calculates the pose (rotation and translation) of an object from 2D-3D point correspondences
    and then derives roll, pitch, and yaw angles.

    Args:
        image_points (np.array): 2D points from the image (e.g., detected landmarks).
                                 Shape: (N, 2) or (N, 1, 2)
        object_points (np.array): 3D points of the object in its own coordinate system.
                                  Shape: (N, 3) or (N, 1, 3)
        camera_matrix (np.array): Intrinsic camera matrix.
                                  Shape: (3, 3)
        dist_coeffs (np.array): Distortion coefficients.
                                Shape: (4,) or (5,) or (8,)

    Returns:
        tuple: A tuple containing:
            - rvec (np.array): Rotation vector (3x1).
            - tvec (np.array): Translation vector (3x1).
            - roll (float): Roll angle in degrees.
            - pitch (float): Pitch angle in degrees.
            - yaw (float): Yaw angle in degrees.
    """
    # Ensure points are in the correct format for solvePnP
    if image_points.shape[1] == 2:
        image_points = image_points.reshape(-1, 1, 2).astype(np.float32)
    if object_points.shape[1] == 3:
        object_points = object_points.reshape(-1, 1, 3).astype(np.float32)

    # Solve for pose
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        print("solvePnP failed to find a solution.")
        return None, None, None, None, None

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Extract Euler angles (roll, pitch, yaw) from the rotation matrix
    # This conversion can be complex and depends on the chosen convention (e.g., ZYX, XYZ)
    # Here, we'll use a common approach for ZYX Euler angles (yaw, pitch, roll)
    # based on https://www.learnopencv.com/rotation-matrix-to-euler-angles/

    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2]) # Roll
        y = np.arctan2(-R[2,0], sy)    # Pitch
        z = np.arctan2(R[1,0], R[0,0]) # Yaw
    else:
        x = np.arctan2(-R[1,2], R[1,1]) # Roll
        y = np.arctan2(-R[2,0], sy)    # Pitch
        z = 0                          # Yaw is zero

    roll = np.degrees(x)
    pitch = np.degrees(y)
    yaw = np.degrees(z)

    return rvec, tvec, roll, pitch, yaw

if __name__ == '__main__':
    # Example Usage (dummy data)
    print("Running example usage for solvePnP.py")

    # 3D model points (e.g., a simple cube or a face model)
    # These points define the object in its own coordinate system
    object_points = np.array([
        (0.0, 0.0, 0.0),    # Nose tip
        (0.0, -330.0, -65.0), # Chin
        (-225.0, 170.0, -135.0), # Left eye corner
        (225.0, 170.0, -135.0),  # Right eye corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype=np.float32)

    image_points = np.array([
        (359, 391),  # Nose tip
        (399, 561),  # Chin
        (263, 345),  # Left eye corner
        (451, 345),  # Right eye corner
        (301, 491),  # Left mouth corner
        (417, 491)   # Right mouth corner
    ], dtype=np.float32)

    # Dummy Camera Matrix (replace with your actual camera intrinsics)
    # fx, fy: focal lengths
    # cx, cy: principal point
    focal_length = 1 * image_points.shape[0] # Example: use image height as focal length
    center = (image_points.shape[1]/2, image_points.shape[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Dummy Distortion Coefficients (replace with your actual camera distortion)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32) # Assuming no distortion

    rvec, tvec, roll, pitch, yaw = get_pose_estimation(image_points, object_points, camera_matrix, dist_coeffs)

    if rvec is not None:
        print("\n--- Pose Estimation Results ---")
        print(f"Rotation Vector (rvec):\n{rvec}")
        print(f"Translation Vector (tvec):\n{tvec}")
        print(f"Roll: {roll:.2f} degrees")
        print(f"Pitch: {pitch:.2f} degrees")
        print(f"Yaw: {yaw:.2f} degrees")
    else:
        print("Pose estimation failed.")
