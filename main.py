import cv2
import mediapipe as mp
from inference.inference_mediapipe import HandTracker
import time
import sys
import csv
import numpy as np
import os
import shutil
import subprocess
from includes.menu import run_interactive_menu

VIDEO_PATH = "videos/test_videos/yaw.mov"  # !CHANGETHIS
OUTPUT_CSV = "videos/output_index_finger.csv"  # !CHANGETHIS
OUTPUT_VIDEO = "videos/output_trajectory.mp4"  # !CHANGETHIS

def draw_gizmo(frame, cx, cy, roll, pitch, yaw, size=40):
    """Draw 3D orientation gizmo on frame."""
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # yaw-pitch-roll order
    R = Rz @ Ry @ Rx

    forward_axis = np.array([1, 0, 0]) * size
    right_axis = np.array([0, 1, 0]) * size
    up_axis = np.array([0, 0, 1]) * size

    axes = np.array([forward_axis, right_axis, up_axis])
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red for Roll, Green for Pitch, Blue for Yaw
    labels = ['Roll', 'Pitch', 'Yaw']

    for i, (color, label) in enumerate(zip(colors, labels)):
        rotated_axis = R @ axes[i]
        end_point = (int(cx + rotated_axis[0]), int(cy - rotated_axis[1]))
        cv2.arrowedLine(frame, (int(cx), int(cy)), end_point, color, 3, tipLength=0.2)
        label_pos = (end_point[0] + 10, end_point[1])
        cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

def extract_frames_with_ffmpeg(video_path, output_dir, frame_interval=1):
    """Extract frames from video using FFmpeg at specified intervals."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    ffmpeg_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'select=not(mod(n\\,{frame_interval}))',
        '-vsync', 'vfr',
        '-q:v', '2',  
        f'{output_dir}/frame_%06d.jpg'
    ]

    print(f"Extracting frames with interval {frame_interval}...")
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        print("Frame extraction completed.")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        return []

    # Get list of extracted frames
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    frame_paths = [os.path.join(output_dir, f) for f in frame_files]

    print(f"Extracted {len(frame_paths)} frames")
    return frame_paths

def process_video_frames(video_path, output_csv, output_video, handle_back_of_hand, flip_back_angles, frame_interval):
    """Process extracted frames from a video, apply hand tracking, and generate output video and CSV."""
    temp_dir = "temp_frames"
    trajectory_points = []
    max_trajectory_length = 50

    try:
        frame_paths = extract_frames_with_ffmpeg(video_path, temp_dir, frame_interval)
        if not frame_paths:
            print("No frames extracted. Check FFmpeg installation and video path.")
            return

        tracker = HandTracker(handle_back_of_hand=handle_back_of_hand, flip_back_angles=flip_back_angles)
        mp_drawing = mp.solutions.drawing_utils

        cap_temp = cv2.VideoCapture(video_path)
        fps = int(cap_temp.get(cv2.CAP_PROP_FPS))
        width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_original_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_temp.release()

        print(f"Processing {len(frame_paths)} frames from video: {width}x{height}, original {fps} fps")

        output_fps = max(1, fps // frame_interval)
        out = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, output_fps, (width, height))

        with open(output_csv, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["frame", "x", "y", "roll", "pitch", "yaw"])

            for frame_idx_processed, frame_path in enumerate(frame_paths):
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = tracker.hands.process(rgb_frame)
                h, w, _ = frame.shape
                roll_text = pitch_text = yaw_text = ""

                # Calculate the original frame number for CSV
                original_frame_number = frame_idx_processed * frame_interval

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            tracker.mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )

                        index_finger_tip = hand_landmarks.landmark[tracker.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                        wrist = hand_landmarks.landmark[tracker.mp_hands.HandLandmark.WRIST]
                        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

                        trajectory_points.append((wrist_x, wrist_y))
                        if len(trajectory_points) > max_trajectory_length:
                            trajectory_points.pop(0)

                        if len(trajectory_points) > 1:
                            for i in range(1, len(trajectory_points)):
                                cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 255, 255), 2)

                        rotation_matrix, wrist_3d_coords, roll, pitch, yaw, is_back_facing = tracker._get_hand_orientation(hand_landmarks, w, h)

                        roll, pitch, yaw = tracker._handle_back_of_hand_adjustment(roll, pitch, yaw, is_back_facing)

                        temp_roll = roll
                        roll = yaw
                        yaw = temp_roll

                        roll_deg = np.degrees(roll)
                        pitch_deg = np.degrees(pitch)
                        yaw_deg = np.degrees(yaw)
                        csv_writer.writerow([
                            original_frame_number, cx, cy, roll_deg, pitch_deg, yaw_deg
                        ])

                        draw_gizmo(frame, wrist_x, wrist_y, roll_deg, pitch_deg, yaw_deg, size=40)
                        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                        roll_text = f"roll: {roll_deg:.2f}"
                        pitch_text = f"pitch: {pitch_deg:.2f}"
                        yaw_text = f"yaw: {yaw_deg:.2f}"
                else:
                    trajectory_points.clear()

                text = f"frame: {original_frame_number}/{total_original_frames} (interval: {frame_interval})"
                cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)

                guide_x = 250
                if roll_text:
                    cv2.putText(frame, roll_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.ellipse(frame, (guide_x, 100), (30, 30), 0, -90, -90 + roll_deg, (0, 0, 255), 2)
                    cv2.line(frame, (guide_x, 100), (guide_x + 30, 100), (0, 0, 255), 1)
                if pitch_text:
                    cv2.putText(frame, pitch_text, (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.ellipse(frame, (guide_x, 140), (30, 30), 90, -90, -90 + pitch_deg, (0, 255, 0), 2)
                    cv2.line(frame, (guide_x, 140), (guide_x, 110), (0, 255, 0), 1)
                if yaw_text:
                    cv2.putText(frame, yaw_text, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.ellipse(frame, (guide_x, 180), (30, 30), 0, 0, yaw_deg, (255, 0, 0), 2)
                    cv2.line(frame, (guide_x, 180), (guide_x + 30, 180), (255, 0, 0), 1)

                if out:
                    out.write(frame)

                if frame_idx_processed % 10 == 0:
                    progress = (frame_idx_processed / len(frame_paths)) * 100
                    print(f"Processing... {frame_idx_processed}/{len(frame_paths)} frames ({progress:.1f}%)")

        if out:
            out.release()
        tracker.release()
        print(f"Done. CSV saved to {output_csv}")
        if output_video:
            print(f"Output video saved to {output_video}")

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("Temporary frames cleaned up.")

def process_camera(output_csv, handle_back_of_hand=True, flip_back_angles=True):
    """Process real-time camera feed for hand tracking."""
    cap = cv2.VideoCapture(0)  # !CHANGETHIS
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    tracker = HandTracker(handle_back_of_hand=handle_back_of_hand, flip_back_angles=flip_back_angles)
    mp_drawing = mp.solutions.drawing_utils

    trajectory_points = []
    max_trajectory_length = 100

    with open(output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["frame", "x", "y", "roll", "pitch", "yaw"])
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                break

            frame = cv2.flip(frame, 1)  # mirror
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = tracker.hands.process(rgb_frame)
            h, w, _ = frame.shape
            roll_text = pitch_text = yaw_text = ""

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        tracker.mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

                    index_finger_tip = hand_landmarks.landmark[tracker.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                    wrist = hand_landmarks.landmark[tracker.mp_hands.HandLandmark.WRIST]
                    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

                    trajectory_points.append((wrist_x, wrist_y))
                    if len(trajectory_points) > max_trajectory_length:
                        trajectory_points.pop(0)

                    if len(trajectory_points) > 1:
                        for i in range(1, len(trajectory_points)):
                            cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 255, 255), 3)

                    rotation_matrix, wrist_3d_coords, roll, pitch, yaw, is_back_facing = tracker._get_hand_orientation(hand_landmarks, w, h)

                    temp_roll = roll
                    roll = yaw
                    yaw = temp_roll

                    roll_deg = np.degrees(roll)
                    pitch_deg = np.degrees(pitch)
                    yaw_deg = np.degrees(yaw)
                    csv_writer.writerow([
                        frame_idx, cx, cy, roll_deg, pitch_deg, yaw_deg
                    ])

                    draw_gizmo(frame, wrist_x, wrist_y, roll_deg, pitch_deg, yaw_deg, size=40)
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                    roll_text = f"roll: {roll_deg:.2f}"
                    pitch_text = f"pitch: {pitch_deg:.2f}"
                    yaw_text = f"yaw: {yaw_deg:.2f}"
            else:
                if len(trajectory_points) > 10:
                    trajectory_points = trajectory_points[5:]
                elif len(trajectory_points) > 0:
                    trajectory_points.clear()

            text = f"frame: {frame_idx}"
            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA)

            guide_x = 250
            if roll_text:
                cv2.putText(frame, roll_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.ellipse(frame, (guide_x, 100), (30, 30), 0, -90, -90 + roll_deg, (0, 0, 255), 2)
                cv2.line(frame, (guide_x, 100), (guide_x + 30, 100), (0, 0, 255), 1)
            if pitch_text:
                cv2.putText(frame, pitch_text, (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.ellipse(frame, (guide_x, 140), (30, 30), 90, -90, -90 + pitch_deg, (0, 255, 0), 2)
                cv2.line(frame, (guide_x, 140), (guide_x, 110), (0, 255, 0), 1)
            if yaw_text:
                cv2.putText(frame, yaw_text, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.ellipse(frame, (guide_x, 180), (30, 30), 0, 0, yaw_deg, (255, 0, 0), 2)
                cv2.line(frame, (guide_x, 180), (guide_x + 30, 180), (255, 0, 0), 1)

            cv2.imshow('Hand Tracking - Camera Live', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                if results.multi_hand_landmarks:
                    tracker.calibrate_neutral_pose(results.multi_hand_landmarks[0])
                    print(f"Calibrated Neutral Rotation Matrix:\n{tracker.neutral_rotation_matrix}")
            elif key == ord('r'):
                trajectory_points.clear()
                print("Trajectory reset")

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()
    print(f"Done. CSV saved to {output_csv}")

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        video_path_arg = sys.argv[1]
        output_csv_arg = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_CSV
        output_video_arg = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_VIDEO
        sample_every_arg = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        process_video_frames(video_path_arg, output_csv_arg, output_video_arg, True, True, sample_every_arg)
        return

    # Interactive menu mode
    run_interactive_menu(process_camera, process_video_frames, VIDEO_PATH, OUTPUT_CSV, OUTPUT_VIDEO)

if __name__ == "__main__":
    main()