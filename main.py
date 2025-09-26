import cv2
import mediapipe as mp
from inference.inference_mediapipe import HandTracker

VIDEO_PATH = "videos/ancuk.mov" #!CHANGETHIS
OUTPUT_CSV = "videos/output_index_finger.csv" #!CHANGETHIS
OUTPUT_VIDEO = "videos/output_trajectory.mp4" #!CHANGETHIS

import sys
import csv
import numpy as np
import os
import shutil
import subprocess

def draw_gizmo(frame, cx, cy, roll, pitch, yaw, size=40):
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
        # labels = ['Forward', 'Right', 'Up']

    for i, (color, label) in enumerate(zip(colors, labels)):
        rotated_axis = R @ axes[i]
        end_point = (int(cx + rotated_axis[0]), int(cy - rotated_axis[1])) 
        cv2.arrowedLine(frame, (int(cx), int(cy)), end_point, color, 3, tipLength=0.2)
        label_pos = (end_point[0] + 10, end_point[1])
        cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

def process_video(video_path, output_csv, output_video=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    tracker = HandTracker()
    mp_drawing = mp.solutions.drawing_utils
    
    # Get video properties for output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Initialize video writer if output video is specified
    out = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    with open(output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["frame", "x", "y", "roll", "pitch", "yaw"])
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = tracker.hands.process(rgb_frame)
            h, w, _ = frame.shape
            roll_text = pitch_text = yaw_text = ""
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks and connections
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        tracker.mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    index_finger_tip = hand_landmarks.landmark[tracker.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                    rotation_matrix, wrist_3d_coords, roll, pitch, yaw = tracker._get_hand_orientation(hand_landmarks, w, h)
                    
                    # Swap roll and yaw variables
                    temp_roll = roll
                    roll = yaw
                    yaw = temp_roll

                    roll_deg = np.degrees(roll)
                    pitch_deg = np.degrees(pitch)
                    yaw_deg = np.degrees(yaw)
                    csv_writer.writerow([
                        frame_idx, cx, cy, roll_deg, pitch_deg, yaw_deg
                    ])
                    
                    # Draw only the current gizmo
                    draw_gizmo(frame, cx, cy, roll_deg, pitch_deg, yaw_deg, size=40)
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                    roll_text = f"roll: {roll_deg:.2f}"
                    pitch_text = f"pitch: {pitch_deg:.2f}"
                    yaw_text = f"yaw: {yaw_deg:.2f}"
            
            text = f"frame: {frame_idx}/{total_frames}"
            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA)
            
            # Draw angle guides
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

            # Write frame to output video if specified
            if out:
                out.write(frame)
            
            # Display frame
            cv2.imshow('Hand Tracking - Video Processing', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Processing stopped by user.")
                break
            
            frame_idx += 1
            
            # Print progress every 30 frames
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Processing... {frame_idx}/{total_frames} frames ({progress:.1f}%)")

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    tracker.release()
    print(f"Done. CSV saved to {output_csv}")
    if output_video:
        print(f"Output video saved to {output_video}")

def process_camera(output_csv):
    cap = cv2.VideoCapture(0)  #!CHANGETHIS
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    tracker = HandTracker()
    # Add MediaPipe drawing utilities
    mp_drawing = mp.solutions.drawing_utils

    with open(output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["frame", "x", "y", "roll", "pitch", "yaw"])
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                break

            frame = cv2.flip(frame, 1) # mirror
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = tracker.hands.process(rgb_frame)
            h, w, _ = frame.shape
            roll_text = pitch_text = yaw_text = ""
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks and connections
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        tracker.mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    index_finger_tip = hand_landmarks.landmark[tracker.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                    rotation_matrix, wrist_3d_coords, roll, pitch, yaw = tracker._get_hand_orientation(hand_landmarks, w, h)
                    
                    # Swap roll and yaw variables
                    temp_roll = roll
                    roll = yaw
                    yaw = temp_roll

                    roll_deg = np.degrees(roll)
                    pitch_deg = np.degrees(pitch)
                    yaw_deg = np.degrees(yaw)
                    csv_writer.writerow([
                        frame_idx, cx, cy, roll_deg, pitch_deg, yaw_deg
                    ])
                    
                    # Draw only the current gizmo
                    draw_gizmo(frame, cx, cy, roll_deg, pitch_deg, yaw_deg, size=40)
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                    roll_text = f"roll: {roll_deg:.2f}"
                    pitch_text = f"pitch: {pitch_deg:.2f}"
                    yaw_text = f"yaw: {yaw_deg:.2f}"
            
            text = f"frame: {frame_idx}"
            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA)
            
            # Draw angle guides
            guide_x = 250
            if roll_text:
                # Roll text and ellipse should be Red (0, 0, 255)
                cv2.putText(frame, roll_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.ellipse(frame, (guide_x, 100), (30, 30), 0, -90, -90 + roll_deg, (0, 0, 255), 2)
                cv2.line(frame, (guide_x, 100), (guide_x + 30, 100), (0, 0, 255), 1) 
            if pitch_text:
                # Pitch text and ellipse should be Green (0, 255, 0)
                cv2.putText(frame, pitch_text, (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.ellipse(frame, (guide_x, 140), (30, 30), 90, -90, -90 + pitch_deg, (0, 255, 0), 2)
                cv2.line(frame, (guide_x, 140), (guide_x, 110), (0, 255, 0), 1) 
            if yaw_text:
                # Yaw text and ellipse should be Blue (255, 0, 0)
                cv2.putText(frame, yaw_text, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.ellipse(frame, (guide_x, 180), (30, 30), 0, 0, yaw_deg, (255, 0, 0), 2)
                cv2.line(frame, (guide_x, 180), (guide_x + 30, 180), (255, 0, 0), 1)  

            cv2.imshow('Hand Tracking - Camera Live', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'): # 'c' for calibrate
                if results.multi_hand_landmarks:
                    tracker.calibrate_neutral_pose(results.multi_hand_landmarks[0])
                    print(f"Calibrated Neutral Rotation Matrix:\n{tracker.neutral_rotation_matrix}")
            
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()
    print(f"Done. CSV saved to {output_csv}")

def show_menu():
    print("\n" + "="*50)
    print("="*50)
    print("option")
    print("1. real time cam)")
    print("2. path video file")
    print("3. exit")
    print("="*50)

def main():
    while True:
        show_menu()
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == '1':
                print("\ncamera")
                print("-" * 30)
                process_camera(OUTPUT_CSV)
                
            elif choice == '2':
                if not os.path.exists(VIDEO_PATH):
                    print(f"\nError: Video file '{VIDEO_PATH}' not found!")
                    continue
                    
                print("-" * 30)
                process_video(VIDEO_PATH, OUTPUT_CSV, OUTPUT_VIDEO)
                
            elif choice == '3':
                print("\nbad")
                break
                
            else:
                print("\ninvalid bad")
                
        except KeyboardInterrupt:
            print("\n\nProgram interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
