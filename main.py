import cv2
import mediapipe as mp
from inference.inference_mediapipe import HandTracker

VIDEO_PATH = "videos/test-mediapipe-1.mov"
OUTPUT_CSV = "videos/output_index_finger.csv"
OUTPUT_VIDEO = "videos/output_trajectory.mp4"

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
    R = Rz @ Ry @ Rx
    axes = np.eye(3) * size
    for i, color in enumerate([(0,0,255), (0,255,0), (255,0,0)]):
        axis = R @ axes[:,i]
        end_point = (int(cx + axis[0]), int(cy + axis[1]))
        cv2.arrowedLine(frame, (int(cx), int(cy)), end_point, color, 2, tipLength=0.3)

def process_video(video_path, output_csv, output_video, sample_every=120):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    tracker = HandTracker()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = 2
    temp_dir = "temp_sampled_frames"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    all_sampled_gizmos = []  # Accumulate all sampled gizmos across frames
    with open(output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["frame", "x", "y", "roll", "pitch", "yaw"])
        frame_idx = 0
        saved_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = tracker.hands.process(rgb_frame)
                h, w, _ = frame.shape
                roll_text = pitch_text = yaw_text = ""
                new_gizmos = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        index_finger_tip = hand_landmarks.landmark[tracker.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                        rvec, wrist_3d_coords, roll, pitch, yaw = tracker._get_hand_orientation(hand_landmarks, w, h)
                        roll_deg = np.degrees(roll)
                        pitch_deg = np.degrees(pitch)
                        yaw_deg = np.degrees(yaw)
                        csv_writer.writerow([
                            frame_idx, cx, cy, roll_deg, pitch_deg, yaw_deg
                        ])
                        new_gizmos.append((cx, cy, roll_deg, pitch_deg, yaw_deg))
                        roll_text = f"roll: {roll_deg:.2f}"
                        pitch_text = f"pitch: {pitch_deg:.2f}"
                        yaw_text = f"yaw: {yaw_deg:.2f}"
                # Accumulate all gizmos
                all_sampled_gizmos.extend(new_gizmos)
                # Draw all accumulated gizmos so far
                for giz in all_sampled_gizmos:
                    cx, cy, roll, pitch, yaw = giz
                    draw_gizmo(frame, cx, cy, roll, pitch, yaw, size=40)
                # Draw all accumulated points as dots
                for giz in all_sampled_gizmos:
                    cx, cy, *_ = giz
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                # Add text annotation (frame number and RPY)
                text = f"Frame: {frame_idx}"
                cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA)
                if roll_text:
                    cv2.putText(frame, roll_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                if pitch_text:
                    cv2.putText(frame, pitch_text, (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                if yaw_text:
                    cv2.putText(frame, yaw_text, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                frame_path = os.path.join(temp_dir, f"frame_{saved_idx:05d}.png")
                cv2.imwrite(frame_path, frame)
                saved_idx += 1
            frame_idx += 1
    cap.release()
    tracker.release()
    # Use ffmpeg to create video from sampled frames
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(output_fps), "-i",
        os.path.join(temp_dir, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_video
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Done. CSV saved to {output_csv}, video saved to {output_video}")
    except Exception as e:
        print(f"ffmpeg failed: {e}")
    shutil.rmtree(temp_dir)

def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        output_csv = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_CSV
        output_video = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_VIDEO
        sample_every = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        process_video(video_path, output_csv, output_video, sample_every)
        return
    process_video(VIDEO_PATH, OUTPUT_CSV, OUTPUT_VIDEO, sample_every=5)
    return

if __name__ == "__main__":
    main()
