import cv2
import mediapipe as mp
import argparse
from inference.inference_mediapipe import HandTracker

# how to run
#  python main.py --video "path/to/your/video.mp4"

def main():
    parser = argparse.ArgumentParser(description="Hand Tracking with MediaPipe")
    parser.add_argument("--video", type=str, default=0,
                        help="Path to video file or 0 for webcam (default: 0)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.video}")
        return

    tracker = HandTracker()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break
        frame = tracker.process_frame(frame)
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:  
            break
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
