import cv2
import mediapipe as mp
from inference.inference_mediapipe import HandTracker

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    tracker = HandTracker()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break
        frame = tracker.process_frame(frame)
        cv2.imshow('Hand Tracking', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press 'Esc' to exit
            break
        elif key == ord('g'): # Press 'g' to clear trajectory
            tracker.clear_trajectory()
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
