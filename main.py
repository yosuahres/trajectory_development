import cv2
import mediapipe as mp
from inference.inference_mediapipe import HandTracker

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
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
