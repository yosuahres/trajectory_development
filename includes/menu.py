import os
import sys
from .config import configure_back_of_hand, configure_sampling_settings

def show_menu():
    print("\n" + "=" * 50)
    print("=" * 50)
    print("Hand Tracking Options")
    print("1. Real time camera")
    print("2. Process video file")
    print("3. Configure back of hand settings")
    print("4. Configure video sampling interval (for option 2)")
    print("5. Exit")
    print("=" * 50)

def run_interactive_menu(process_camera_func, process_video_func, video_path, output_csv, output_video):
    handle_back_of_hand = True
    flip_back_angles = True
    video_frame_interval = 30 

    while True:
        show_menu()
        try:
            choice = input("Enter your choice (1-5): ").strip()

            if choice == '1':
                print("\nStarting camera...")
                print("-" * 30)
                process_camera_func(output_csv, handle_back_of_hand, flip_back_angles)

            elif choice == '2':
                if not os.path.exists(video_path):
                    print(f"\nError: Video file '{video_path}' not found!")
                    continue
                print(f"Processing video with sampling every {video_frame_interval} frames...")
                print("-" * 30)
                process_video_func(video_path, output_csv, output_video, handle_back_of_hand, flip_back_angles, video_frame_interval)

            elif choice == '3':
                handle_back_of_hand, flip_back_angles = configure_back_of_hand()

            elif choice == '4':
                video_frame_interval = configure_sampling_settings()
                print(f"Video processing frame interval set to: {video_frame_interval}")

            elif choice == '5':
                print("\nExiting...")
                break

            else:
                print("\nInvalid choice. Please try again.")

        except KeyboardInterrupt:
            print("\n\nProgram interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
