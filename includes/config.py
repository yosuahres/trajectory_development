def configure_back_of_hand():
    print("\n" + "=" * 40)
    print("Back of Hand Configuration")
    print("=" * 40)
    handle_back = True
    flip_angles = True

    while True:
        print(f"Current settings: Back of hand detection: {'Enabled' if handle_back else 'Disabled'}, Angle flipping: {'Enabled' if flip_angles else 'Disabled'}")
        print("\nOptions:")
        print("1. Toggle back of hand detection")
        print("2. Toggle angle flipping")
        print("3. Return to main menu")
        print("=" * 40)

        choice = input("Enter your choice (1-3): ").strip()

        if choice == '1':
            handle_back = not handle_back
            print(f"Back of hand detection: {'Enabled' if handle_back else 'Disabled'}")

        elif choice == '2':
            flip_angles = not flip_angles
            print(f"Angle flipping for back of hand: {'Enabled' if flip_angles else 'Disabled'}")

        elif choice == '3':
            return handle_back, flip_angles

        else:
            print("Invalid choice. Please try again.")

def configure_sampling_settings():
    print("\n" + "=" * 40)
    print("Sampling Configuration")
    print("=" * 40)
    print("Frame sampling options:")
    print("1. Every 30th frame (30) - 1 sec intervals")
    print("2. Every 60th frame (60) - 2 sec intervals")
    print("3. Every 120th frame (120) - 4 sec intervals")
    print("4. Every 180th frame (180) - 6 sec intervals")
    print("5. Custom interval")
    print("6. Return to main menu (will use default 120)")
    print("=" * 40)

    while True:
        choice = input("Enter your choice (1-6): ").strip()

        if choice == '1':
            return 30
        elif choice == '2':
            return 60
        elif choice == '3':
            return 120
        elif choice == '4':
            return 180
        elif choice == '5':
            try:
                interval = int(input("Enter custom sampling interval (1-500): "))
                if 1 <= interval <= 500:
                    return interval
                else:
                    print("Invalid interval. Please enter a value between 1 and 500.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        elif choice == '6':
            return 120  
        else:
            print("Invalid choice. Please try again.")
