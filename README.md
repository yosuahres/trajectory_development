# trajectory_development

to do:   
- add rotation.  
- collect dataset.  
- train.  
- use robobrain 1.  
- develop rotation per dot trajectory, so modify robobrain 1.  

## How to Run the Script

This project uses Python and requires several dependencies. It is highly recommended to use a virtual environment (`venv`) to manage these dependencies.

### 1. Create and Activate a Virtual Environment

First, create a virtual environment in the project directory:

```bash
python3 -m venv venv
```

Then, activate the virtual environment:

*   **On macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```
*   **On Windows:**
    ```bash
    .\venv\Scripts\activate
    ```

### 2. Install Dependencies

Once the virtual environment is activated, install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Modify `main.py` (Optional)

The `main.py` script is the entry point for running the hand tracking and visualization. You might need to modify certain parameters within `main.py` depending on your specific use case.

For example, you might want to change:
*   The input video file path.
*   Visualization settings.
*   Output file paths.

Open `main.py` and adjust the relevant variables as needed.

### 4. Run the Script

After installing dependencies and making any necessary modifications, you can run the main script:

```bash
python3 main.py
```

This will start the hand tracking process and display the visualization.

## How to Test Roll, Pitch, and Yaw

### General Setup:
*   Use either your left or right hand. The system should be able to track both.

### Testing Roll (Rotation around the forward axis of the hand):
*   **Starting Position:** Hold your hand flat, palm facing down, fingers pointing straight ahead (away from your body). This should represent a "neutral" roll.
*   **Movement:** Rotate your hand along its long axis.
    *   **Positive Roll (Palm Down):** Rotate your hand so your palm faces more towards the ground.
    *   **Negative Roll (Palm Up):** Rotate your hand so your palm faces more towards the sky.
*   **Observation:** You should see the red axis (Roll) on the gizmo rotate, and the "Roll" value in the top left corner of the video change accordingly. The red angle guide should also reflect this rotation.

### Testing Pitch (Rotation around the side-to-side axis of the hand):
*   **Starting Position:** Hold your hand flat, palm facing forward, fingers pointing straight up. This should represent a "neutral" pitch.
*   **Movement:** Rotate your hand up and down as if you're waving "bye-bye" with your fingers.
    *   **Positive Pitch (Fingers Up):** Tilt your fingers upwards, bending at the wrist.
    *   **Negative Pitch (Fingers Down):** Tilt your fingers downwards, bending at the wrist.
*   **Observation:** You should see the green axis (Pitch) on the gizmo rotate, and the "Pitch" value change. The green angle guide should also reflect this rotation.

### Testing Yaw (Rotation around the up-down axis of the hand):
*   **Starting Position:** Hold your hand flat, palm facing forward, fingers pointing straight ahead (away from your body). This should represent a "neutral" yaw.
*   **Movement:** Rotate your entire hand left and right as if you're turning a doorknob or pointing in different directions.
    *   **Positive Yaw (Pointing Right):** Rotate your hand to point to your right.
    *   **Negative Yaw (Pointing Left):** Rotate your hand to point to your left.
*   **Observation:** You should see the blue axis (Yaw) on the gizmo rotate, and the "Yaw" value change. The blue angle guide should also reflect this rotation.
