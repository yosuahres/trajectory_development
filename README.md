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

### To be Noted:
*   Camera angle positioning is affecting the result, as im using my laptop webcam, i need to make sure my hands are on the laptops webcam height level.
*   How far your hands from the camera it does not matter.
*   Yaw is waving should be roll
*   Pitch is correct, yaw and roll should be switched