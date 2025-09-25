# Trajectory for Data collect RoboBrain 1.0 -- BRIN

## Getting started

This project uses Python and requires several dependencies. It is highly recommended to use a virtual environment (`venv`) to manage these dependencies.

### 1. Clone this repo:

Using bash or other terminal

```
git clone https://github.com/yosuahres/trajectory_development.git && cd trajectory_development  
```

### 2. Create and Activate a Virtual Environment

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

### 3. Install Dependencies

Once the virtual environment is activated, install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Modify `!CHANGETHIS` (Optional)

!CHANGETHIS is a comment you need to look up and adjust it based on your need.

### 4. Run the Script

After installing dependencies and making any necessary modifications, you can run the main script:

```bash
python3 main.py
```

This will start the hand tracking process and display the visualization.  
Option 1: real time processing using your desired cam.   
Option 2: processing the video path declare on main.py.  
Option 3: exit program.   

### 5. Commit message
This project uses the conventional commit specification for better readability and clarity. It is mandatory to use conventional commit messages. Read more about conventional commits [here](https://www.conventionalcommits.org/en/v1.0.0/).

### 6. To be Noted:
*   Camera angle positioning is affecting the result, as im using my laptop webcam, i need to make sure my hands are on the laptops webcam height level.
*   Use either your left or right hand. The system should be able to track both.
*   How far your hands from the camera it does matter.
*   Any background does not affect the result
*   Lighting condition does affect the tracker of the mediapipe
*   Camera mirror does affect the result.
*   Initially, there was a confusion where the calculated 'roll' value was visually represented as 'yaw', and vice-versa. This has been corrected in `main.py`. Specifically, the value initially calculated as 'roll' by the `_get_hand_orientation` function is now assigned to the 'yaw' variable, and the value initially calculated as 'yaw' is assigned to the 'roll' variable. The underlying calculations for the hand orientation were always correct, but the labels and visual indicators (gizmo colors, text displays) were swapped.

### 7. To do:
*   Handle the back of the hand

## ⁉️ author?   
Author: Yosua Hares.  
Email: haresyosuaa[at]gmail.com