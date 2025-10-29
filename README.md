# Trajectory for Data collect RoboBrain 1.0 -- BRIN

## Getting started

This project uses Python and requires several dependencies. It is highly recommended to use a virtual environment (`venv`) to manage these dependencies.

### 0. Initial setup on a new PC 

If you are setting up this project on a brand‑new PC, follow the dedicated setup guide first:

- See [README_SETUP.md](./README_SETUP.md)  

Continue with the rest of this README after completing the steps in README_SETUP.md.

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

This will start the hand tracking process and display the visualization. The following options are available:
- **1. Real time camera**: Process hand tracking using your desired webcam in real-time.
- **2. Process video file**: Process hand tracking from a specified video file path declared in `main.py`.
- **3. Configure back of hand settings**: Adjust settings related to back-of-hand detection.
- **4. Configure video sampling interval (for option 2)**: Set the sampling interval for video processing when using option 2.
- **5. Exit**: Terminate the program.

### 5. Troubleshooting

Here are some common issues you might encounter and how to resolve them:

*   **`ModuleNotFoundError` or other dependency issues**:
    *   Ensure you have activated your virtual environment (`venv`).
    *   Make sure all dependencies are installed by running `pip install -r requirements.txt` within your activated virtual environment.
*   **Camera not detected or accessible**:
    *   Check if your webcam is properly connected and not being used by another application.
    *   Verify that your operating system grants Python (or your terminal application) permission to access the camera.
    *   If using a specific camera, ensure it's correctly configured in `main.py` if applicable.
*   **Poor hand tracking performance or accuracy**:
    *   Ensure good lighting conditions. Mediapipe's hand tracking can be sensitive to poor lighting.
    *   Keep your hand within a reasonable distance from the camera. Too close or too far can reduce accuracy.
    *   Avoid cluttered backgrounds if possible, although the system is designed to be robust to them.
    *   Check the camera's resolution and frame rate. Higher quality input generally leads to better tracking.
*   **Incorrect Roll, Pitch, Yaw values**:
    *   Be aware that camera mirroring can affect the perceived orientation. Adjust your interpretation or camera settings accordingly.
    *   Ensure your hand is positioned consistently relative to the camera.

### 6. Commit message
This project uses the conventional commit specification for better readability and clarity. It is mandatory to use conventional commit messages. Read more about conventional commits [here](https://www.conventionalcommits.org/en/v1.0.0/).

### 7. To do:
*   Handle calibration for zeroing initial hand position

## ⁉️ author?   
Author: Yosua Hares.  
Email: haresyosuaa[at]gmail.com
