# Initial Setup Guide for a New PC

This guide provides instructions for setting up your development environment on a brand-new PC to work with the Robobrain and Gemini Robotics project.

## 0. Initial Setup on a New PC

### 0.1. Install Git

Git is essential for cloning repositories and managing code versions.

*   **macOS:**
    ```bash
    xcode-select --install
    ```
    (This will install Xcode Command Line Tools, which includes Git.)
    Alternatively, you can install Git using Homebrew:
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install git
    ```
*   **Windows:**
    Download and install Git from the official website: [https://git-scm.com/download/win](https://git-scm.com/download/win)
    During installation, you can generally accept the default options.
*   **Linux (Debian/Ubuntu):**
    ```bash
    sudo apt update
    sudo apt install git
    ```

### 0.2. Install Python and Core System Dependencies

This project requires Python 3 and several system-level libraries, including `ffmpeg` for video processing. It's recommended to install the latest stable version of Python.

*   **macOS:**
    Python 3 is often pre-installed. You can check its version:
    ```bash
    python3 --version
    ```
    If you need a newer version or prefer a managed installation, use Homebrew:
    ```bash
    brew install python ffmpeg
    ```
*   **Windows:**
    Download the Python installer from the official website: [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
    **IMPORTANT:** During installation, make sure to check the box that says "Add Python X.Y to PATH".
    For `ffmpeg`, download the binaries from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add the `bin` directory to your system's PATH environment variable.
*   **Linux (Debian/Ubuntu):**
    ```bash
    sudo apt update
    sudo apt install -y python3 python3-venv python3-pip ffmpeg libgl1-mesa-glx libglib2.0-0 build-essential
    ```

### 0.3. Install Visual Studio Code (Recommended IDE)

Visual Studio Code is a popular and powerful code editor.

*   Download and install VS Code from the official website: [https://code.visualstudio.com/](https://code.visualstudio.com/)

### 0.4. Create and Activate a Python Virtual Environment

It is highly recommended to use a virtual environment (`venv`) to manage Python dependencies.

*   **Create venv:**
    ```bash
    python3 -m venv venv
    ```
*   **Activate venv (macOS/Linux):**
    ```bash
    source venv/bin/activate
    ```
*   **Activate venv (Windows PowerShell):**
    ```powershell
    .\venv\Scripts\Activate
    ```

### 0.5. Install Python Dependencies

Once the virtual environment is activated, upgrade `pip` and install the required Python packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 0.6. Verify Installations

Open a new terminal (and activate your virtual environment if applicable) and run the following commands to ensure everything is installed correctly:

```bash
git --version
python3 --version
pip3 --version
ffmpeg -version
python -c "import cv2; print(cv2.__version__); cap=cv2.VideoCapture(0); print('open', cap.isOpened()); cap.release()"
```

You should see the installed versions of Git, Python, Pip, and FFmpeg. The OpenCV command will check if your camera can be accessed.
