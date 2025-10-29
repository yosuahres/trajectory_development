# Initial setup on a new PC 

This short guide helps preparing a brand-new PC before following the main README steps.

Linux (Ubuntu / Debian)
- Update packages and install essentials:
  ```bash
  sudo apt update && sudo apt install -y git python3 python3-venv python3-pip ffmpeg libgl1-mesa-glx libglib2.0-0 build-essential
  ```
- Verify Python:
  ```bash
  python3 --version
  ```
- Create and activate venv:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- Upgrade pip and install requirements:
  ```bash
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Notes: Grant camera access if required by your distro/DE.

macOS
- Install Homebrew (if missing): 
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
- Install Python and ffmpeg:
  ```bash
  brew install python ffmpeg
  ```
- Verify Python and create venv:
  ```bash
  python3 --version
  python3 -m venv venv
  source venv/bin/activate
  ```
- Upgrade pip and install requirements:
  ```bash
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Notes: Open System Preferences → Security & Privacy → Camera to allow camera access for the terminal/IDE.

Windows
- Install Python from https://www.python.org/ (choose latest 3.x and check "Add Python to PATH").
- Optional: install Visual C++ Build Tools if compiling native wheels.
- In PowerShell (run as user):
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Notes: Allow camera access in Settings → Privacy → Camera.

Common checks after setup
- Confirm OpenCV can access camera:
  ```bash
  python -c "import cv2; print(cv2.__version__); cap=cv2.VideoCapture(0); print('open', cap.isOpened()); cap.release()"
  ```
- If camera doesn't open: try different device index (0,1...) or check OS privacy settings.

Troubleshooting
- If pip install fails on some packages, install system build tools (build-essential on Linux, Visual C++ on Windows) and retry.
- For missing GUI libraries (errors from cv2), ensure libgl and related packages are installed (examples above).

After completing the steps in this file, return to README.md and continue with "Getting started".
