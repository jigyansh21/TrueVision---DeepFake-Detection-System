@echo off
echo ========================================
echo DeepFake Detection - Python 3.8 Setup
echo ========================================
echo.

REM Check if Python 3.8 is installed
python --version | findstr "3.8" >nul
if errorlevel 1 (
    echo [ERROR] Python 3.8 not found!
    echo.
    echo Please install Python 3.8 from:
    echo https://www.python.org/downloads/release/python-3810/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [OK] Python 3.8 found!
python --version
echo.

REM Navigate to Deploy directory
cd Deploy

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv_py38
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv_py38\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install dependencies
echo.
echo Installing dependencies...
echo This may take a few minutes...
echo.
pip install tensorflow==2.15.0 --quiet
pip install flask==2.2.2 werkzeug==2.2.2 --quiet
pip install opencv-python numpy==1.24.3 --quiet
pip install imageio imageio-ffmpeg Pillow --quiet

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To use the application:
echo.
echo 1. Activate the environment:
echo    cd Deploy
echo    venv_py38\Scripts\activate.bat
echo.
echo 2. Run the application:
echo    python app.py
echo.
echo 3. Open browser:
echo    http://localhost:5000
echo.
pause




