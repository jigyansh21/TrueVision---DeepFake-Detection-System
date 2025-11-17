@echo off
echo ========================================
echo DeepFake Detection - Python 3.8
echo ========================================
echo.

cd /d "%~dp0"

REM Activate virtual environment
call venv_py38\Scripts\activate.bat

REM Check if model exists
if not exist "models\inceptionNet_model.h5" (
    echo [ERROR] Model file not found!
    echo Please ensure models\inceptionNet_model.h5 exists
    pause
    exit /b 1
)

echo Starting Flask server...
echo.
echo Access the application at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py



