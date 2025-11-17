@echo off
echo ========================================
echo DeepFake Detection - TensorFlow Fix
echo ========================================
echo.
echo This will downgrade TensorFlow to 2.15.0
echo for compatibility with the existing model.
echo.
pause

echo.
echo Step 1: Uninstalling TensorFlow 2.20.0...
pip uninstall tensorflow keras -y

echo.
echo Step 2: Installing TensorFlow 2.15.0...
pip install tensorflow==2.15.0 --user

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Please restart the application with:
echo   python app.py
echo.
pause




