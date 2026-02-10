@echo off
echo ================================================
echo Speech Emotion Recognition - Quick Start
echo ================================================
echo.

REM Check if model exists
if not exist "emotion_model.h5" (
    echo Creating demo model...
    python create_demo_model.py
    echo.
)

echo Starting web application...
echo.
python run_app.py

pause
