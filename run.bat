@echo off
ECHO ===================================================
ECHO  Lyric Genre & Content Classifier - Setup & Launch
ECHO ===================================================

:: Step 1: Check for Python installation
ECHO [1/6] Checking for Python...
where python >nul 2>nul
if %errorlevel% neq 0 (
    ECHO Python is not installed or not in PATH. Please install Python and try again.
    pause
    exit /b
)
ECHO Python found!

:: Step 2: Create a virtual environment if it doesn't exist
set VENV_PATH=venv\Scripts\activate.bat
if not exist venv (
    ECHO [2/6] Creating virtual environment 'venv'...
    python -m venv venv
) else (
    ECHO [2/6] Virtual environment 'venv' already exists.
)

:: Step 3: Install dependencies only if not already installed
if not exist "venv\Lib\site-packages\nltk" (
    ECHO [3/6] Activating environment and installing dependencies...
    call %VENV_PATH%
    pip install -r requirements.txt
) else (
    ECHO [3/6] Dependencies already installed.
    call %VENV_PATH%
)

:: Step 4: Download NLTK data if not already present
python -c "import nltk; nltk.data.find('sentiment/vader_lexicon.zip')" 2>nul
if %errorlevel% neq 0 (
    ECHO [4/6] Downloading NLTK vader_lexicon...
    python -m nltk.downloader vader_lexicon
) else (
    ECHO [4/6] NLTK vader_lexicon already downloaded.
)

:: Step 5: Train the model if the model file doesn't exist
if not exist "Model\logistic_regression_pipeline.joblib" (
    ECHO [5/6] Model file not found. Running training script...
    python save_model.py
) else (
    ECHO [5/6] Model file already exists. Skipping training.
)

:: Step 6: Launch the application
ECHO [6/6] Launching the application...
ECHO Starting Flask API server in a new window...
start "Flask API" cmd /c "call %VENV_PATH% && python app.py"

ECHO Waiting for server to start...
timeout /t 5 /nobreak >nul

ECHO Opening the frontend in your browser...
start "" templates\index.html

ECHO ===================================================
ECHO  Setup complete! The app is now running.
ECHO  You can close this window. The API server is running in a separate window.
ECHO ===================================================
pause
