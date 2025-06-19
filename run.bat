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
ECHO [2/6] Setting up virtual environment...
if not exist venv (
    ECHO Creating virtual environment 'venv'...
    python -m venv venv
) else (
    ECHO Virtual environment 'venv' already exists.
)

:: Step 3: Activate virtual environment and install dependencies
ECHO [3/6] Activating environment and installing dependencies from requirements.txt...
call venv\Scripts\activate.bat
pip install -r requirements.txt

:: Step 4: Download NLTK data for sentiment analysis
ECHO [4/6] Downloading NLTK data...
python -m nltk.downloader vader_lexicon

:: Step 5: Train the model if the model file doesn't exist
ECHO [5/6] Checking for model files...
if not exist "Model\logistic_regression_pipeline.joblib" (
    ECHO Model file not found. Running the training script...
    python save_model.py
) else (
    ECHO Model file already exists. Skipping training.
)

:: Step 6: Launch the API and the frontend UI
ECHO [6/6] Launching the application...
ECHO Starting the Flask API server in a new window...
start "Flask API" cmd /c "venv\Scripts\python.exe app.py"

ECHO Waiting for server to start...
timeout /t 5 /nobreak >nul

ECHO Opening the user interface in your browser...
start index.html

ECHO ===================================================
ECHO  Setup complete! The app is now running.
ECHO  You can close this window. The API server is running in a separate window.
ECHO ===================================================

