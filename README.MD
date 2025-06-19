# Lyric Genre & Content Classifier

Welcome to our **Lyric Genre & Content Classifier** project!  
This tool uses Natural Language Processing (NLP) and machine learning to predict a song’s **genre** and **emotional tone** based on its lyrics.

---

## How to Run the Project

Simply **double-click `run.bat`** or run it via terminal:

```
run.bat
```

This script handles:
1. Checking if Python is installed.
2. Creating a virtual environment (if not already present).
3. Installing required dependencies (if not already installed).
4. Downloading NLTK’s VADER sentiment lexicon.
5. Training the model (if not already trained).
6. Launching the Flask API in a new window.
7. Opening the web UI located at `templates/index.html`.

> If you've already set up the project, running the batch file again will simply **start the API and frontend** — no redundant setup.

---

## Project Overview

The project is an NLP-powered classifier that analyzes a song’s lyrics to:
- **Predict the genre** using machine learning (Logistic Regression pipeline).
- **Analyze sentiment** using VADER sentiment analysis.

We preprocessed lyrics datasets, vectorized the text, and built a custom classifier that runs via a web-based UI.

---

## Repository Structure

```
├── run.bat                       # One-click setup and launcher
├── app.py                        # Flask API backend
├── save_model.py                 # Model training script
├── requirements.txt              # Python dependencies
├── Model/
│   └── logistic_regression_pipeline.joblib  # Trained model
├── templates/
│   └── index.html                # Frontend interface
├── notebooks/
│   ├── gurmat_singh_work.ipynb
│   ├── robert_seibel_work.ipynb
│   └── minh_nhat_work.ipynb     # Each member’s contribution
├── Presentation/
│   └── lyric_classifier_slides.pptx  # Final presentation
└── README.md
```

---

## Team Members (Sweathog Subset)

- Gurmat Singh Sour
- Robert Seibel
- Minh Nhat Mai

---

## Demo & Usage

1. Run `run.bat`.
2. The Flask backend will start on `http://127.0.0.1:5000/predict`.
3. Your default browser will open the frontend UI where you can:
   - Enter song lyrics
   - Submit the form
   - View predicted genre and sentiment instantly

---

# Contributing to Lyric Genre & Content Classifier

Thank you for your interest in contributing to this project!

## How to Contribute

1. **Fork the repository**  
   Click the "Fork" button at the top right of this page to create your own copy of the repo.

2. **Clone your fork**  
   ```bash
   git clone https://github.com/YOUR-USERNAME/Song-Classifier.git

3. **Create a new branch for your changes**
    ```bash
    git checkout -b feature/your-feature-name

4. **Make your changes**

5. **Commit and push your code**
    ```bash
    git add .
    git commit -m "Add your message here"
    git push origin feature/your-feature-name

6. Create a Pull Request
    ```bash
    Go to the original repo and click “Compare & pull request”.



Thank you for checking out our project!