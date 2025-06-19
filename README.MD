# Lyric Genre & Content Classifier

This project is a web application that classifies the genre of song lyrics using a trained Logistic Regression model. It also provides content analysis, including sentiment and the use of explicit language. The application consists of a Python Flask backend that serves the machine learning model via an API, and a simple HTML/CSS/JavaScript frontend for user interaction.

## Project Structure


.
├── app.py                      # The Flask API server to handle predictions.
├── save_model.py               # Script to train and save the ML model pipeline.
├── requirements.txt            # Python dependencies for the project.
├── cleaned_rock_country_hiphop_rb.csv # The dataset used for training.
├── logistic_regression_pipeline.joblib  # Generated model file.
├── label_to_genre.joblib       # Generated label mapping file.
└── index.html                  # The front-end user interface (or provided separately).


## Setup Instructions

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

- Python 3.7+
- `pip` (Python package installer)

### 2. Environment Setup

It is highly recommended to use a virtual environment to keep project dependencies isolated.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3. Install Dependencies

Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

4. Download NLTK Data

The sentiment analysis feature requires the VADER lexicon from the NLTK library. Run the following command in a Python shell to download it:

import nltk
nltk.download('vader_lexicon')

Running the Application

The application requires a two-step process to run: first, train and save the model; second, start the API server.
Step 1: Train and Save the Model

Before starting the API, you need to generate the model artifact file (.joblib). Make sure your dataset cleaned_rock_country_hiphop_rb.csv is in the project's root directory.

Run the save_model.py script:

python save_model.py

This will create two files: logistic_regression_pipeline.joblib and label_to_genre.joblib. You only need to do this once.
Step 2: Start the Flask API Server

With the model files created, you can now start the API server.

python app.py

The server will start, and you should see output indicating it is running, typically on http://127.0.0.1:5000/. The API is now ready to accept requests.
Step 3: Use the Frontend

Open the provided HTML file (the Lyric Classifier UI) in your web browser. You can now paste lyrics into the text area and click "Classify". The frontend will send a request to your local Flask API and display the results.
API Endpoint

The Flask application exposes the following endpoint:

    URL: /predict

    Method: POST

    Description: Takes song lyrics and returns a classification report.

    Request Body (JSON):

    {
      "lyrics": "Is this the real life? Is this just fantasy? ..."
    }

    Success Response (JSON):

    {
      "predictedGenre": "Rock",
      "genreProbabilities": {
        "Country": 0.15,
        "Hip-Hop": 0.10,
        "R&B": 0.20,
        "Rock": 0.55
      },
      "sentimentLabel": "Negative",
      "sentimentScore": -0.45,
      "explicitnessLabel": "Not Explicit",
      "explicitWordCount": 0
    }

    Error Response (JSON):

    {
      "error": "Error message here."
    }

