#!/usr/bin/env python3

""" Flask App for Image Classification """

import os
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from fastai.vision.all import *
import subprocess

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For flash messages
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    """ Define the allowed extensions for uploaded files """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load the pre-trained model (if available)
path = Path(UPLOAD_FOLDER)
model_path = Path('static/models/export.pkl')


@app.route('/')
def index():
    """ Home Page """
    return render_template('index.html')


@app.route('/train', methods=['GET', 'POST'])
def train_classifier():
    """ Route to Train Classifier Page """
    if request.method == 'POST':
        categories = request.form.get('categories')
        if not categories or len(categories.split(',')) < 2:
            flash("Please specify at least 2 items", 'error')
            return render_template('train.html')

        categories_list = [item.strip() for item in categories.split(',')]

        try:
            result = subprocess.run(
                ['python3', 'train.py', *categories_list],
                check=True,  # Will raise an error if the script fails
                capture_output=True,  # Capture the output so we can display it
                text=True
            )
            # Print the result of the training process for debugging
            print(f"Training output: {result.stdout}")
            # print(f"Training error output: {result.stderr}")
            flash("Training Complete! Go classify your images now.", 'success')
        except subprocess.CalledProcessError as e:
            print(f"Error while running train.py: {e}")
            print(f"Error output: {e.stderr}")
            flash("An error occurred during training")

        return redirect(url_for('train_classifier'))

    return render_template('train.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """ Route to Predict Page """
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Load the model
            if model_path.exists():
                learn = load_learner(model_path)
                img = PILImage.create(filepath)
                pred_class, pred_idx, outputs = learn.predict(img)
                prediction = str(pred_class)
                probability = round(outputs[pred_idx].item(), 2)
            else:
                prediction = "Model not trained yet."

            return render_template('predict.html',
                                   prediction=prediction,
                                   probability=probability,
                                   image_name=filename)

    return render_template('predict.html')


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
