import pickle

import numpy as np
from flask import Flask, render_template, request

# Load the scaler and the ensemble model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract the input features from the form
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        smoking = int(request.form['smoking'])
        hx_smoking = int(request.form['hx_smoking'])
        hx_radiotherapy = int(request.form['hx_radiotherapy'])
        thyroid_function = int(request.form['thyroid_function'])
        physical_exam = int(request.form['physical_exam'])
        adenopathy = int(request.form['adenopathy'])
        focality = int(request.form['focality'])
        risk = int(request.form['risk'])
        t_stage = int(request.form['t_stage'])
        n_stage = int(request.form['n_stage'])
        m_stage = int(request.form['m_stage'])
        stage = int(request.form['stage'])
        response = int(request.form['response'])

        # Create the feature array
        features = np.array([[age, smoking, physical_exam, adenopathy, focality, risk, t_stage, n_stage, stage, response]])

        # Scale the input features
        scaled_features = scaler.transform(features)

        # Predict using the ensemble model
        prediction = model.predict(scaled_features)

        # Return the result
        if prediction == 1:
            result = "Recurrence of Thyroid Cancer is predicted."
        else:
            result = "No Recurrence of Thyroid Cancer is predicted."

        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
