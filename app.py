from flask import Flask, render_template, request
import numpy as np
import joblib  # Use joblib if you saved your model with it
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load(os.path.join(os.path.dirname(__file__), 'students_marks_predictor.pkl'))

# model = joblib.load('students_marks_predictor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the value from the form input
        study_hours = float(request.form['study_hours'])

        # Validate input
        if study_hours < 1 or study_hours > 24:
            return render_template('index.html', prediction_text="Please enter valid hours between 1 and 24.")

        # Make prediction
        input_data = np.array([[study_hours]])
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction_text=f'Predicted Marks: {round(prediction, 2)}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
