from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('Predictor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive data from frontend as JSON
        data = request.get_json()
        study_hours = float(data['study_hours'])

        # Make prediction
        input_data = np.array([[study_hours]])
        prediction = model.predict(input_data)[0]

        return jsonify({'prediction_text': f'Predicted Marks: {round(prediction, 2)}'})
    except Exception as e:
        return jsonify({'prediction_text': f'Error: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render will use PORT env
    app.run(host='0.0.0.0', port=port)
