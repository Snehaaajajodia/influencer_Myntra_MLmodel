# app/routes.py
from flask import Flask, jsonify, request
from main import data_loader, data_preprocess, model_trainer, Prediction
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to Influencer Score API'

@app.route('/train_model', methods=['POST'])
def train():
    df = data_loader()
    df = data_preprocess(df)
    model, mse, r2 = model_trainer(df)
    return jsonify({
        'message': 'Model training completed successfully',
        'mse': mse,
        'r2': r2
    })

@app.route('/predict_score', methods=['POST'])

def predict():
    data = request.json  # Receive JSON input
    df = pd.DataFrame(data)  # Convert JSON to DataFrame
    result = Prediction(df)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)