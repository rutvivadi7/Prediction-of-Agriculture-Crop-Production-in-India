"""
app.py
A Flask web application for crop production prediction.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import json

app = Flask(__name__)

# Load the best model
MODEL_PATH = "models/random_forest.pkl"  # Adjust with your best model filename
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Load feature information
FEATURE_INFO_PATH = "data/feature_info.json"
with open(FEATURE_INFO_PATH, 'r') as file:
    feature_info = json.load(file)

# Load scaler and encoder
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/encoder.pkl"
with open(SCALER_PATH, 'rb') as file:
    scaler = pickle.load(file)
with open(ENCODER_PATH, 'rb') as file:
    encoder = pickle.load(file)

# Load state and district information
STATES_PATH = "data/states_districts.json"
with open(STATES_PATH, 'r') as file:
    states_districts = json.load(file)

# Load crop information
CROPS_PATH = "data/crops.json"
with open(CROPS_PATH, 'r') as file:
    crops = json.load(file)

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html', 
                          states=list(states_districts.keys()),
                          crops=crops)

@app.route('/get_districts', methods=['POST'])
def get_districts():
    """Get districts for a selected state."""
    state = request.json.get('state')
    if state in states_districts:
        return jsonify({'districts': states_districts[state]})
    return jsonify({'districts': []})

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    # Get form data
    data = request.form.to_dict()
    
    # Process input data
    processed_data = process_input(data)
    
    # Make prediction
    prediction = model.predict(processed_data)[0]
    
    # Format prediction
    formatted_prediction = format_prediction(prediction, data)
    
    return render_template('result.html', 
                         prediction=formatted_prediction,
                         input_data=data)

def process_input(data):
    """Process input data for prediction."""
    # Create a DataFrame from input data
    input_df = pd.DataFrame([data])
    
    # Extract numerical features
    numerical_features = [f for f in feature_info['numerical_features'] 
                        if f in input_df.columns]
    
    # Extract categorical features
    categorical_features = [f for f in feature_info['categorical_features'] 
                          if f in input_df.columns]
    
    # Convert numerical values to float
    for feature in numerical_features:
        input_df[feature] = input_df[feature].astype(float)
    
    # Scale numerical features
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    
    # Encode categorical features
    categorical_array = encoder.transform(input_df[categorical_features])
    categorical_df = pd.DataFrame(
        categorical_array, 
        columns=encoder.get_feature_names_out(categorical_features)
    )
    
    # Combine all features
    final_df = pd.concat([input_df[numerical_features], categorical_df], axis=1)
    
    return final_df.values

def format_prediction(prediction, input_data):
    """Format the prediction result."""
    # Round to 2 decimal places
    prediction = round(prediction, 2)
    
    # Get crop and area information
    crop = input_data.get('Crop')
    area = float(input_data.get('Area', 0))
    
    # Calculate yield (production per hectare)
    yield_per_hectare = prediction / area if area > 0 else 0
    
    return {
        'crop': crop,
        'area': area,
        'total_production': prediction,
        'yield_per_hectare': round(yield_per_hectare, 2)
    }

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.route('/documentation')
def documentation():
    """Render the documentation page."""
    return render_template('documentation.html')

if __name__ == '__main__':
    app.run(debug=True)
