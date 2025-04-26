# Prediction-of-Agriculture-Crop-Production-in-India
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
Agricultural Crop Production Prediction System
A machine learning-based system for predicting agricultural crop yields across different regions of India based on environmental and agricultural parameters.
<h2>Project Overview</h2>
This project implements a comprehensive pipeline for predicting crop production in India using machine learning techniques. The system analyzes historical crop data along with various environmental factors such as soil characteristics, weather conditions, and agricultural inputs to make accurate yield predictions.
<h2>Features</h2>
<ul>
    <li>Data preprocessing pipeline for agricultural datasets</li>
    <li>Implementation of multiple machine learning models (Random Forest, Gradient Boosting, Neural Networks, etc.)</li>
    <li>Comprehensive model evaluation and comparison</li>
    <li>Web application for farmers and agricultural experts to make predictions</li>
    <li>Visualization of prediction results and model performance</li>
</ul>
<h2>Installation</h2>
<ul>
    <li>Clone the repository:<br>
git clone https://github.com/rutvivadi7/Prediction-of-Agriculture-Crop-Production-in-India.git<br>
cd crop-prediction-system
</li>
    <li>Create and activate a virtual environment:<br>
python -m venv venv<br>
source venv/bin/activate      # On Windows: venv\Scripts\activate</li>
    <li>Install required dependencies:<br>
pip install -r requirements.txt</li>
</ul>
<h2>Usage</h2>
<h3>Running the complete pipeline</h3>
<ul>
    <li>
      To run the complete pipeline from data preprocessing to model evaluation:<br>
      python main.py</li>
    <li>You can also run specific components:<br>
        python main.py --skip-preprocessing<br>
        python main.py --skip-training   <br>
        python main.py --skip-evaluation </li>
    <h3>Starting the web application</h3>
  <li>The web application will be available at <br> http://localhost:5000.</li>
  
</ul>
<h2>Data Sources</h2>
<ul>
  The system uses agricultural data from multiple sources:
<ol>
    <li>Crop production data from the Ministry of Agriculture & Farmers Welfare, Government of India</li>
    <li>Weather data from the Indian Meteorological Department</li>
    <li>Soil data from the National Bureau of Soil Survey and Land Use Planning</li>
</ol>
</ul>
<h2>Machine Learning Approach</h2>
<p>The project compares several machine learning algorithms:</p>
<ul>
    <li>Linear Regression</li>
    <li>Support Vector Regression</li>
    <li>Random Forest Regression</li>
    <li>Gradient Boosting Regression</li>
    <li>Neural Networks</li>
</ul>
</body>
</html>
