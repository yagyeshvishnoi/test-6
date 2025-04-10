import flask
import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
import numpy as np
import pickle

application = flask.Flask(__name__)
CORS(application)  # Enable CORS for all routes

# Only enable Flask debugging if an env var is set to true
application.debug = os.environ.get('FLASK_DEBUG') in ['true', 'True']

# Get application version from env
app_version = os.environ.get('APP_VERSION')

# Get cool new feature flag from env
enable_cool_new_feature = os.environ.get('ENABLE_COOL_NEW_FEATURE') in ['true', 'True']

# Load the pre-trained model for recommendation
with open('RandomForest.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the pre-trained model for yield
with open('preprocesser.pkl', 'rb') as preprocessor_file:
    prepro = pickle.load(preprocessor_file)
with open('model.pkl', 'rb') as model_file:
    model2 = pickle.load(model_file)


@application.route('/')
def hello_world():
    message = "Hello, world!"
    return flask.render_template('index.html',
                                  title=message,
                                  flask_debug=application.debug,
                                  app_version=app_version,
                                  enable_cool_new_feature=enable_cool_new_feature)

@application.route('/predict')
def predict():
    try:
        # Extract the required features from query parameters
        required_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_data = [float(request.args.get(feature)) for feature in required_features]
        
        # Convert input data to a NumPy array
        input_array = np.array(input_data).reshape(1, -1)  # Ensure 2D array
        
        # Make prediction
        prediction = model.predict(input_array)
        
        # Return prediction as a JSON response
        return jsonify({
            'success': True,
            'prediction': prediction.tolist()
        })
    except Exception as e:
        # Handle errors and send response
        return jsonify({
            'success': False,
            'error': str(e)
        })


@application.route('/predict2')
def crop_yield():
    data = request.args
    # Extract required features from the input data
    Crop_Year = data.get('Crop_Year')
    Area = data.get('Area')
    Production = data.get('Production')
    Annual_Rainfall = data.get('Annual_Rainfall')
    Fertilizer = data.get('Fertilizer')
    Pesticide = data.get('Pesticide')
    Crop = data.get('Crop')
    Season = data.get('Season')
    State = data.get('State')

    # Check if any required feature is missing
    if None in [Crop_Year, Area, Production, Annual_Rainfall, Fertilizer, Pesticide, Crop, Season, State]:
        return jsonify({
            'success': False,
            'error': 'Missing required fields in input data'
        })

    # Prepare features for prediction
    features = np.array([[Crop_Year, Area, Production, Annual_Rainfall, Fertilizer, Pesticide, Crop, Season, State]], dtype=object)
    transform_features = prepro.transform(features)

    # Make prediction
    predicted_yield = model2.predict(transform_features).reshape(-1, 1)
    return jsonify({
            'success': True,
            'prediction': predicted_yield[0][0]
        })


if __name__ == '__main__':
    application.run(host='0.0.0.0')
