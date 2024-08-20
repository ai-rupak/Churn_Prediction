# app.py

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('churn_model.pkl', 'rb'))

# If you have encoders, load them
# encoders = pickle.load(open('encoders.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

def preprocess_input(df):
    # Example preprocessing steps
    # 1. Encode categorical variables
    # for col, encoder in encoders.items():
    #     df[col] = encoder.transform(df[col])
    
    # 2. Convert data types
    # df = df.astype(float)
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form.to_dict()
    
    # Convert form data to DataFrame
    data = pd.DataFrame([form_data])
    
    # Preprocess data
    # data = preprocess_input(data)
    
    # Make prediction
    prediction = model.predict(data)
    
    # Convert prediction to human-readable form
    if prediction[0] == 1:
        result = 'The customer is likely to churn.'
    else:
        result = 'The customer is not likely to churn.'
    
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
