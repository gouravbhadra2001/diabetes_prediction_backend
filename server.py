from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def make_prediction(data):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import pickle

# Load the saved model from the pickle files
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('saved_model.pkl', 'rb') as file:
        model = pickle.load(file)




# Convert smoking history to one-hot encoded format
    smoking_history_types = [0, 0, 0, 0, 0, 0]
    smoking_history_types[data['smoking_history']-1] = 1
    data.pop('smoking_history')  # Remove original smoking history value
    data.update({'smoking_history_' + str(i+1): smoking_history_types[i] for i in range(len(smoking_history_types))})

# Create a DataFrame with the user input
    user_df = pd.DataFrame([data])

# Ensure feature names match the ones used during training
    user_df = user_df.rename(columns={'smoking_history_' + str(i+1): 'smoking_history_' + name for i, name in enumerate(['No Info', 'current', 'ever', 'former', 'never', 'not current'])})

# Scale the user input data
   
    cols_to_scale = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    user_df[cols_to_scale] = scaler.transform(user_df[cols_to_scale])

# Make the prediction
    user_pred = model.predict(user_df)

# Display the prediction
    
    prediction_result = ""

    if user_pred[0] == 1:
        prediction_result = "You are predicted to have diabetes."
    else:
        prediction_result = "You are predicted not to have diabetes."


    return prediction_result


@app.route('/')
def home():
    return "Hello World"
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Uncomment the following lines when you implement prediction logic
        prediction_result = make_prediction(data)
        response_data = {'prediction_result': prediction_result}

        # For now, return the received data
        print(response_data)
        return jsonify(response_data)
    
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
