
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open('xgb_classifier.pkl', 'rb') as model_file:
    xgb_classifier = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract input parameters from the JSON data
    input_data = [
        data['HighBP'],
        data['HighChol'],
        data['CholCheck'],
        data['Smoker'],
        data['Stroke'],
        data['HeartDiseaseorAttack'],
        data['Ment_Hlth'],
        data['AnyHealthcare'],
        data['GenHlth'],
        data['Fruits'],
        data['NoDocbcCost'],
        data['BMI'],
        data['Veggies'],
        data['Phys_Hlth']
    ]

    # Make a prediction using the loaded model
    prediction = xgb_classifier.predict([input_data])[0]

    # Return the prediction as a JSON response
    response = {'prediction': prediction}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
