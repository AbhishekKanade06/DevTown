from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('/Users/abhishekkanade/Documents/Data Structures/DAA_Assignment/DAA_DEMO/DevTown/boston_housing_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # List of feature names
            feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

            # Get feature values from the form, defaulting to 0 if the field is empty
            features = []
            for feature in feature_names:
                value = request.form.get(feature)
                if value:  # Check if the value is not None or empty
                    features.append(float(value))
                else:
                    features.append(0.0)  # Set missing features to 0.0 (or choose another default)

            # Convert to numpy array for model prediction
            features = np.array(features).reshape(1, -1)

            # Make prediction
            prediction = model.predict(features)
            output = prediction[0]

            # Return the prediction with the result
            return render_template('index.html', prediction_text=f'Predicted value: {output:.2f}')

        except Exception as e:
            # Handle any other errors
            return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
