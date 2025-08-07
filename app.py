from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['sepal_length']),
            float(request.form['sepal_width']),
            float(request.form['petal_length']),
            float(request.form['petal_width']),
        ]
        prediction = model.predict([features])[0]
        class_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        result = class_map.get(prediction, "Unknown")
        return render_template('result.html', prediction=result, features=features)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
