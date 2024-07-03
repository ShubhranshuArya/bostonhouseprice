import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
# Load the pickled model
reg_model = pickle.load(open('lin_reg_model.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    reshaped_data = np.array(list(data.values())).reshape(1, -1)
    print(reshaped_data)
    scaled_data = scalar.transform(reshaped_data)
    prediction = reg_model.predict(scaled_data)
    output = prediction[0]
    print(output)
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

