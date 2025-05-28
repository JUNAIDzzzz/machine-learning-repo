from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

# Load dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Prepare data
x = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']

# Standardize data
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Train-test split and train model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.asarray(data['input_data']).reshape(1, -1)
    std_data = scaler.transform(input_data)
    prediction = classifier.predict(std_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)