from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and prepare the training data
train_data_path = "C:\\Users\\USER\\Downloads\\Genre Classification Dataset\\train_data.txt"
train_data = pd.read_csv(train_data_path, sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine='python')

# Feature extraction
t_v = TfidfVectorizer(stop_words='english', max_features=100000)
X_train = t_v.fit_transform(train_data['DESCRIPTION'])

# Encode genre labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['GENRE'])

# Train model
clf = LinearSVC()
clf.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    description = data['description']
    t_v1 = t_v.transform([description])
    pred_label = clf.predict(t_v1)
    genre = label_encoder.inverse_transform(pred_label)[0]
    return redirect(url_for('result', title=data['title'], year=data['year'], director=data['director'], description=description, genre=genre))

@app.route('/result')
def result():
    title = request.args.get('title')
    year = request.args.get('year')
    director = request.args.get('director')
    description = request.args.get('description')
    genre = request.args.get('genre')
    return render_template('result.html', title=title, year=year, director=director, description=description, genre=genre)

if __name__ == "__main__":
    app.run(debug=True)
