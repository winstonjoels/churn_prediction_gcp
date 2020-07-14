from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

global model, cols

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/model/<name>',methods=['POST'])
def model(name):
    global model, cols
#     model = load_model(name)
    if name=="infy_bank":
        cols = ['Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
    else:
        cols = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'Department', 'salary']
    file = name+".html"
    return render_template(file)

@app.route('/predict/<name>',methods=['POST'])
def predict(name):
    global model
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    model = load_model(name)
    data_unseen = pd.DataFrame([final], columns = cols)
    if name=='churn':
        prediction = predict_model(model, data=data_unseen, round = 0)
        prediction = int(prediction.Label[0])
    else:
        prediction = model.predict(data_unseen)
        prediction = int(prediction)
    file = name+".html"
    return render_template(file,pred='The chance of this person is {}'.format(prediction))

if __name__ == '__main__':
    app.run()