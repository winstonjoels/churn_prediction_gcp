#!/usr/bin/env python
# coding: utf-8

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
    cols = ['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
    file = name+".html"
    return render_template(file)

@app.route('/predict/<name>',methods=['POST'])
def predict(name):
    global model
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    name1 = name+"_training_pipeline"
    model = load_model(name1)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = model.predict(data_unseen)
    if int(prediction)==1:
        pred="Poisonous"
    else:
        pred="Edible"
    file = name+".html"
    return render_template(file,pred='The mushroom is {}'.format(pred))

if __name__ == '__main__':
    app.run()
