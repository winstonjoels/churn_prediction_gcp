from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
import random

def randN():
    N=7
    min = pow(10, N-1)
    max = pow(10, N) - 1
    id = random.randint(min, max)
    return id

app = Flask(__name__)

global model, cols, id

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/model/<name>',methods=['POST'])
def model(name):
    global model, cols, id
    id = randN()
    if name=="infy_bank":
        cols = ['Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
    else:
        cols = ['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
    file = name+".html"
    return render_template(file, id=id)

@app.route('/predict/<name>',methods=['POST'])
def predict(name):
    global model, cols, id
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    if name == 'mush':
        name1 = name+"_training_pipeline"
    else:
        name1 = name
    model = load_model(name1)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = model.predict(data_unseen)
    if name == 'mush':
        if int(prediction)==1:
            pred="The mushroom is Poisonous"
        else:
            pred="The mushroom is Edible"
    else:
        pred='The chance of this person is {}'.format(int(prediction))
    
    actual = '?'
    list = [id, name, int(prediction), actual]
    list2 = ['ID','Name', 'Predicted', 'Actual']
    df = np.array(list)
    df = pd.DataFrame([df], columns=list2)
    
    df.to_csv('data/Details.csv', mode='a', header=False, index=False)
    
    file = name+".html"
    return render_template(file,pred='{}'.format(pred))

if __name__ == '__main__':
    app.run()
