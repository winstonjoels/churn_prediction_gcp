from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np
import os
import pymysql
import requests

db_user = os.environ.get('CLOUD_SQL_USERNAME')
db_password = os.environ.get('CLOUD_SQL_PASSWORD')
db_name = os.environ.get('CLOUD_SQL_DATABASE_NAME')
db_connection_name = os.environ.get('CLOUD_SQL_CONNECTION_NAME')

app = Flask(__name__)

global model, cols

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/model/<name>',methods=['POST'])
def model(name):
    global model, cols
    if name=="infy_bank":
        cols = ['Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
    else:
        cols = ['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
    file = name+".html"
    return render_template(file)

@app.route('/predict/<name>',methods=['POST'])
def predict(name):
    global model, cols
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
        unix_socket = '/cloudsql/{}'.format(db_connection_name)
        try:
            if os.environ.get('GAE_ENV') == 'standard':
                conn = pymysql.connect(user=db_user, password=db_password,
                                    unix_socket=unix_socket, db=db_name,
                                    cursorclass=pymysql.cursors.DictCursor
                                    )
        except pymysql.MySQLError as e:
            print(e)
#         conn = open_connection()
        with conn.cursor() as cursor:
            cursor.execute('INSERT INTO details (Name, Predicted, Actual) VALUES ('infy_bank', '0', '0')')
        conn.commit()
        conn.close()
    else:
        pred='The chance of this person is {}'.format(int(prediction))
        
    file = name+".html"
    return render_template(file,pred='{}'.format(pred))

if __name__ == '__main__':
    app.run()
