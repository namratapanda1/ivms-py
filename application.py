#from flask import Flask
#app = Flask(__name__)

#@app.route("/")
#def hello():
#    return "Hello Namrata!"

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os
from pathlib import Path


# Your API definition
model = None
app = Flask(__name__)


def load_model():
    #Create a path to your sub-folder and file name:
    curr_dir=os.getcwd()
    #print(os.getcwd())
    model_filename = os.path.join(curr_dir,'model','reg_model_v1.0.pk')
    
    global model
    
    # model variable refers to the global variable
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    return model



@app.route('/')
def home_endpoint():
    return 'Regression Model to predict Overall Score!'


@app.route('/predict', methods=['GET','POST'])
def predict():
    #pred_finalscore = []
    # Works only for a single sample
    if request.method == 'POST':
        input_json = request.get_json(force=True)  # Get data posted as a json
        print("..............Input JSON..........",input_json)
        print(type(input_json))
        inputRecdf = pd.DataFrame.from_dict(input_json, orient='columns')
        print(inputRecdf)
        pred_finalscore = model.predict(inputRecdf)  # runs globally loaded model on the data
        pred_finalscore_list = list(pred_finalscore)
        print(pred_finalscore_list)
        coeff = model.coef_
        
    return jsonify({'prediction': str(round(pred_finalscore_list[0],2)), 'coeff': {'pricing_coef':str(round(coeff[0],2)),'quality_coef':str(round(coeff[1],2)),'delivery_coef':str(round(coeff[2],2)),'financial_coef':str(round(coeff[3],2))}})


if __name__ == '__main__':
    model = load_model()  # load model at the beginning once only
    print('Model loaded')
    print(model)
    app.run(port=5000)
