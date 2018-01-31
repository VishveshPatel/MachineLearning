# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:58:06 2018

@author: A664092
"""
import pandas as pd
import datetime
from datetime import timedelta
import math
import json
import requests
from keras.models import load_model
import numpy as np
from flask import Flask,request
app = Flask(__name__)



model=load_model("MLP.h5")


def equal(s1, s2):
    d1=datetime.datetime.strptime(s1, '%Y-%m-%d %H:%M:%S')
    d2=datetime.datetime.strptime(s2, '%Y-%m-%d %H:%M:%S')
    difference=(d1-d2)
    return difference.seconds

def get_input_data(s1):
    ## Load Data
    test_data=pd.read_csv("test.csv",header=None)
    
    #s1="2012-01-01 00:16:40"
    d1=datetime.datetime.strptime(s1, '%Y-%m-%d %H:%M:%S')
    d1=d1-timedelta(seconds=10)
    s1=str(d1)
    ## get the difference and index
    index=int(equal(s1,test_data.iloc[0,0])/10)
    diff=equal(s1,test_data.iloc[index,0])

    if diff != 0:
        for i in range(test_data.shape[0]):
            diff=equal(s1,test_data.iloc[i,0])
            if diff == 0:
                index=i
                break

    ## Get last 100 data
    input_data=test_data.iloc[index-99:index+1,1]
    return input_data

#
#@app.route('/Get_Data',methods = ['POST'])
#def get_data():
#    req = request.get_json()
#    args=req["timestamp"]
#    #print(req)
#    if args != None:
#        input_data=get_input_data(str(args))
#        input_data=list(input_data)
#      
#        return json.dumps({"Input":input_data})


@app.route('/Predict_The_Causes',methods = ['POST'])
def predict_the_causes():
    req = request.get_json()
    args=req["timestamp"]
    print(args)
    if args != None:
        input_data=get_input_data(str(args))
        input_data=list(input_data)
#        url = "http://127.0.0.1:5000/Get_Data"
#        payload={}
#        payload['timestamp']=args
#        headers = {
#        'content-type': "application/json",
#        'cache-control': "no-cache",
#        }
#        response = requests.request("POST", url, json=payload, headers=headers)
#        data = response.json()
#        Input_data=data["Input"]
        Input_data=np.array(input_data)
        Input_data=Input_data.reshape([1,100])
        y_pred=model.predict(Input_data)
        ##Causes
        class_names=['Anemometer errors','BladeAccumulatorPressureIssues', 'CoolingSystemIssues', 'Generator speed discrepancies', 'Oil Leakage', 'Overheated oil']
    
        ## Display Result
        pred=list(y_pred[0])
        pred1=pred.copy()
        out_list=[]
        print("Output: \n")
        for i in range(0,2):
            max_val=max(pred1)
            out=str(round(max_val * 100,2)) + "% chances of " + str(class_names[pred.index(max_val)])
            out_list.append(out)
            pred1.remove(max_val)
        return (json.dumps({"Output":out_list}))
    return 'hi'
    
@app.route('/Test',methods = ['GET'])
def test():
    return 'Congo!'

if __name__ == "__main__":
    app.run(host='0.0.0.0')
