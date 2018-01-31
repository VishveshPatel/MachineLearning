# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:01:41 2018

@author: A664092
"""

## Load Library
import pandas as pd
import datetime
from datetime import timedelta
import json
import requests
import numpy as np
from flask import Flask,request
app = Flask(__name__)

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


@app.route('/Get_Data',methods = ['POST'])
def get_data():
    req = request.get_json()
    args=req["timestamp"]
    #print(req)
    if args != None:
        input_data=get_input_data(str(args))
        input_data=list(input_data)
      
        return json.dumps({"Input":input_data})
    
if __name__ == '__main__':
   app.run()