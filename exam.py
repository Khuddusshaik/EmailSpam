# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:42:13 2023

@author: khudd
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st

def pred(input_data):
    fName = "trained.sav"
    fName2 = "labels"
    df = pd.read_csv('spam.csv',encoding='latin-1')
    indx = 0
    for i in range(len(df.v2)):
        if df.v2[i] == input_data :
            indx = i
        break
    loaded_model = pickle.load(open(fName,"rb"))
    labels = pickle.load(open(fName2,"rb"))
    input_data = labels[i]
    input_data = np.asarray(input_data)
    
    var = labels[0].reshape(1,-1)
    pred = loaded_model.predict(var)
    return pred
    
def Deploy():
    st.title("Email Spam Detection")
    
    v2 = st.text_input('V2 Value')
    result = ""
    if st.button("Show Result"):
        input_data = v2
        result = pred(input_data)
    st.success(result)
    
Deploy()