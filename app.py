#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import accident_severity
import matplotlib
matplotlib.use('Agg')

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__,template_folder='templates')
app.config['EXPLAIN_TEMPLATE_LOADING'] = True
model = pickle.load(open('severity.pk1', 'rb'))



@app.route('/')
def home():
    return render_template('Flight_HomePage.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) if type(x)==int else float(x) for x in request.form.values()]
    print("int_features",int_features)
    final_features = [np.array(int_features)]
    print("final_features",final_features)
    prediction = model.predict(final_features)
    print("prediction",prediction)

    output = round(prediction[0], 2)
    
    if(output==0):
        output = "Highly_Fatal_And_Damaging"
    elif(output==1):
        output = "Minor_Damage_And_Injuries"
    elif(output==2):
        output = "Significant_Damage_And_Fatalities"
    else:
        output = "Significant_Damage_And_Serious_Injuries"
        
    return render_template('Flight_HomePage.html', prediction_text='The severity of Accident is $ {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




