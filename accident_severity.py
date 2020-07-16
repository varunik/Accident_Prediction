#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


aero_data = pd.read_csv(r"C:\Users\varun\flight_dataset\train.csv")


# In[3]:


aero_data.head()


# In[4]:


aero_data.shape


# In[5]:


# from sklearn.preprocessing import LabelEncoder

# lb_make = LabelEncoder()
# aero_data["Severity"] = lb_make.fit_transform(aero_data["Severity"])

# aero_data

# aero_data.nlargest(1,['Accident_ID'])
#print(aero_data.nsmallest(1,['Safety_Score']))

# aero_data.max()

# aero_data.min()

# df = pd.DataFrame(columns={'Max Values','Min Values'})
# df['Max Values'] =  aero_data.max()
# df['Min Values'] =  aero_data.min()
# df

aero_data.isnull().sum()


# In[6]:


aero_data.dtypes


# In[ ]:





# In[7]:


type(aero_data)


# In[8]:


aero_data = aero_data.as_matrix()


# In[9]:


aero_data


# In[10]:


x = aero_data[:,1:-1]


# In[11]:


y= aero_data[:,0]


# In[12]:


y[0:10]


# In[13]:


label_encoded_y[0:10]


# In[15]:


# encode string class values as integers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
label_encoded_y = label_encoder.transform(y)


# In[16]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,label_encoded_y,test_size=.3,random_state=1)


# In[17]:


x_train


# In[18]:


from xgboost import XGBClassifier
xgb = XGBClassifier()


# In[19]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost


# In[20]:


param_grid={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
}
               


# In[21]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[22]:


xgb = XGBClassifier( n_estimators=100,booster='gbtree')


# In[23]:


random_search=RandomizedSearchCV(xgb,param_distributions=param_grid,n_iter=5,scoring='accuracy',n_jobs=-1,cv=5,verbose=3)


# In[24]:


from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search_fit =random_search.fit(x,label_encoded_y)
timer(start_time) # timing ends here for "start_time" variable


# In[25]:


#optimized_tree = GridSearchCV(xgb, param_grid, scoring = 'accuracy', cv = 5, verbose=3)


# In[26]:


random_search_fit.fit(x_train, y_train)


# In[27]:


y_pred = random_search_fit.predict(x_test)
#y_pred_labelencoded = list(label_encoder.inverse_transform(y_pred))


# In[28]:


x_test


# In[29]:


y_pred


# In[30]:


from sklearn import metrics


# In[31]:


metrics.accuracy_score(y_test,y_pred)


# In[32]:


# from sklearn.model_selection import GridSearchCV


# In[33]:


# param_grid = { "criterion"      : ['gini', 'entropy'],    
#                "max_features"   : [2,3,4,5,6,7],             
#                "splitter"       : ['best', 'random'],  
#                "min_samples_leaf" : [2,3,5],
#                "max_depth"        : [1,2,3,4,5,6],
              
#                 }


# In[34]:


# optimized_tree = GridSearchCV(ctree, param_grid, scoring = 'accuracy', cv=15,verbose = 2)  #inplace of accuracy use scoring = 'recall'


# In[35]:


# optimized_tree.fit(x_train, y_train)


# In[36]:


# final_model = optimized_tree.best_estimator_
# final_model


# In[37]:


# y_test_pred = final_model.predict(x_test)
# y_test_pred


# In[38]:


#metrics.accuracy_score(y_test,y_test_pred)


# In[39]:


# aero_data_test = pd.read_csv(r"C:\Users\varun\flight_dataset\test.csv")

# aero_data_test = aero_data_test.drop('Accident_ID',axis=1)

# aero_data_test

# y_test_aero = random_search_fit.predict(aero_data_test)

# y_test_aero

# final_model = pd.DataFrame(columns={'Severity'})
# final_model['Severity'] =  y_test_aero
# final_model

# my_dict = {}

# #final_model.to_excel

# final_model['Severity'].value_counts()

# final_model.to_csv('aeroplane2.csv')


# In[40]:


import pickle

pickle.dump(random_search_fit,open('severity.pk1','wb'))


# In[ ]:


# model1 =  pickle.load(open("Aeroplane_severity_le.pk1", "rb"))


# In[ ]:




