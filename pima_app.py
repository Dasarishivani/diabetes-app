#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install joblib 


# In[8]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib


# In[18]:


df = pd.read_csv("diabetes.csv")
df


# In[19]:


X = df.drop('class',axis=1)
Y = df['class']


# In[21]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[22]:


model=DecisionTreeClassifier(random_state=42)
kf=KFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(model,X_scaled,y,cv=kf)
print("Cross_validation scores:", scores)
print("Mean accuracy:",scores.mean())


# In[23]:


model.fit(X_scaled,y)
joblib.dump(model,'model.pkl')
joblib.dump(scaler,'scaler.pkl')


# In[ ]:




