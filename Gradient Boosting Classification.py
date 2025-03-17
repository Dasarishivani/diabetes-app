#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[9]:


df=pd.read_csv('diabetes.csv')
df


# In[10]:


X = df.drop('class', axis=1)
y = df['class' ]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[15]:


X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)


# In[23]:


#Instantiate the model and define the parameters
gbc=GradientBoostingClassifier(random_state=42)
#Setup KFold cross-validation
kfold=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#Hyperparameter grid
param_grid={
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0]
}

#Grid search with cross-validation
grid_search=GridSearchCV(estimator=gbc,
                         param_grid=param_grid,
                         cv=kfold,
                         scoring='recall',
                         n_jobs=-1,
                         verbose=1)


# In[24]:


grid_search.fit(X_train,y_train)
print("Best parameters:",grid_search.best_params_)
print("Best Cross-Validated Recall:",grid_search.best_score_)


# In[26]:


best_model=grid_search.best_estimator_
y_pred=best_model.predict(X_test)
print("\nClassification Report:\n",confusion_matrix(y_test,y_pred))


# In[ ]:




