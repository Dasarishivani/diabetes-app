#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install xgboost


# In[8]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# In[3]:


df=pd.read_csv('diabetes.csv')
df


# In[4]:


X=df.drop('class',axis=1)
y=df['class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[5]:


scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
print(X_train_scaled)
print("--------------------------------")
print(X_test_scaled)


# In[13]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Initialize XGBClassifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Parameter grid for grid search
param_grid = {
    'n_estimators': [100, 150, 200, 300],
    'learning_rate': [0.01, 0.15],
    'max_depth': [2, 3, 4, 5],
    'subsample': [0.8, 1.0],  # Corrected typo 'sunsample' -> 'subsample'
    'colsample_bytree': [0.8, 1.0]
}

# StratifiedKFold cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV setup
grid_search = GridSearchCV(estimator=xgb,
                           param_grid=param_grid,
                           scoring='recall',
                           cv=skf,
                           verbose=1,
                           n_jobs=-1)


# In[15]:


# Fit the grid search to the scaled training data
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Print the best hyperparameters
print("Best Parameters:", grid_search.best_params_)

# Print the best cross-validated recall score
print("Best Cross-Validated Recall:", grid_search.best_score_)

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test_scaled)


# In[16]:


from sklearn.metrics import confusion_matrix, classification_report

# Print confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[17]:


import pandas as pd

# Create a DataFrame with feature importances
features = pd.DataFrame(best_model.feature_importances_, 
                        index=df.iloc[:, :-1].columns, 
                        columns=["Importances"])

# Sort the features by importance
df1 = features.sort_values(by="Importances", ascending=False)

# Display the sorted DataFrame
df1


# In[18]:


import seaborn as sns
sns.barplot(data=df1,x=features.index,y="Importances",hue=features.index,palette="sell")


# In[ ]:




