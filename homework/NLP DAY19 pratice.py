#!/usr/bin/env python
# coding: utf-8

# In[1]:


#運用scikit-learn API 實現K-fold分割資料
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
dataset = pd.read_csv('Z:\研究所\自然語言馬拉松\DAY18 DAY19 DAY20\Social_Network_Ads.csv')
dataset


# In[2]:


X = dataset[['User ID', 'Gender', 'Age', 'EstimatedSalary']].values
Y = dataset['Purchased'].values


# In[3]:


#將訓練資料按照順序切割成10等分
kf = KFold(n_splits=10, shuffle=False)
kf.get_n_splits(X)

print(kf)


# In[4]:


#將訓練資料隨機切割成10等分
kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(X)

print(kf)


# In[7]:


#取出 切割資料對應位置
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)


# In[10]:


#取出切割資料
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
print(X_train,',',X_test)


# In[ ]:




