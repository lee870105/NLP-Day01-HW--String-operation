#!/usr/bin/env python
# coding: utf-8

# In[1]:


#參考課程實作並在datasets_483_982_spam.csv的資料集中獲得90% 以上的 accuracy (testset)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import codecs
import re


# In[2]:


dataset = pd.read_csv(r'Z:\研究所\自然語言馬拉松\DAY18 DAY19 DAY20\datasets_483_982_spam.csv',encoding='latin-1')
dataset


# In[3]:


dataset.loc[dataset.loc[:, "v1"] == "spam", "v1"] = 1
dataset.loc[dataset.loc[:, "v1"] == "ham", "v1"] = 0
dataset.loc[:,"v1"]


# In[12]:


all_data=[]
for content,label in dataset[['v2','v1']].values:
    all_data.append([content, label])
all_data = np.array(all_data)
X = all_data[:,0]
Y = all_data[:,1].astype(np.uint8)


# In[5]:


print('Training Data Examples : \n{}'.format(X[:5]))


# In[6]:


#文字預處理
from nltk.stem.porter import PorterStemmer
corpus=[]
row=len(X)
for i in range(0,row):
    review=re.sub('[^a-zA-Z]',' ',X[i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    ## 這裡先不用stopwords 因為 review中很多反定詞會被移掉 如isn't good, 會變成 good
    review=[ps.stem(word) for word in review ]
    review=' '.join(review)
    corpus.append(review)
corpus


# In[7]:


#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
#max_features是要建造幾個column，會按造字出現的頻率高低去篩選，1500並沒有特別含義，大家可以自己嘗試不同數值或不加入限制
cv=CountVectorizer()# or cv=CountVectorizer(max_features = 1500)
X=cv.fit_transform(corpus).toarray()


# In[8]:


X.shape


# In[9]:


#將資料拆成 train/test set
from sklearn.model_selection import train_test_split
# random_state 是為了讓各為學員得到相同的結果，平時可以移除
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[11]:


#Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)#metric：預設為"minkowski"(明可夫斯基距離);p為2時所使用的是曼哈頓距離：兩點絕對值距離,p為1時所使用的是歐式距離
classifier.fit(X_train, y_train)


# In[14]:


y_pred = classifier.predict(X_test)
#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
f,ax=plt.subplots()
cm = confusion_matrix(y_test, y_pred)#y_test 真實的label，一維數組格式，縱軸 ; y_pred 模型預測的label，一維數組，橫軸
print(cm)
sns.heatmap(cm,annot=True,ax=ax) #熱力圖
 
ax.set_title('confusion matrix') #標題
ax.set_xlabel('predict') #x軸
ax.set_ylabel('true') #y軸
accuracy_score(y_test, y_pred)


# In[ ]:




