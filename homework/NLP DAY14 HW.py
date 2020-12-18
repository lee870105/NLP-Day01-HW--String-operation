#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
import jieba
import nltk
from collections import Counter
import re
#tsv是指用tab分開字元的檔案
dataset=pd.read_csv('Z:\研究所\自然語言馬拉松\DAY14\movie_feedback.csv', header=None, encoding='Big5')
X = dataset[0].values
Y = dataset[1].values
dataset


# In[2]:


print('review before preprocessing : {}'.format(X[0]))


# In[3]:


# 去除a-zA-Z以外的字元，並將他們取代為空格' '
review=re.sub('^a-zA-Z','',X[0])
review


# In[4]:


#把全部變成小寫
review=review.lower()
print('review after lower : {}'.format(review))


# In[6]:


review = nltk.word_tokenize(review)
print('review after tokenize : {}'.format(review))


# In[7]:


#stopwords.words('english') 是一個建立好的list，包含一些常見的英文贅字
review=[word for word in review if not word in set(stopwords.words('english'))]#如果不是贅字則提取
print('review after removeing stopwords : {}'.format(review))


# In[8]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
review=[ps.stem(word) for word in review]
print('review after stemming : {}'.format(review))


# In[10]:


#練習清理所有的句子
X = dataset[0].values
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


# In[12]:


#轉bag-of-words vector
from sklearn.feature_extraction.text import CountVectorizer
#Creating bag of word model
#tokenization(符號化)
from sklearn.feature_extraction.text import CountVectorizer
#max_features是要建造幾個column，會按造字出現的高低去篩選 
cv = CountVectorizer(max_features=1500)
#toarray是建造matrixs
#X現在為sparsity就是很多零的matrix
X_ = cv.fit_transform(corpus).toarray()
Y_ = dataset[1].values


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size = 0.1)

# Feature Scaling

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

message='I like you!!'
## 要使用一樣的前處理
review=re.sub('[^a-zA-Z]',' ',message)
review=review.lower()
review=review.split()
ps=PorterStemmer()
review=[ps.stem(word) for word in review]
review = ' '.join(review)
input_ = cv.transform([review]).toarray()
prediction = classifier.predict(input_)
prediction


# In[ ]:




