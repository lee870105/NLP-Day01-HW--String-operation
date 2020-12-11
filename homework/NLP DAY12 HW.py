#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#nltk.download()


# In[2]:


dataset=pd.read_csv('Z:\研究所\自然語言馬拉松\DAY12\Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
corpus=dataset['Review'].values
whole_words = []
for sentence in corpus:
    tokenized_sentence = nltk.word_tokenize(sentence)
    for word in tokenized_sentence:
        whole_words.append(word)
print(whole_words)


# In[3]:


#移除重複單字
ordered_tokens = set()
result = []
for word in whole_words:
    if word not in ordered_tokens:
        ordered_tokens.add(word)
        result.append(word)
print('共有{}個單字'.format(len(result)))


# In[4]:


#建立字典使每一個單字有對應數值
word_index = {}
index_word = {}
n = 0
for word in result:
    word_index[word] = n 
    index_word[n] = word
    n+=1
word_index


# In[5]:


index_word


# In[6]:


def _get_bag_of_words_vector(sentence, word_index_dic, whole_words):
    sentence = sentence
    vector = np.zeros(len(whole_words))
    for word in nltk.word_tokenize(sentence):
        if word in whole_words:
            vector[word_index[word]]+=1
    return vector


# In[8]:


_get_bag_of_words_vector('Wow... Loved this place.', word_index, whole_words)


# In[ ]:




