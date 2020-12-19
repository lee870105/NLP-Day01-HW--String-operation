#!/usr/bin/env python
# coding: utf-8

# In[1]:


#搭建一個TFIDF模型
import nltk
#nltk.download()
documentA = 'the man went out for a walk'
documentB = 'the children sat around the fire'


# In[2]:


#首先我們做tokenize，並取出所有文件中的單詞
tokenize_A = nltk.word_tokenize(documentA)
tokenize_B = nltk.word_tokenize(documentB)

uniqueWords = set(tokenize_A).union(set(tokenize_B)) ##所有文件中的單詞
uniqueWords


# In[3]:


numOfWordsA = dict.fromkeys(uniqueWords, 0)
numOfWordsB = dict.fromkeys(uniqueWords, 0)
print(numOfWordsA,',',numOfWordsB)


# In[4]:


#計算每個文件中，所有uniqueWords出現的次數
for word in tokenize_A:
    numOfWordsA[word] += 1
for word in tokenize_B:
    numOfWordsB[word] += 1


# In[6]:


print(numOfWordsA,',',numOfWordsB)


# In[7]:


def computeTF(wordDict, tokenize_item):
    """
    wordDict : 文件內單詞對應出現數量的字典
    tokenize_item : 文件tokenize後的輸出
    """
    tfDict = {}
    bagOfWordsCount = len(tokenize_item) ## tokenize_item單詞數量
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount) ##單詞在該文件出現的次數/該文件擁有的所有單詞數量
    return tfDict


# In[8]:


#定義function:計算IDF
def computeIDF(documentsDict):
    """
    documentsDict:為一個list，包含所有文件的wordDict
    """
    import math
    N = len(documentsDict)
    
    idfDict = dict.fromkeys(documentsDict[0].keys(), 0)
    for document in documentsDict:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1 ## 計算單詞在多少文件中出現過
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val)) ## 計算IDF，Log (所有文件的數目/包含這個單詞的文件數目)
    return idfDict


# In[9]:


#定義function:計算TFIDF
def computeTFIDF(tf_item, idfs):
    tfidf = {}
    for word, val in tf_item.items():
        tfidf[word] = val * idfs[word]
    return tfidf


# In[11]:


tfA = computeTF(numOfWordsA, tokenize_A)
tfB = computeTF(numOfWordsB, tokenize_B)

idfs = computeIDF([numOfWordsA, numOfWordsB])


tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)
print(tfidfA,',',tfidfB)


# In[ ]:




