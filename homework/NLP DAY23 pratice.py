#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('Y:\研究所\自然語言馬拉松\DAY21 DAY22 DAY23\Tweets.csv')
data.head(5)


# In[3]:


data[['text','airline_sentiment']]


# In[4]:


data.loc[data.loc[:, "airline_sentiment"] == "negative", "airline_sentiment"] = 0
data.loc[data.loc[:, "airline_sentiment"] == "neutral", "airline_sentiment"] = 1
data.loc[data.loc[:, "airline_sentiment"] == "positive", "airline_sentiment"] = 2
Y=data.loc[:,"airline_sentiment"]
Y = np.array(Y, dtype=int)
Y


# In[5]:


## 去除開頭航空名稱 ex. @VirginAmerica
X = data['text'].apply(lambda x: ' '.join(x.split(' ')[1:])).values
X


# In[6]:


str1 = 'My name is Chris Chen.'
str2 = '192.168.43.43'
str3 = 'DF-664-897-99'

list1 = str1.split(' ')[1:]
print(list1)


# In[7]:


#文字預處理
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
import re
import nltk
nltk.download('stopwords')
# Lemmatize with POS Tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
## 創建Lemmatizer
lemmatizer = WordNetLemmatizer() 
def get_wordnet_pos(word):
    """將pos_tag結果mapping到lemmatizer中pos的格式"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
def clean_content(X):
    # remove non-alphabet characters
    X_clean = [re.sub('[^a-zA-Z]',' ', str(x)).lower() for x in X]
    # tokenize
    X_word_tokenize = [nltk.word_tokenize(x) for x in X_clean]
    # stopwords_lemmatizer
    X_stopwords_lemmatizer = []
    stop_words = set(stopwords.words('english'))#創建英文停用字的字典
    for content in X_word_tokenize:
        content_clean = []
        for word in content:
            if word not in stop_words:
                word = lemmatizer.lemmatize(word, get_wordnet_pos(word))
                content_clean.append(word)
        X_stopwords_lemmatizer.append(content_clean)
    
    X_output = [' '.join(x) for x in X_stopwords_lemmatizer]
    
    return X_output


# In[8]:


X = clean_content(X)
X


# In[9]:


#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
#max_features是要建造幾個column，會按造字出現的頻率高低去篩選，3600並沒有特別含義(筆者測試幾次最佳結果)
#大家可以自己嘗試不同數值或不加入限制
cv=CountVectorizer(max_features = 3600)
X_T=cv.fit_transform(X).toarray()
# 有 14640 個樣本，每個樣本用3600維表示
X_T.shape 


# In[10]:


#將資料拆成 train/test set
from sklearn.model_selection import train_test_split
# random_state 是為了讓各為學員得到相同的結果，平時可以移除
X_train, X_test, y_train, y_test = train_test_split(X_T, Y, test_size = 0.2, random_state = 0)


# In[11]:


#Naive Bayes
from sklearn.naive_bayes import MultinomialNB#多項式貝氏分類器主要用在離散變數，比方說次數、類別。
from sklearn.naive_bayes import BernoulliNB#伯努力貝氏分類器主要適用於二元的特徵，比方說特徵是否出現。
from sklearn.naive_bayes import GaussianNB#高斯分類器主要用於特徵為連續變數時，比方說特徵長度為幾公分、重量為幾公斤等等。
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

clf_M = MultinomialNB()
clf_M.fit(X_train, y_train)


# In[12]:


#畫出 Confusion Matrix 結果
import numpy as np
from sklearn.metrics import confusion_matrix
def plot_cm_output(cm, labels=['negative', 'neutral', 'positive']):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm,cmap=plt.cm.Blues,)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# In[13]:


import itertools
#import matplotlib.pyplot as plt
# 繪製混淆矩陣
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 計算出的混淆矩陣的值
    - classes : 混淆矩阵中每一行每一列對應的列
    - normalize : True:顯示百分比, False:顯示個數
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[14]:


output_ = clf_M.predict(X_test)
cm_output = confusion_matrix(y_test, output_)
#print(cm_output)
attack_types = ['negative', 'neutral', 'positive']
plot_confusion_matrix(cm_output, classes=attack_types, normalize=False, title='Confusion matrix of the classifier')


# In[15]:


from sklearn.metrics import confusion_matrix, accuracy_score
print("accuracy score=",accuracy_score(y_test, output_))


# In[ ]:




