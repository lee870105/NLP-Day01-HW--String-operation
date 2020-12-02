#!/usr/bin/env python
# coding: utf-8

# In[7]:


#了解N-Gram如何藉由文本計算機率
#以Bigram模型下判斷語句是否合理
#已知的機率值有
#p(i|start) = 0.25
#p(english|want) = 0.0011
#p(food|english) = 0.5
#p(end|food) = 0.68
#P(want|start) = 0.25
#P(english|i) = 0.0011
import numpy as np
import pandas as pd
words = ['i', 'want', 'to', 'eat', 'chinese', 'food', 'lunch', 'spend']
word_cnts = np.array([2533, 927, 2417, 746, 158, 1093, 341, 278]).reshape(1, -1)#-1為自動計算行數
df_word_cnts = pd.DataFrame(word_cnts, columns=words)
print(df_word_cnts)


# In[8]:


# 記錄當前字與前一個字詞的存在頻率
#由上表可知當前一個字詞(列)是want的時候, 當前字詞(行)是to的頻率在文本中共有608次
bigram_word_cnts = [[5, 827, 0, 9, 0, 0, 0, 2], [2, 0, 608, 1, 6, 6, 5, 1], [2, 0, 4, 686, 2, 0, 6, 211],
                    [0, 0, 2, 0, 16, 2, 42, 0], [1, 0, 0, 0, 0, 82, 1, 0], [15, 0, 15, 0, 1, 4, 0, 0],
                    [2, 0, 0, 0, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0]]

df_bigram_word_cnts = pd.DataFrame(bigram_word_cnts, columns=words, index=words)
print(df_bigram_word_cnts)


# In[9]:


#請根據給出的總詞頻(df_word_cnts)與bigram模型的詞頻(df_bigram_word_cnts)計算出所有詞的配對機率(ex:p(want|i))
df_bigram_prob = df_bigram_word_cnts.copy()
for word in words:
    df_bigram_prob.loc[word, :] = df_bigram_word_cnts.loc[word, :] / df_word_cnts.loc[0, word]#loc為些列後行讀取loc[word, :]為word這列然後從第一行讀到最後一行
    
df_bigram_prob


# In[17]:


#請根據已給的機率與所計算出的機率(df_bigram_prob), 試著判斷下列兩個句子哪個較為合理
#s1 = “i want english food”
#s2 = "want i english food"
ps1=0.25*df_bigram_prob.iloc[0,1]*0.0011*0.5*0.68
ps2=0.25*df_bigram_prob.iloc[1,0]*0.0011*0.5*0.68
print(ps1,ps2)


# In[ ]:


#P(s1) > P(s2)
#所以S1較為合理

