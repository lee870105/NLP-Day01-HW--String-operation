#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#TSV文件有別於CSV的文件是使用\ t作為分隔符，CSV則是使用，作為分隔符。
dataset = pd.read_csv(r'Z:\研究所\自然語言馬拉松\DAY01\Restaurant_Reviews.tsv', sep='\t')
all_review = dataset['Review'].values


# In[2]:


pratice_sentence = all_review[0]#讀取al_review的第一個句子
## format 用法之後會有詳細講解，這裡先了解會將後方字串放入{}即可
print('原始字串: {}'.format(pratice_sentence))
## 運用len()得到字串長度
print('字串長度: {}'.format(len(pratice_sentence)))


# In[3]:


#計算有多少個句子是以 . 結尾
n=0
for sentence in all_review:#sentence逐一叫出all_review的每個句子
    if sentence.endswith('.'):
        n+=1
print('There are {} sentences end with .'.format(n))


# In[4]:


#將所有. 換成 ,
for sentence_number in range(len(all_review)):#運用len()得到all_review句子的數目(為1000個句子)，sentence_number為0到999的數字
    all_review[sentence_number] = all_review[sentence_number].replace('.',',')
all_review


# In[5]:


#將所有sentence 中的第一個 the 置換成 The
for sentence_number in range(len(all_review)):
    input_sentence = all_review[sentence_number]
    if 'the' in input_sentence:#若input_sentence有the
        location = input_sentence.find('the')#用來尋找字串中字元所在位置，index()不同於find()，當字元不存在時，會報錯，find()則會返回-1
        input_sentence = ''.join((input_sentence[:location],'T',input_sentence[location+1:]))
        all_review[sentence_number] = input_sentence
all_review


# In[6]:


#將偶數句子全部轉換為大寫，基數句子全部轉換為小寫
for sentence_number in range(len(all_review)):
    if sentence_number%2 == 0:#它基本上可以測試數字是否為奇數，除2等於0代表為偶數並改成大寫
        all_review[sentence_number] = all_review[sentence_number].upper()
    else:
        all_review[sentence_number] = all_review[sentence_number].lower()
all_review


# In[7]:


#將所有句子合併在一起，並以' / ' 為間隔
print('/'.join(all_review))

