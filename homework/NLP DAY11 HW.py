#!/usr/bin/env python
# coding: utf-8

# In[1]:


#自行尋找一篇超過100字的文章，首先運用結巴斷詞，自行新增字典使得斷詞更為正確，使用jieba 完成 PoS Taggin，新增的詞也必須賦予詞性
import pandas as pd 
import jieba
import jieba.posseg as pseg
with open('Z:\研究所\自然語言馬拉松\DAY11\練習用文章.txt',encoding="utf-8") as f:
     content_list = f.read()
print(content_list)


# In[2]:


jieba.set_dictionary('C:\\Users\\lab506\\Desktop\\jieba-master\\jieba-master\\extra_dict\\dict.txt.big')    #詞庫
jieba.load_userdict('Z:\\研究所\\自然語言馬拉松\\DAY11\\userdict.txt')        #自定義使用者字典
cut_word= pseg.cut(content_list)
words=[(word, flag) for (word, flag) in cut_word]
print(words)

