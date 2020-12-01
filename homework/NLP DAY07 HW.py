#!/usr/bin/env python
# coding: utf-8

# In[3]:


from ckiptagger import WS, POS, NER
ws = WS("F:/data") #建構斷詞
sentence_list = [
    "傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。",
    "美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。",
    "",
    "土地公有政策?？還是土地婆有政策。.",
    "… 你確定嗎… 不要再騙了……",
    "最多容納59,000個人,或5.9萬人,再多就不行了.這是環評的結論.",
    "科長說:1,坪數對人數為1:3。2,可以再增加。",
]
#斷詞
word_sentence_list = ws(
    sentence_list,
    sentence_segmentation = True, # To consider delimiters
    segment_delimiter_set = {",", "。", ":", "?", "!", ";"}) # This is the defualt set of delimiters
print(word_sentence_list)


# In[4]:


#詞性標註
pos = POS("F:/data")

pos_sentence_list = pos(word_sentence_list)
#word_sentence_lsit為已經斷詞過不是原文本
print(pos_sentence_list)


# In[5]:


#命名實體識別
ner = NER("F:/data")

entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
#word_sentence_lsit為已經斷詞過，不是原文本
#pos_sentence_list為已經雌性標註過，不是原文本
print(entity_sentence_list)


# In[8]:


#合併顯示輸出結果
def combine_wandp(word_sentence_list, pos_sentence_list):
    assert len(word_sentence_list) == len(pos_sentence_list)#判斷word_sentence_list和pos_sentence_list是否一樣長
    
    for w, p in zip(word_sentence_list, pos_sentence_list):
        print (f'{w}({p})', end='\u3000')
        
for i, sentence in enumerate(sentence_list):
    print(f'Input sentence: \n {sentence}')
    print('\n')
    
    print(f'Segmentation with PoS: \n')
    combine_wandp(word_sentence_list[i], pos_sentence_list[i])
    
    print ('\n')
    print('Named Entity Recognition:')
    for n in sorted(entity_sentence_list[i]):
        print (n)
    print ('\n')


# In[11]:


#帶入自定義字典
#觀察上面的輸出發現，斷詞結果將年前斷為年、前, 請使用自定義字典使斷詞結果依然維持為年前
from ckiptagger import construct_dictionary
word_to_weight = {
    "年前": 1
}
dictionary = construct_dictionary(word_to_weight)
word_sentence_list = ws(
    sentence_list,
    sentence_segmentation = True, # To consider delimiters
    segment_delimiter_set = {",", "。", ":", "?", "!", ";"}, # This is the defualt set of delimiters
    coerce_dictionary = dictionary)
print(word_sentence_list)


# In[ ]:




