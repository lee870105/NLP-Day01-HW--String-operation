#!/usr/bin/env python
# coding: utf-8

# In[1]:


#isnumeric(), isdigit(), isdecimal() 各有幾個
test_string = ['5.9', '30', '½', '³', '⑬']


# In[2]:


def spam(s, isnumeric_count, isdigit_count, isdecimal_count):
    for attr in ['isnumeric', 'isdecimal', 'isdigit']:
        if attr == 'isnumeric':
            if getattr(s, attr)():#獲取attr=isnumeric屬性，成立的話isumeric_count+1
                isnumeric_count+=1
        elif attr == 'isdecimal':
            if getattr(s, attr)():
                isdecimal_count+=1
        elif attr == 'isdigit':
            if getattr(s, attr)():
                isdigit_count+=1
    return isnumeric_count, isdigit_count, isdecimal_count


# In[3]:


isnumeric_count = 0
isdigit_count = 0 
isdecimal_count = 0

for s in test_string:
    isnumeric_count, isdigit_count, isdecimal_count = spam(s, isnumeric_count, isdigit_count, isdecimal_count)
    #呼叫spam 並且計算完傳回 isnumeric_count, isdigit_count, isdecimal_count
print('isnumeric_count: {}'.format(isnumeric_count))     
print('isdigit_count: {}'.format(isdigit_count))     
print('isdecimal_count: {}'.format(isdecimal_count))


# In[10]:


#運用formatting 技巧 output
#輸出小數點下兩位 Accuracy: 98.13%, Recall: 94.88%, Precision: 96.29%
accuracy = 98.129393
recall =   94.879583
precision =96.294821
print('accuracy:{:.2f}%'.format(accuracy),'recall:{:.2f}%'.format(recall),'precision:{:.2f}%'.format(precision))


# In[13]:


#依照只是轉換number output format
#轉換為科學符號表示法 (小數點後兩位);轉換為%;補零成為3.14159300
number = 3.1415926
print('科學符號:{:.2e};'.format(number),'轉換%:{:.2%};'.format(number),'補零:{:0<10f};'.format(number))


# In[ ]:




