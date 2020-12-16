#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
import nltk
#nltk.download('wordnet')
## 創建stemmer
ps=PorterStemmer()
## 創建Lemmatizer
lemmatizer = WordNetLemmatizer() 


# In[2]:


print('Stemming amusing : {}'.format(ps.stem('amusing')))
print('lemmatization amusing : {}'.format(lemmatizer.lemmatize('amusing',pos = 'v')))


# In[3]:


# Define the sentence to be lemmatized
sentence = "The striped bats are hanging on their feet for best"

# Tokenize: Split the sentence into words
word_list = nltk.word_tokenize(sentence)
print(word_list)


# In[4]:


# stemming提取每個單詞的詞幹
stemming_output = ' '.join([ps.stem(w) for w in word_list])
print(stemming_output)


# In[5]:


# lemmatize提取每個單詞的lemma
lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
print(lemmatized_output)


# In[6]:


#有時單詞的lemma會隨著詞性而有所改變
print('lemmatization amusing : {}'.format(lemmatizer.lemmatize('amusing',pos = 'v'))) ##動詞
print('lemmatization amusing : {}'.format(lemmatizer.lemmatize('amusing',pos = 'a'))) ##形容詞


# In[22]:


# Lemmatize with POS Tag
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """將pos_tag結果mapping到lemmatizer中pos的格式"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
word = 'using'
print(lemmatizer.lemmatize(word, get_wordnet_pos(word)))


# In[13]:


sentence = "The striped bats are hanging on their feet for best"
print([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])


# In[ ]:




