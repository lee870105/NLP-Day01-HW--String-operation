#!/usr/bin/env python
# coding: utf-8

# In[7]:


#實作cosine similarity
#在比較兩個詞向量的相似度時可以使用cosine similarity
#請實作cosine similarity並計算共現矩陣課程範例中you向量([0,1,0,0,0,0,0])與I([0,1,0,1,0,0,0])向量的相似度
import numpy as np
I = np.array([0,1,0,0,0,0,0])
You = np.array([0,1,0,1,0,0,0])

def cos_similarity(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)
    return np.dot(nx, ny)

print(f"Similarity: {cos_similarity(I, You)}")

