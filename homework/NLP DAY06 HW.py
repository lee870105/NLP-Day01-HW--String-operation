#!/usr/bin/env python
# coding: utf-8

# In[1]:


#這份作業我們會使用中文飯店評論資料集來作為斷詞練習。
import pandas as pd 
import jieba
df = pd.read_csv('Z:\\研究所\\自然語言馬拉松\\DAY06\\ChnSentiCorp_htl_all.csv')
df.head(5)


# In[2]:


pd_positive = df.loc[df['label'] == 1,:]### .loc先行後列，中間用逗號（,）分割
pd_negative = df.loc[df['label'] == 0,:]

print(f'Total: {len(df)}, Positive: {len(pd_positive)}, Negative: {len(pd_negative)}')


# In[3]:


#移除缺失值
df.dropna(inplace=True)


# In[4]:


#建構結巴斷詞Function
#建構可將讀入的Pandas DataFrame的文本資料，外加一欄名為cut的review欄位斷詞結果
class JiebaCutingClass(object):
    '''Class to use jeiba to parse corpus from dataframe and cut the corpus
    
    Parameters
    -----------
    key_to_cut: str
        the dataframe key to parse the sentence for jieba cutting
    dic: str
        the dictionary to use for jieba, default is None (use default dictionary)
    userdict: str
        the user defined dictionary to use for jieba, default is None
    '''
    
    def __init__(self, key_to_cut:str, dic:str=None, userdict:str=None):
        
        if dic is not None:
            jieba.set_dictionary(dic)
        
        if userdict is not None:
            jieba.load_userdict(userdict)
        
        self.key_to_cut = key_to_cut
        
        jieba.enable_paddle() #將paddle 功能開啟
        
    @staticmethod
    def cut_single_sentence(sentence, use_paddle=False, use_full=False, use_search=False):
        
        if use_search:
            out = jieba.cut_for_search(sentence)
        else:
            out = jieba.cut(sentence, use_paddle=use_paddle, cut_all=use_full)
        
        return out
            
    
    def cut_corpus(self, corpus: pd.DataFrame, mode: str) -> pd.DataFrame:
        '''Method to read and cut sentence from dataframe and append another column named cut
        
        Paremeters
        --------------
        corpus: pd.DataFrame
            Input corpus in dataframe
        mode: str
            Jieba mode to be used
        
        Return
        ----------------
        out: pd.Dataframe
            Output corpus in dataframe
        '''
        
        # checking valid mode
        if mode not in ['paddle', 'full', 'precise', 'search']:
            raise TypeError(f'only support `paddle`, `full`, `precise`, and `search` mode, but get {mode}')
            
        # cut the corpus based on mode
        if mode == 'paddle':
            out = self._paddle_cut(corpus)
        elif mode == 'full':
            out = self._full_cut(corpus)
        elif mode == 'precise':
            out = self._precise_cut(corpus)
        elif mode == 'search':
            out = self._search_cut(corpus)

        return out
    
    def _paddle_cut(self, corpus):
        '''paddle mode
        '''
        #enable paddle
        jieba.enable_paddle() 
        
        out = []
        for single_review in corpus[self.key_to_cut]:
            out.append([word for word in JiebaCutingClass.cut_single_sentence(single_review, use_paddle=True)])
        
        corpus['cut'] = out#out[]放入corpus且命名為cut
        
        return corpus
    
    def _full_cut(self, corpus):
        '''full mode
        '''
        
        out = []
        for single_review in corpus[self.key_to_cut]:
            out.append([word for word in JiebaCutingClass.cut_single_sentence(single_review, use_full=True)])
        
        corpus['cut'] = out
        
        return corpus
    
    def _precise_cut(self, corpus):
        '''precise mode
        '''
        
        out = []
        for single_review in corpus[self.key_to_cut]:
            out.append([word for word in JiebaCutingClass.cut_single_sentence(single_review)])
        
        corpus['cut'] = out
        
        return corpus
    
    def _search_cut(self, corpus):
        '''search mode
        '''
        
        out = []
        for single_review in corpus[self.key_to_cut]:
            out.append([word for word in JiebaCutingClass.cut_single_sentence(single_review, use_search=True)])
        
        corpus['cut'] = out
        
        return corpus


# In[5]:


jieba_cut = JiebaCutingClass(key_to_cut='review')
pd_cut = jieba_cut.cut_corpus(df.loc[:50, :], mode='precise') #為了避免處理時間過久, 這裡我們只使用前50個進行斷詞

pd_cut.head()


# In[6]:


test_string = '我愛cupoy自然語言處理馬拉松課程'

jieba_cut = JiebaCutingClass(key_to_cut='', dic='C:\\Users\\lab506\\Desktop\\jieba-master\\jieba-master\\extra_dict\\dict.txt.big')
out_string = jieba_cut.cut_single_sentence(test_string, use_paddle=True) #paddle 模式
print(f'Paddle模式: {[string for string in out_string]}')

out_string = jieba_cut.cut_single_sentence(test_string, use_full=True) #全模式
print(f'全模式: {[string for string in out_string]}')

out_string = jieba_cut.cut_single_sentence(test_string, use_search=True) #搜尋模式
print(f'搜尋模式: {[string for string in out_string]}')

out_string = jieba_cut.cut_single_sentence(test_string) #精確模式
print(f'精確模式: {[string for string in out_string]}')


# In[ ]:




