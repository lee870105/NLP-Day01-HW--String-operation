#!/usr/bin/env python
# coding: utf-8

# In[1]:


#讀取文本資料
import pandas as pd
import re
import email
with open('Z:\研究所\自然語言馬拉松\DAY03 DAY04\sample_emails.txt', 'r', encoding="utf8", errors='ignore') as f:
    sample_corpus = f.read()#open('檔案','模式') 開啟文件，r : 讀取;f.read(size): 將文字讀取成string(若換行會包含\n)，其中size為要讀取長度，若不填則讀取全部。


# In[2]:


sample_corpus


# In[3]:


pattern = r'From:.*'
match = re.findall(pattern, sample_corpus)
match


# In[4]:


pattern = r'\".*\"'

for info in match:
    print(re.search(pattern, info).group())


# In[5]:


pattern = r'\w\S*@.*\b' #\b 是因為結尾一定為com

for info in match:
    print(re.search(pattern, info).group())


# In[6]:


pattern = r'(?<=@)\w*'

for info in match:
    print(re.search(pattern, info).group())


# In[7]:


for ad in match:
    for line in re.findall(r'\w\S*@.*(?=\.)', ad):
        username, domain_name = re.split("@", line)
        print("{}, {}".format(username, domain_name))


# In[8]:


###讀取文本資料###
with open('C:\\Users\\lab506\\Desktop\\all_emails.txt', 'r', encoding="utf8", errors='ignore') as f:
    corpus = f.read()
###若遇到 SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \UXXXXXXXX escape 把\改成\\
###切割讀入的資料成一封一封的email###
###我們可以使用list來儲存每一封email###
###注意！這裡請仔細觀察sample資料，看資料是如何切分不同email###
emails = re.split(r"From r", corpus, flags=re.M)#使用多行配對模式
emails = emails[1:] #移除第一項的空元素
len(emails) #查看有多少封email


# In[9]:


emails_list = [] #創建空list來儲存所有email資訊
for mail in emails[:20]: #只取前20筆資料 (處理速度比較快)
    emails_dict = dict() #創建空字典儲存資訊
    ###取的寄件者姓名與地址###
    
    #Step1: 取的寄件者資訊 (hint: From:)
    sender = re.search(r"From:.*", mail)
    
    #Step2: 取的姓名與地址 (hint: 要注意有時會有沒取到配對的情況)
    if sender is not None: #有取到配對
        sender_mail = re.search(r"\w\S*@.*\b", sender.group())
        sender_name = re.search(r"(?<=\").*(?=\")", sender.group())
    else: #沒取到配對
        sender_mail = None
        sender_name = None

    #Step3: 將取得的姓名與地址存入字典中
    if sender_mail is not None:
        emails_dict["sender_email"] = sender_mail.group()#使用.group() or .group(0)返回配對的字串
    else:
        emails_dict["sender_email"] = sender_mail #None
    
    if sender_name is not None:
        emails_dict["sender_name"] = sender_name.group()
    else:
        emails_dict["sender_name"] = sender_name #None
        
    
    ###取的收件者姓名與地址###
    #Step1: 取的寄件者資訊 (hint: To:)
    recipient = re.search(r"To:.*", mail)
    
    #Step2: 取的姓名與地址 (hint: 要注意有時會有沒取到配對的情況)
    if recipient is not None:
        r_email = re.search(r"\w\S*@.*\b", recipient.group())
        r_name = re.search(r"(?<=\").*(?=\")", recipient.group())
    else:
        r_email = None
        r_name = None
        
    #Step3: 將取得的姓名與地址存入字典中
    if r_email is not None:
        emails_dict["recipient_email"] = r_email.group()
    else:
        emails_dict["recipient_email"] = r_email #None
    
    if r_name is not None:
        emails_dict["recipient_name"] = r_name.group()
    else:
        emails_dict["recipient_name"] = r_name #None
        
        
    ###取得信件日期###
    #Step1: 取得日期資訊 (hint: To:)
    date_info = re.search(r"Date:.*", mail)
    
    #Step2: 取得詳細日期(只需取得DD MMM YYYY)
    if date_info is not None:
        date = re.search(r"\d+\s\w+\s\d+", date_info.group())
    else:
        date = None
        
    #Step3: 將取得的日期資訊存入字典中
    if date is not None:
        emails_dict["date_sent"] = date.group()
    else:
        emails_dict["date_sent"] = date
        
        
    ###取得信件主旨###
    #Step1: 取得主旨資訊 (hint: Subject:)
    subject_info = re.search(r"Subject: .*", mail)
    
    #Step2: 移除不必要文字 (hint: Subject: )
    if subject_info is not None:
        subject = re.sub(r"Subject: ", "", subject_info.group())#把配對到的Subject這個字替換成空白，也就是移除Subject
    else:
        subject = None
    
    #Step3: 將取得的主旨存入字典中
    emails_dict["subject"] = subject
    
    
    ###取得信件內文###
    #這裡我們使用email package來取出email內文 (可以不需深究，本章節重點在正規表達式)
    try:
        full_email = email.message_from_string(mail)
        body = full_email.get_payload()
        emails_dict["email_body"] = body
    except:
        emails_dict["email_body"] = None
    
    ###將字典加入list###
    emails_list.append(emails_dict)


# In[10]:


#將處理結果轉化為dataframe
emails_df = pd.DataFrame(emails_list)
emails_df

