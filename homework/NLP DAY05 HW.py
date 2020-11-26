#!/usr/bin/env python
# coding: utf-8

# In[1]:


#了解斷詞演算法的背後計算
#若有一個人連續觀察到三天水草都是乾燥的(Dry), 則這三天的天氣機率為何？(可參考講義第13頁) (Hint: 共有8種可能機率)
observations = ('dry', 'dry', 'dry') #實際上觀察到的狀態序列為dry, dry, dry
states = ('sunny', 'rainy')#隱層的狀態序列
start_probability = {'sunny': 0.4, 'rainy': 0.6}#狀態初始概率
transition_probability = {'sunny':{'sunny':0.6, 'rainy':0.4},
                          'rainy': {'sunny':0.3, 'rainy':0.7}}#轉移矩陣
emission_probatility = {'sunny': {'dry':0.6, 'damp':0.3, 'soggy':0.1},
                        'rainy': {'dry':0.1, 'damp':0.4, 'soggy':0.5}}#發射矩陣


# In[2]:


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1,len(obs)):
        print(len(obs))
        V.append({})
        newpath = {}

        for cur_state in states:
            (prob, state) = max([(V[t-1][pre_state] * trans_p[pre_state][cur_state] * emit_p[cur_state][obs[t]], pre_state) for pre_state in states])
            V[t][cur_state] = prob
            newpath[cur_state] = path[state] + [cur_state]

        # Don't need to remember the old paths
        path = newpath
    
    (prob, state) = max([(V[len(obs) - 1][final_state], final_state) for final_state in states])
    return (prob, path[state])


# In[3]:


result = viterbi(observations,
                 states,
                 start_probability,
                 transition_probability,
                 emission_probatility)


# In[4]:


result


# In[ ]:




