#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


def read_training_data(filepath):
    '''Reads filepath into Pandas DataFrame'''
    raw_data = pd.DataFrame()
    try:
        raw_data = pd.read_csv(
            filepath, 
            names=['attribute', 'ID', 'ignore', 'title', 'url'], 
            engine="python", 
            skiprows=7
        )
    except Exception as E:
        print(E)

    return raw_data


# In[5]:


data = read_training_data('anonymous-msweb.data')
data


# In[15]:


def create_attributes(dataframe):
    attributes = dataframe[dataframe['attribute'] == 'A']
    attributes = attributes[['ID', 'title', 'url']]

    return attributes

def create_user(dataframe):
    users = pd.read_csv("anonymous-msweb.data",header=None,skiprows=range(0,301), names=['attribute', 'ID','ignored'])
    user = []
    record = []
    temp_sequence = None
    for index, row in users.iterrows():
        if row['attribute']=='C':
            if temp_sequence!= None:
                record.append(temp_sequence)
            user.append(row['ID'])
            temp_sequence=[]
        else:
            temp_sequence.append(row['ID'])
    record.append(temp_sequence)
    users = pd.DataFrame({"user":user,"visit":record})
    return users


# In[16]:


attributes = create_attributes(data)
users = create_user(data)
users


# In[10]:


attributes


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
record_dict = {}
for index, row in users.iterrows():
    sequence=row['visit']
    for i in sequence:
        if i in record_dict.keys():
            record_dict[i]+=1
        else:
            record_dict[i]=1
attr_list = [i for i in record_dict.keys()]
hit_list = [record_dict[i] for i in attr_list]

hit_df = pd.DataFrame({"attribute":attr_list,"visited time":hit_list})
hit_df.sort_values("visited time",inplace=True,ascending=[False])
hit_df['attribute']= hit_df['attribute'].map(lambda x:str(x))
sns.relplot(x="attribute",y="visited time",data=hit_df,size=3, alpha=0.5).set(yscale='log')
plt.savefig('test1.svg',dpi=600)


# In[ ]:


attributes.sort_values("visited time",inplace=True)
attributes[attributes["visited time"]==0]["title"]
attributes.sort_values("visited time",inplace=True,ascending=[False])
attributes.head[["title","visited time",'url']]


# In[11]:


users["visit len"] = users.apply(lambda x: len(x['visit']), axis=1)
users


# In[15]:


import seaborn as sns
visit_len = users["visit len"].tolist()
visit_len_record = {i:0 for i in range(1,36)}
for i in visit_len:
    visit_len_record[i]+=1
len_df = pd.DataFrame({"frequent":[visit_len_record[i] for i in range(1,36)],"visited length":[i for i in range(1,36)]})
len_df.sort_values("visited length",inplace=True,ascending=[False])
sns.relplot(x="visited length",y="frequent",data=len_df, alpha=0.5)
plt.savefig("test2.jpg",dpi=600)


# In[16]:


import myApriori
l, supportdata = myApriori.apriori(users['visit'],id2name,min_support=0.05)


# In[23]:



def apriori(data:pd.Series,translate_dict,min_support=0.3):
    l1 = set()
    for i in data:
        for j in i:
            l1.add(frozenset([j]))
    l1 = list(l1)
    l1.sort()
    l1,support_data = scan(data,l1,min_support)
    l = [l1]
    k = 2
    print("=========k = 1==========")
    for i in l1:
        print("{:} {:.5f}".format([translate_dict[j] for j in list(i)],support_data[i]/len(data)))
    while len(l[k-2])>0:
        ck = nextCk(l[k-2])
        lk,temp_data = scan(data,ck,min_support)
        l.append(lk)
        support_data.update(temp_data)
        print("=========k = {}==========".format(k))
        for i in l[k-1]:
            print("{:} {:.5f}".format([translate_dict[j] for j in list(i)],support_data[i]/len(data)))
        k+=1
    return l,support_data

def isSubsequence(s: list, t: list):
    b = iter(t)
    return all(((i in b) for i in s ))

def mergeSeq(l1:tuple,l2:tuple):
    # check lenth
    if len(l1)!=len(l2) or len(l1)==0:
        return None
    l1 = list(l1)
    l2 = list(l2)
    # check difference
    length = len(l1)
    sub = [i for i in l1 if i in l2]
    if len(sub)!=length-1:
        return None
    # find difference pos
    pos1 = 0 # first difference pos
    pos2 = length-1  # second difference pos
    while l1[pos1]==l2[pos1]:
        pos1+=1
    while l1[pos2]==l2[pos2]:
        pos2-=1
    # get new sequence
    ans = []
    if pos1==pos2:
        ans.append(l1[:pos1+1]+[l2[pos1]]+l1[pos1+1:])
        ans.append(l1[:pos1]+[l2[pos1]]+l1[pos1:])
    else:
        if l1[pos1+1]==l2[pos1]:
            ans.append(l1[:pos2+1]+[l2[pos2]]+l1[pos2+1:])
        else:
            ans.append(l2[:pos2+1]+[l1[pos2]]+l2[pos2+1:])
    ans = [tuple(i) for i in ans]
    return ans



def seq_scan(data:pd.Series,ck:list,min_support=0.5):
    hit_times = {}
    for l in data:
        for cur_set in ck:
            if isSubsequence(list(cur_set),list(l)):
                if cur_set in hit_times.keys():
                    hit_times[cur_set]+=1
                else:
                    hit_times[cur_set]=1
    threshold = len(data)*min_support
    return [i for i in hit_times.keys() if hit_times[i]>threshold],hit_times

def seq_nextCk(lk:list):
    len_k = len(lk)
    if (len_k==0) :
        return None
    res_list = []
    for i in range(len_k):
        for j in range(len_k):
            new_seq = mergeSeq(lk[i],lk[j])
            if new_seq!=None:
                for k in new_seq:
                    res_list.append(k)
    return list(set(res_list))

def seq_apriori(data:pd.Series,translate_dict,min_support=0.5):
    l1 = set()
    for i in data:
        for j in i:
            l1.add(tuple([j]))
    l1 = list(l1)
    l1.sort()
    l1 = list(l1)
    l1.sort()
    l1,support_data = seq_scan(data,l1,min_support)
    l = [l1]
    k = 2
    print("=========k = 1==========")
    for i in l1:
        print("{:} {:.5f}".format([translate_dict[j] for j in list(i)],support_data[i]/len(data)))
    while len(l[k-2])>0:
        ck = seq_nextCk(l[k-2])
        lk,temp_data = seq_scan(data,ck,min_support)
        l.append(lk)
        support_data.update(temp_data)
        print("=========k = {}==========".format(k))
        for i in l[k-1]:
            print("{:} {:.5f}".format([translate_dict[j] for j in list(i)],support_data[i]/len(data)))
        k+=1
    return l,support_data

#l, supportdata = apriori(user_df['visit'],min_support=0.05)

def scan(data:pd.Series,ck:list,min_support=0.5):
    hit_times = {}
    for l in data:
        for cur_set in ck:
            if set(cur_set).issubset(set(l)):
                if cur_set in hit_times.keys():
                    hit_times[cur_set]+=1
                else:
                    hit_times[cur_set]=1
    threshold = len(data)*min_support
    return [i for i in hit_times.keys() if hit_times[i]>threshold],hit_times

def nextCk(lk:list):
    k = len(lk)
    if (k==0) :
        return None
    res_list = []
    for i in range(k):
        for j in range(i+1, k):
            l1 = list(lk[i])[:-2]
            l2 = list(lk[j])[:-2]
            if l1==l2:
                res_list.append(lk[i]|lk[j])
    return list(set(res_list))


# In[21]:


id2name = pd.Series(attributes['title'].values,index=attributes['ID']).to_dict()
id2name


# In[24]:


l, supportdata = apriori(users['visit'],id2name,min_support=0.05)


# In[25]:


l, supportdata = apriori(users['visit'],id2name,min_support=0.03)


# In[26]:


l, supportdata = seq_apriori(users['visit'],id2name,min_support=0.04)


# In[ ]:




