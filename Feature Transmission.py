#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np #mathematical library
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# In[35]:


df= pd.read_csv('supershops.csv') #import csv


# In[14]:


df.shape #


# In[30]:


df2 = df.copy()
df3 = df.copy()
df4 = df.copy()
df5 = df.copy()
df6 = df.copy()
df7 = df.copy()


# In[33]:


m= MinMaxScaler();


# In[40]:


df[['Marketing_Spend_New']] = m.fit_transform(df[['Marketing Spend']]) #fit and add the column 
df[['Administration']] = m.fit_transform(df[['Administration']])


# In[41]:


df.head()


# In[38]:





# In[10]:


MarketingNewArray = df['Marketing_Spend_New']
#for i in range (0,MarketingNewArray.length()):
 #   print(i)
MarketingNewArray
for key,value in df.iteritems():   #print using for loop
    print(key[0],value)
    


# In[44]:


from sklearn.preprocessing import FunctionTransformer
ft = FunctionTransformer(np.log1p) 
df2['Transport'] = ft.fit_transform(df2[['Transport']])


# In[43]:


df2.head()


# In[ ]:




