#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np


# In[12]:


book = pd.read_csv('C:/Users/prate/Downloads/Assignment/Recommendation System/book.csv')


# In[13]:


book.head()


# In[14]:


book.drop(book.filter(regex="Unname"),axis=1, inplace=True)


# In[15]:


book.head()


# In[16]:


book.rename({"Unnamed: 0":"a","User.ID":"userid","Book.Title":"Book_Title","Book.Rating":"Book_Rating"}, axis="columns", inplace=True)


# In[17]:


book.head()


# In[24]:


book.drop_duplicates('userid',inplace=True,keep="first")


# In[25]:


book.duplicated('userid')


# In[27]:


book_new = book.pivot(index='userid',
                                 columns='Book_Title',
                                 values='Book_Rating').reset_index(drop=True)
book_new


# In[30]:


len(book.userid.unique())


# In[31]:


#Impute those NaNs with 0 values
book_new.fillna(0, inplace=True)


# In[32]:


book_new


# In[33]:


len(book.Book_Title.unique())


# In[34]:


#Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[35]:


book_new1 = 1 - pairwise_distances( book_new.values,metric='cosine')


# In[38]:


book_new1


# In[39]:


#Store the results in a dataframe
book_new1_df = pd.DataFrame(book_new1)


# In[41]:


#Set the index and column names to user ids 
book_new1_df.index = book.userid.unique()
book_new1_df.columns = book.userid.unique()


# In[42]:


book_new1_df.iloc[0:5, 0:5]


# In[43]:


np.fill_diagonal(book_new1, 0)
book_new1_df.iloc[0:5, 0:5]


# In[44]:


#Most Similar Users
book_new1_df.idxmax(axis=1)[0:5]


# In[45]:


book[(book['userid']==276737) | (book['userid']==276729)]


# In[ ]:




