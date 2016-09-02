
# coding: utf-8

# In[3]:
# Demo Example: https://www.opendatascience.com/blog/preprocessing-data-a-python-workflow-part-1/?utm_source=Open+Data+Science+Newsletter&utm_campaign=feeab557df-Newsletter_Vol_598_25_2016&utm_medium=email&utm_term=0_2ea92bb125-feeab557df-245816209
get_ipython().system('pip3 install --upgrade bokeh')


# In[4]:

import pandas as pd
import numpy as np


# In[12]:

df1 = pd.DataFrame([[2,4,6],[10,20,30]], columns=["Price", "Age", "Value"], index=["First","Second"])


# In[13]:

df1.shape


# In[14]:

df1


# In[15]:

# Pass Dictionaries into Pandas DataFrame
df2 = pd.DataFrame([{"Name":"John", "Surname": "Johns"}, {"Name": "Jack"}])


# In[16]:

df2.shape


# In[17]:

df2


# In[18]:

type(df2)


# In[19]:

dir(df1)


# In[20]:

df1.mean()


# In[21]:

df1.mean().mean()


# In[22]:

df1.Price


# In[23]:

df1.Price.mean()


# In[27]:

get_ipython().system('pip3 install --upgrade pandas_datareader')
from pandas_datareader import data, wb
df_wb = wb.download(
                    # Specify indicator to retrieve
                    indicator='SP.POP.TOTL',
                    country=['all'],
                    # Start Year
                    start='2008',
                    # End Year
                    end=2016
                )


# In[28]:

df_wb.shape


# In[29]:

df_wb.head()


# In[ ]:




# In[36]:

get_ipython().system('pip3 install --upgrade requests')
# !pip3 install --upgrade BeautifulSoup
from bs4 import BeautifulSoup
import requests
print(1)


# In[37]:

r = requests.get("https://en.wikipedia.org/wiki/Eagle")
print(r.content)


# In[38]:

soup = BeautifulSoup(r.content)
print(soup.prettify)


# In[ ]:



