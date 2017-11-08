
# coding: utf-8

# # MILK PRODUCTION

# In[1]:


import pandas as pd


# In[2]:


xls = pd.read_csv('monthly-milk-production-pounds-p.csv', skipfooter = 2,converters= {'Month': pd.to_datetime})


# In[3]:


xls


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


get_ipython().magic('matplotlib notebook')


# In[6]:


import numpy as np


# In[7]:


import statsmodels.api as sm


# # DATA TRANSFORMATION

# In[8]:


Y=xls['Monthly milk production: pounds per cow. Jan 62 ? Dec 75']


# In[9]:


for m in range(1, 13):
    xls['M'+str(m)]=xls['Month'].apply(lambda x: 1 if x.month == m else 0)
    
xls['Year']=xls['Month'].apply(lambda x: x.year)

xls


# In[10]:


X = xls.as_matrix()[:, 2:15]
X = X.astype(np.int32)
X


# In[11]:


X=sm.add_constant(X)


# # ESTIMATION

# In[12]:


model = sm.OLS(Y,X)


# In[13]:


results = model.fit()


# In[14]:


results.params


# In[15]:


xls['fit'] = np.dot(X, results.params.values)


# In[18]:


Data= xls.set_index(['Month'])[['Monthly milk production: pounds per cow. Jan 62 ? Dec 75', 'fit']]
Data.resample('1M').mean().plot()


# # PREDICTION 2000 - 2010

# In[28]:


newXls = pd.read_csv('estimation 2000-2010.csv', converters= {'Month': pd.to_datetime})



# In[30]:


for m in range(1, 13):
    newXls['M'+str(m)]=newXls['Month'].apply(lambda x: 1 if x.month == m else 0)
    
newXls['Year']=newXls['Month'].apply(lambda x: x.year)

newX = newXls.as_matrix()[:, 1:14]
newX = newX.astype(np.int32)
newX = sm.add_constant(newX)
newX


# In[34]:


newXls['Estimated production'] = np.dot(newX, results.params.values)
Data= newXls.set_index(['Month'])[['Estimated production']]
Data.resample('1M').mean().plot()

