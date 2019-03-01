#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd


# In[38]:


DataSet = pd.read_csv('OnlineNewsPopularity/OnlineNewsPopularity.csv')
DataSet.head()


# In[39]:


size = DataSet.shape
last_column_index = size[1]
X = DataSet.iloc[:,2:last_column_index-1].values    #Extract all rows and all Columns execpt first two columns
Y = DataSet.iloc[:, [last_column_index-1]].values     #Extract Dependent Variable
print(X.shape)
print(Y.shape)


# In[40]:


from sklearn.model_selection  import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 0)


# In[41]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train , Y_train)
regressor


# In[42]:


y_predict = regressor.predict(X_test)
print(y_predict)


# In[43]:


print(Y_test)

