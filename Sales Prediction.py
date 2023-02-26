#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv("Desktop/Advertising.csv",index_col=0)
data


# In[2]:


data.info()


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,6))
sns.lineplot(data=data,x=data['TV'],y=data['Sales'])


# In[4]:


plt.figure(figsize=(15,6))
sns.lineplot(data=data,x=data['Radio'],y=data['Sales'])


# In[7]:


plt.figure(figsize=(15,6))
sns.lineplot(data=data,x=data['Newspaper'],y=data['Sales'])


# In[9]:


data.describe()


# In[23]:


x = data.drop(['TV','Radio','Newspaper'],axis=1)
print(x)
y = data['Sales']


# In[27]:


plt.scatter(x,y)
plt.title("Sales Rate")
plt.xlabel("Products")
plt.ylabel("Sales")
plt.show()


# In[18]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)


# In[19]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)


# In[29]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()


# In[30]:


rfr.fit(x_train,y_train)


# In[31]:


y_pred = rfr.predict(x_test)


# In[32]:


accuracy = rfr.score(x_test,y_test)
print("Accuracy:",accuracy*100)


# In[ ]:




