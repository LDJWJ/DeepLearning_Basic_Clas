#!/usr/bin/env python
# coding: utf-8

# ### 딥러닝 모델 구현해 보기
#  * 첫번째 데이터 셋 : 자전거 공유 업체 시간대별 데이터
#  * **두번째 데이터 셋 : 타이타닉 데이터 셋**

# In[56]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import tensorflow as tf


# In[57]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[58]:


print(keras.__version__)


# In[59]:


train = pd.read_csv("./titanic/train.csv")
test = pd.read_csv("./titanic/test.csv")
print(train.shape, test.shape)


# In[60]:


train.info()


# In[61]:


test.info()


# In[62]:


input_col = ['Pclass', 'SibSp', 'Parch']
labeled_col = ['Survived']


# In[63]:


X = train[ input_col ]
y = train[ labeled_col ]
X_val = test[ input_col ]


# In[64]:


seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)


# In[65]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                random_state=0)


# In[66]:


print(X_train.shape, X_test.shape)
print()
print(y_train.shape, y_test.shape)


# ## 딥러닝 구조 

# In[67]:


from keras.models import Sequential
from keras.layers import Dense


# In[71]:


model = Sequential()
model.add(Dense(30, input_dim=3, activation='relu'))
model.add(Dense(15, activation='relu') )
model.add(Dense(1, activation='sigmoid'))


# ### 딥러닝 설정 및 학습

# In[72]:


model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=10)


# ### 모델 평가

# In[73]:


model.evaluate(X_test, y_test)


# In[74]:


print("\n Accuracy : %.4f" % (model.evaluate(X_test, y_test)[1]))


# In[98]:


pred = model.predict(X_val)


# In[99]:


sub = pd.read_csv("./titanic/gender_submission.csv")
sub.columns


# In[107]:


pred[:, 0] > 0.5


# In[113]:


sub['Survived'] = pred[:, 0] > 0.5


# In[114]:


sub.to_csv("titanic_submit0528.csv", index=False)


# In[ ]:




