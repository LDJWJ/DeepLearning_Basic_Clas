#!/usr/bin/env python
# coding: utf-8

# ### 딥러닝 모델 구현해 보기
#  * 첫번째 데이터 셋 : 자전거 공유 업체 시간대별 데이터
#  * 두번째 데이터 셋 : 타이타닉 데이터 셋

# In[14]:


import tensorflow as tf
import keras


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


# In[17]:


print("tf version : {}".format(tf.__version__))
print("keras version : {}".format(keras.__version__))
print("numpy version : {}".format(np.__version__))
print("matplotlib version : {}".format(matplotlib.__version__))
print("pandas version : {}".format(pd.__version__))


# ### 데이터 셋 불러오기

# In[18]:


## train 데이터 셋 , test 데이터 셋
## train 은 학습을 위한 입력 데이터 셋
## test 은 예측을 위한 새로운 데이터 셋(평가)
## parse_dates : datetime 컬럼을 시간형으로 불러올 수 있음
train = pd.read_csv("./bike/bike_mod_tr.csv", parse_dates=['datetime'])
test = pd.read_csv("./bike/bike_mod_test.csv", parse_dates=['datetime'])


# ### 데이터 탐색

# In[19]:


train.columns


# In[20]:


test.columns


# In[21]:


print(train.info())
print()
print(test.info())


# ### 모델을 위한 데이터 선택 
#  * X : hour, temp : 시간, 온도
#  * y : count - 자전거 시간대별 렌탈 대수 

# In[22]:


input_col = [ 'hour', 'temp']
labeled_col = ['count']


# In[51]:


X = train[ input_col ]
y = train[ labeled_col ]
X_val = test[input_col]


# In[52]:


from sklearn.model_selection import train_test_split


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                random_state=0)


# In[54]:


print(X_train.shape)
print(X_test.shape)


# In[55]:


### 난수 발생 패턴 결정 0
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)


# ## 딥러닝 구조 결정

# * 케라스 라이브러리 중에서 Sequential 함수는 딥러닝의 구조를 한층 한층 쉽게 쌓아올릴 수 있다.
# * Sequential() 함수 선언 후, 신경망의 층을 쌓기 위해 model.add() 함수를 사용한다
# * input_dim 입력층 노드의 수
# * activation - 활성화 함수 선언 (relu, sigmoid)
# * Dense() 함수를 이용하여 각 층에 세부 내용을 설정해 준다.

# In[56]:


from keras.models import Sequential
from keras.layers import Dense


# In[57]:


model = Sequential()
model.add(Dense(30, input_dim=2, activation='relu'))
model.add(Dense(15, activation='relu') )
model.add(Dense(15, activation='relu') )
model.add(Dense(1))


# ### 미니배치의 이해
#  * 이미지를 하나씩 학습시키는 것보다 여러 개를 한꺼번에 학습시키는 쪽이 효과가 좋다.
#  * 많은 메모리와 높은 컴퓨터 성능이 필요하므로 일반적으로 데이터를 적당한 크기로 잘라서 학습시킨다.
#   * **미니배치**라고 한다.

# ### 딥러닝 실행

# In[58]:


model.compile(loss = 'mean_squared_error', optimizer='rmsprop')
model.fit(X_train, y_train, epochs=20, batch_size=10)


# In[59]:


### 평가 확인
model.evaluate(X_test, y_test)


# In[60]:


pred = model.predict(X_val)


# In[61]:


sub = pd.read_csv("./bike/sampleSubmission.csv")
sub['count'] = pred

sub.loc[sub['count']<0, 'count'] = 0


# In[62]:


sub.to_csv("nn_sub_0528.csv", index=False)


# ## 점수 : 1.04514

# In[ ]:




