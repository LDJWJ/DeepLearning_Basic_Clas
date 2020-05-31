#!/usr/bin/env python
# coding: utf-8

# ### MNIST 분류 모델 만들기 - 신경망

# In[1]:


from keras.datasets import mnist
from keras.utils import np_utils


# In[34]:


import numpy
import sys
import tensorflow as tf


# In[35]:


seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)


# ### 데이터 다운로드

# In[36]:


# 처음 다운일 경우, 데이터 다운로드 시간이 걸릴 수 있음. 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[37]:


import matplotlib.pyplot as plt


# In[38]:


figure,axes = plt.subplots(nrows=3, ncols=5)
figure.set_size_inches(18,12)

plt.gray()
print("label={}".format(y_train[0:15]))

col = 0
for row in range(0,3):
    col = row * 5
    axes[row][0].matshow(X_train[col])
    axes[row][1].matshow(X_train[col+1])
    axes[row][2].matshow(X_train[col+2])
    axes[row][3].matshow(X_train[col+3])
    axes[row][4].matshow(X_train[col+4])


# ### X_train의 데이터 정보를 하나 보기

# In[39]:


print(X_train.shape)  # 60000 만개, 28행, 28열
X_train[0].shape


# ### 신경망에 맞추어 주기 위해 데이터 전처리
#  * 학습 데이터 
#  * 테스트 데이터

# In[40]:


X_train = X_train.reshape(X_train.shape[0],784)   # 60000, 28, 28 -> 60000, 784로 변경
# 데이터 값의 범위 0~255 -> 0~1 
X_train.astype('float64')  
X_train = X_train/255

# 이렇게도 가능
# X_train = X_train.reshape(X_train.shape[0],784).astype('float64') / 255


# In[41]:


import numpy as np


# In[42]:


print(X_train.shape)               # 데이터 크기
np.min(X_train), np.max(X_train)   # 값의 범위


# In[43]:


# 테스트 데이터 전처리
X_test = X_test.reshape(X_test.shape[0],784)
X_test.astype('float64')
X_test = X_test/255


# In[44]:


print(X_test.shape)               # 데이터 크기
np.min(X_test), np.max(X_test)   # 값의 범위


# ## 출력데이터 검증을 위해 10진수의 값을 One-Hot Encoding을 수행

# In[45]:


# OneHotEncoding - 10진수의 값을 0, 1의 값을 갖는 벡터로 표현
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# ### 변환 전과 후

# In[14]:


y_train[0:4]


# In[15]:


Y_train[0:4]


# ### 딥러닝 만들어 보기

# In[16]:


from keras.models import Sequential
from keras.layers import Dense


# In[17]:


m = Sequential()


# In[18]:


m.add(Dense(512,input_dim=784, activation='relu'))
m.add(Dense(128, activation='relu') )
m.add(Dense(10,activation='softmax'))#softmax


# ### 오차함수 :categorical_crossentropy, 최적화 함수 : adam

# In[19]:


m.compile(loss="categorical_crossentropy", 
         optimizer='adam',
         metrics=['accuracy'])


# In[20]:


### 배치 사이즈 200, epochs 30회 실행,
history = m.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                epochs=30,
                batch_size=200,
                verbose=1)


# In[24]:


print("Test Accuracy : %.4f" %(m.evaluate(X_test, Y_test)[1]))


# In[25]:


pred = m.predict(X_test)


# In[26]:


print( pred.shape )
print( pred[1] )


# In[ ]:




