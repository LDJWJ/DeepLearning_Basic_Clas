#!/usr/bin/env python
# coding: utf-8

# ### TF2.0 신경망 만들기
# * 손글씨 데이터 셋을 이용한 신경망 만들기

# ### 사전 작업
# * tf2.0 설치 후, 재시작(설치 적용을 위해)
# * 런타임 - 런타임 유형 변경 - GPU 설정

# In[ ]:


import tensorflow as tf


# In[2]:


print(tf.__version__)


# In[ ]:


get_ipython().system('pip install -q tensorflow-gpu==2.0.0-rc1')
import tensorflow as tf


# In[4]:


print(tf.__version__)


# ### MNIST 데이터 셋을 이용한 신경망 구성

# In[5]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# In[8]:


print("학습용 데이터 : x: {}, y:{}".format(x_train.shape, y_train.shape) )
print("테스트 데이터 : x: {}, y:{}".format(x_test.shape, y_test.shape) )


# ### 신경망 구성
# * tf.keras.Sequential를 이용한 모델 구성
# *

# In[ ]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),    # 2D -> 1D
  tf.keras.layers.Dense(128, activation='relu'),    # 활성화 함수 - relu
  tf.keras.layers.Dropout(0.2),                     # Dropout적용
  tf.keras.layers.Dense(10, activation='softmax')   # 활성화 함수 - softmax
])


# ### 구성
# * sparse_categorical_crossentropy : 다중 분류 손실함수 (정수값 기준)
# * categorical_crossentropy : 다중 분류 손실함수 (one-hot-encoding 기준 (예측과 실제 결과값))

# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ### 모델 훈련 및 평가

# In[13]:


model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)


# ### REF
# * TF2.0 Tutorial : https://www.tensorflow.org/tutorials/quickstart/beginner
# * tf.keras.Sequential : https://www.tensorflow.org/api_docs/python/tf/keras/Sequential

# In[ ]:




