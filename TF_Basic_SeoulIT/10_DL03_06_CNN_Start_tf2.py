#!/usr/bin/env python
# coding: utf-8

# ### TF2.0 신경망 만들기
# * CNN 신경망 이해

# In[1]:


get_ipython().system('pip install -q tensorflow-gpu==2.0.0-rc1')


# In[ ]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models


# In[2]:


print(tf.__version__)


# In[5]:


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images.shape, test_images.shape


# ### 합성곰 층 만들기
# * 3D 텐서 : (이미지 높이, 이미지 너비, 컬러채널)
# * MNIST 데이터 셋은 흑백이미지 : (28,28,1)

# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# In[7]:


### 모델의 구조 출력
model.summary()


# * Total params 는 Param #을 전부 더해준것.

# ### 마지막 Dense 층 추가(FC)
# * Dense 층은 벡터(1D)를 입력으로 받는다. 현재 입력은 3D이므로 이를 1D로 펼치기 위해 Flatten()를 사용
# 

# In[ ]:


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# ### 모델의 구조 확인

# In[9]:


model.summary()


# ### 모델의 컴파일과 훈련

# In[11]:


get_ipython().run_cell_magic('time', '', "\nmodel.compile(optimizer='adam',\n              loss='sparse_categorical_crossentropy',\n              metrics=['accuracy'])\n\nmodel.fit(train_images, train_labels, epochs=5)")


# ### 모델 평가

# In[ ]:


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


# In[ ]:


print(test_acc)


# In[ ]:





# ### REF
# * CNN : https://www.tensorflow.org/tutorials/images/cnn

# In[ ]:




