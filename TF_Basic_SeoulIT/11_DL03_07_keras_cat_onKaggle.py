#!/usr/bin/env python
# coding: utf-8

# ## 개 고양이 분류

# ### 캐글 대회
#  * https://www.kaggle.com/c/dogs-vs-cats
# 
# 
# ### 커널 참조 
#  * https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### 라이브러리 임포트

# In[ ]:


import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import seaborn as sns
print(os.listdir("../input/dogs-vs-cats"))


# ### 상수 정의 

# In[ ]:


FAST_RUN = True  
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3


# ### 학습용 데이터 준비

# In[ ]:


import zipfile

with zipfile.ZipFile('../input/dogs-vs-cats/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/train_img')


# In[ ]:


filenames = os.listdir("/kaggle/working/train_img/train")

categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


# In[ ]:


df.head()


# ### Sample 데이터 확인
#  * load_img ( ) : PIL 포맷으로 이미지를 불러온다

# In[ ]:


sample = random.choice(filenames)
image = load_img("/kaggle/working/train_img/train/"+sample)
plt.imshow(image)


# ## 모델 생성

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()


# ### EarlyStopping 함수 
#  * 어떤 성능을 모니터링 할 것인가?
#  * 언제 멈출 것인가?
#  
# ### ReduceLROnPlateau 함수 
#  * 평가 지표가 개선되지 않을 때, 학습률을 동적으로 조정해준다.

# ## Callbacks

# In[ ]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ### EarlyStopping 함수
#  * 10에폭까지는 기본적으로 진행

# In[ ]:


earlystop = EarlyStopping(patience=10)


# ### 학습률 조정
#  * 2번째 스텝까지는 참기.(patience)
#  * 참고할 지표 : monitor
#  * factor : 기존의 학습률에 곱하기
#     * 새로운 학습률 구하기 new_lr = lr * factor

# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[ ]:


callbacks = [earlystop, learning_rate_reduction]


# In[ ]:


df.head()


# ### category 컬럼을 cat, dog로 변경
#   * why? 우리는 image geneartor를 사용한다. 여기에서는 범주형은 문자열 컬럼이어야 한다.
#   * 추후 이 컬럼은 one-hot encoding으로 변환될 것이다.

# In[ ]:


df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 


# ### 데이터 나누기
#  * test는 20%, train은 80%
#  * 데이터를 나누고 index를 초기화 한다.(reset_index)

# In[ ]:


train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


# In[ ]:


plt.figure(figsize=(12,10))

plt.subplot(2,1,1)
sns.countplot(train_df['category']).set_title('train')

plt.subplot(2,1,2)
sns.countplot(validate_df['category']).set_title('validation')


# ### 데이터 크기 확인

# In[ ]:


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15


# ### 학습용 이미지 생성기(Training Generator)
#  * ImageDataGenerator 클래스 : 실시간 데이터 증강을 통해 텐서 이미지 데이터를 생성한다.

# In[ ]:


train_datagen = ImageDataGenerator(
    rotation_range=15,  # 정수 무작위 회전의 각도 범위
    rescale=1./255,     # 크기 재조절. 기본은 None이고 None인 경우 크기 재조절이 안됨.
    shear_range=0.1,    # 부동 소수점. 층밀리기 강도(도 단위의 반시계 방향 층밀리기 각도)
    zoom_range=0.2,     # 부동 소수점.[하한, 상한] 무작위 줌의 범위 [1-zoom_range, 1+zoom_range]
    horizontal_flip=True,  # 인풋을 무작위로 가로로 뒤집기
    width_shift_range=0.1, # 1D 형태의 유사배열 혹은 정수
    height_shift_range=0.1 # 1D 형태의 유사배열 혹은 정수
)


# ### ImageDataGenerator를 사용하여 이미지 로드 및 이미지 증식이 가능
#  * flow
#  * flow_from_directory
#  * flow_from_dataframe

# ### flow_from_dataframe
#   * ref : https://keras.io/ko/preprocessing/image/
#   * 데이터 프레임과 디렉터리 위치를 전달받아 증강/정규화된 데이터의 배치를 생성

# In[ ]:


train_generator = train_datagen.flow_from_dataframe(
    train_df,    # 데이터 프레임 
    "/kaggle/working/train_img/train/",   # 이미지 위치 
    x_col='filename',  # 
    y_col='category',  # target 데이터를 갖는 데이터 프레임의 컬럼
    target_size=IMAGE_SIZE,    # 정수의 튜플 (높이, 넓이) 디폴트 값(256, 256) 모든 이미지의 크기 조정.
    class_mode='categorical',  #  "categorical", "binary", "sparse", "input", "other" 중 하나
    batch_size=batch_size      # 데이터 배치의 크기. 배치사이즈
)


# ## Validation Generator

# In[ ]:


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "/kaggle/working/train_img/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# ### 어떻게 동작하는지 확인해 보기

# In[ ]:


example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "/kaggle/working/train_img/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)


# In[ ]:


plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# ## 모델 학습시키기
#  * 케라스에서 모델을 학습시킬 때 주로 fit()함수를 사용하지만, 제너레이터로 생성된 배치 학습시에는 fit_generator()함수를 사용.
#  * ImageDataGenerator라는 제너레이터로 이미지를 담고 있는 배치로 학습

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nepochs=3 if FAST_RUN else 50   # FAST_RUN은 기본적으로 False로 되어 있음.\nhistory = model.fit_generator(\n    train_generator, \n    epochs=epochs,\n    validation_data=validation_generator,\n    validation_steps=total_validate//batch_size,\n    steps_per_epoch=total_train//batch_size,\n    callbacks=callbacks\n)')


# ### 3 epoch : 12분 50초
# ### 50 epoch : 

# ## 모델 저장

# In[ ]:


model.save_weights("model.h5")


# ## 학습 훈련 데이터 시각화

# In[ ]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

# 추가(축 부분)
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# ## 테스트 데이터 준비

# In[ ]:


import zipfile

with zipfile.ZipFile('../input/dogs-vs-cats/test1.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/test_img')


# In[ ]:


test_filenames = os.listdir("/kaggle/working/test_img/test1")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]


# ### 테스트 데이터 제너레이터(생성기)

# In[ ]:


test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    "/kaggle/working/test_img/test1",  # 이미지 위치
    x_col = 'filename',
    y_col = None,
    class_mode = None,
    target_size =  IMAGE_SIZE,
    batch_size = batch_size,
    shuffle = False
)


# ### 예측 수행하기

# In[ ]:


step_num = np.ceil(nb_samples/batch_size)  # 반올림값 반환

predict = model.predict_generator(test_generator, steps = step_num) 


# In[ ]:


test_df['category'] = np.argmax(predict, axis=-1)

label_map = dict((v,k) for k,v in train_generator.class_indices.items())
print(label_map)
test_df['category'] = test_df['category'].replace(label_map)


# In[ ]:


test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })


# ### 이미지 예측 결과

# In[ ]:


sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("/kaggle/working/test_img/test1/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()


# ## 제출

# In[ ]:


submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)


# In[ ]:




