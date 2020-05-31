#!/usr/bin/env python
# coding: utf-8

# ### Tensorflow 시작하기
# * 라이브러리 불러오기
# * tf.constant 알아보기
# * tf.add() 실습
# * sess() 실습
# * 행렬 곱

# ### 01. 라이브러리 임포트
# * 설치 : pip install tensorflow

# In[1]:


import tensorflow as tf


# In[2]:


print(tf.__version__)


# ### 02. 값 저장하기
# * Tensor 자료형 : 
# * shape(), string
# * tf.constant() : Creates a constant tensor.

# In[3]:


hello = tf.constant("Hello, Tensorflow")
print(hello)
print(type(hello))


# ### 03. 텐서플로워 프로그램 두 가지 과정
# * 그래프의 생성
# * 그래프의 실행

# ### 그래프의 생성
#  * 그래프는 생성 단계에서 연산과정을 그래프로 표시

# ### 덧셈

# In[4]:


num1 = tf.constant(10)
num2 = tf.constant(20)
num3 = tf.add(num1,num2)
print(type(num1), type(num2), type(num3))
print(num3)

# 2행 3열의 텐서를 생성.
tensor = tf.constant(-1.0, shape=[2, 3])


# ### 그래프의 실행
#  * 실제 연산 부분 C++로 구현한 코어 라이브러리 실행
#  * 모델 구성과 실행을 분리시켜 프로그램을 작성함.

# In[5]:


## 세션 연결 후, run을 통해 실행
sess = tf.Session()
print(sess.run(hello))


# In[6]:


print(sess.run([num3]))


# In[7]:


print(sess.run(tensor))


# In[8]:


sess.close()


# ### 04. 텐서 플로워의 자료형
# * 플레이스 홀더 : 나중에 값을 입력받기 위한 사용되는 매개변수(parameter)
# * shape(?,3)은 두번째 차원 요소가 3이고, 앞의 차원은 나중에 지정.     

# In[9]:


# None은 크기가 정해져 있지 않음을 의미
tensorX = tf.placeholder(tf.float32, [None, 2])
print(type(tensorX))
print(tensorX)


# In[10]:


# X에 넣을 데이터 초기화
x_data = [[1,1], [2,2]]


# In[11]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(tensorX, feed_dict={tensorX:x_data}))
sess.close()


# ### 05 행렬 연산
# * Y = X (2행2열) * W(2행*2열) + b(2행)

# #### tf.random_normal : 정규 분포로부터 랜덤한 값 생성

# In[12]:


# None은 크기가 정해져 있지 않음을 의미
tensorX = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.random_normal([2,2]))
b = tf.Variable(tf.random_normal([2,1]))
print(W)
print(b)


# In[13]:


expr = tf.matmul(tensorX, W) + b
expr


# ### 연산 결과 및 결과 출력

# In[15]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("x_data ==")
print(x_data)
print("W ==")
print(sess.run(W))
print("B ==")
print(sess.run(b))

print("expr ==")
print(sess.run(expr, feed_dict={tensorX:x_data}))
sess.close()


# ### REF
# * [What does the 'b' character do in front of a string literal?](https://stackoverflow.com/questions/6269765/what-does-the-b-character-do-in-front-of-a-string-literal)
# * 골빈해커의 3분 딥러닝
