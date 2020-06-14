# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:01:15 2020

@author: seoul it
"""

#%%
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from PIL import Image

import os, glob
import numpy as np

# 현재 경로 확인이 가능.
print(os.getcwd())

#%%
root_dir = "./kfood/"

# 카테고리 정보를 담는 리스트를 선언
categories = ['Chicken', 'Dolsotbab', 'Jeyugbokk-eum',
              'Kimchi','Samgyeobsal','SoybeanPasteStew']

nb_classes = len(categories)

image_width = 64
image_height = 64

#%%  데이터 변수 
X = []   # 이미지 데이터
Y = []   # 레이블 데이터 

for idx, category in enumerate(categories):
    image_dir = root_dir + category 
    files = glob.glob(image_dir + "/" + "*.jpg" )  # 해당 경로의 파일을 넘겨받는다.
    print(image_dir + "/" + "*.jpg")
    print("해당 폴더 파일 개수 : ", len(files)) 
    print(files)
    for i, f in enumerate(files):
        ## 이미지를 로딩
        # 01 이미지 파일을 불러온다.
        # 02 RGB로 변환한다.
        # 03 이미지 크기를 resize
        # 04 해당 이미지를 숫자 배열 데이터 변경.
        # 05 변경한 데이터를 X의 리스트에 추가한다.
        # 06 해당 Y의 리스트에 idx(이미지가 속한 범주)를 추가 
        img = Image.open(f)   
        img = img.convert("RGB")
        img = img.resize( (image_width, image_height))
        data = np.asarray(img)
        X.append(data)
        Y.append(idx)
X = np.array(X)
Y = np.array(Y)
print(X.shape, Y.shape)

#%%        
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

xy = (X_train ,X_test, Y_train, Y_test)

# 데이터 파일로 저장.
np.save(root_dir + "kfood.npy", xy)
    
    



