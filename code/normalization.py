#將MFECfeature做正規化，把值調整在-1~1之間
import os
import numpy as np
import io
import scipy.io.wavfile as wav
import math
import speechpy
import random

train = np.load('trainframe.npy')
test = np.load('testframe.npy')


train_data=[]
test_data=[]
reg=[]
mixframe=[]
count=0

for data in train :
	for frame in data :
		reg=[]
		min = np.amin(frame)
		max = np.amax(frame)
		for nor in frame :
			normalization = 2*((nor-min)/(max-min))-1
			normalization = normalization.astype(np.float32)
			reg.append(normalization)
		mixframe.append(reg)
		count=count+1
		if count == 40 :
			train_data.append(mixframe)
			mixframe=[]
			count=0
reg=[]
mixframe=[]
count=0			
for data in test :
	for frame in data :
		reg=[]
		min = np.amin(frame)
		max = np.amax(frame)
		for nor in frame :
			normalization = 2*((nor-min)/(max-min))-1
			normalization = normalization.astype(np.float32)
			reg.append(normalization)
		mixframe.append(reg)
		count=count+1
		if count == 40 :
			test_data.append(mixframe)
			mixframe=[]
			count=0
print("正規化訓練集數量:",len(train_data))
print(len(train))
print("正規化測試集數量:",len(test_data))
print(len(test))

np.save('traindata', train_data)
np.save('testdata', test_data)








