#程式用途:訊號前處理(端點檢測、傅立葉轉換、梅爾頻譜轉換)
import os
import numpy as np
import io
import scipy.io.wavfile as wav
import math
import speechpy

train_frame = []
train_label = []
test_frame = []
test_label = []
frameofeverysong = []
'''
label = 1 #標籤分類總共15個人所以有15個label
sof = 320 #Size Of Frame..音框大小
nof =    #Number Of Frame..幾個音框合併成inputsize
overlap = 160
'''
rate = 16000

train_wavedata = np.load('trainwavedata.npy')
test_wavedata = np.load('testwavedata.npy')

label = 1 #標籤分類總共15個人所以有15個label
totalframe=0
totallabel=0
labelnumberT=0
labelnumberD=0

#####################Training Data#####################
for sig in train_wavedata:
	mfecfeature = speechpy.feature.lmfe(sig, rate, frame_length=0.02, frame_stride=0.01, num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
	logenergy_feature_cube = speechpy.feature.extract_derivative_feature(mfecfeature)
	train_frame.append(logenergy_feature_cube)
	totalframe += len(mfecfeature)
	frameofeverysong.append(len(mfecfeature))#記錄每個檔案切出幾個音框
	'''
	for number in range(len(mfecfeature)):
		train_label.append(label)
	if labelnumberT < 130 and totallabel < 650 :
		labelnumberT += 1
		totallabel += 1			
		if labelnumberT >= 130 and totallabel < 650 :
			label += 1
			labelnumberT = 0
		if totallabel == 650 :
			label += 1			
	if labelnumberD < 80 and totallabel >= 650:
		labelnumberD += 1
		totallabel += 1
		if labelnumberD >= 80 and totallabel >= 650 :
			labelnumberD = 0
			label += 1
	'''
		
#print(frameofeverysong)		
print("train_frame:",len(train_frame))
print("totaltraining:",totalframe)
#print(len(train_label))

#####################Testing Data#####################
totalframe = 0
totallabel = 0
labelnumberT = 0
labelnumberD = 0
label = 1 #標籤分類總共15個人所以有15個label

for sig in test_wavedata :
	mfecfeature = speechpy.feature.lmfe(sig, rate, frame_length=0.02, frame_stride=0.01, num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
	logenergy_feature_cube = speechpy.feature.extract_derivative_feature(mfecfeature)
	test_frame.append(logenergy_feature_cube)
	totalframe += len(mfecfeature)
	frameofeverysong.append(len(mfecfeature))#記錄每個檔案切出幾個音框
	'''
	for number in range(len(mfecfeature)) :
		test_label.append(label)
	if labelnumberT < 20 :
		labelnumberT += 1
		totallabel += 1
		if labelnumberT >= 20 :
			label += 1
			labelnumberT = 0
	'''

print("test_frame:",len(test_frame))
print("totaltesting:",totalframe)
#print(len(test_label))


np.save('feature_mfec', train_frame)
#np.save('trainlabel', train_label)
np.save('feature_mfectest', test_frame)
#np.save('testlabel', test_label)