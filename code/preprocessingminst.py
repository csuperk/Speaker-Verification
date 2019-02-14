#程式用途:訊號前處理(傅立葉轉換、梅爾頻譜轉換)
import os
import numpy as np
import io
import scipy.io.wavfile as wav
import math
import speechpy
import random

train_frame = []
train_label = []
test_frame = []
test_label = []
frameofeverysong = []
reg=[]

shuffle_copy=[]
merge_shuffle=[]
shuffle_index=[]

reg_frame=[]
reg_label=[]
reg_index=[]
onesongfeature =[]

trainPdata=[]
testPdata=[]

'''
label = 1 #標籤分類總共15個人所以有15個label
sof = 320 #Size Of Frame..音框大小
nof =    #Number Of Frame..幾個音框合併成inputsize
overlap = 160
'''
rate = 16000

train_wavedata = np.load('trainwavedata.npy')
test_wavedata = np.load('testwavedata.npy')


label = 0 #標籤分類總共15個人所以有15個label
totalframe=0
totallabel=0
labelnumberT=0
labelnumberD=0
mixframe = 40
person= input("幾個人測試:") #多少人做測驗有幾個改幾個

#####################Training Data#####################
for one in train_wavedata:
	onesongfeature=[]
	for sig in one :
		mfecfeature = speechpy.feature.lmfe(sig, rate, frame_length=0.02, frame_stride=0.01, num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
		onesongfeature.extend(mfecfeature)
	do = math.floor(len(onesongfeature)/mixframe)
	frameofeverysong.append(do)#記錄每個檔案切出幾個TrainingData
	totalframe += do
	for times in range(do) :
		#train_frame.append(mfecfeature[(times-1)*mixframe:times*mixframe])
		x = (np.array(onesongfeature[times*mixframe:(times+1)*mixframe]).astype(np.float32))
		reg_frame.append(x)
	for number in range(do):
		reg_label.append(label)
	if int(person) <= 5 :
		if labelnumberT < 130 and totallabel < int(person)*130 :
			labelnumberT += 1
			totallabel += 1			
			if labelnumberT >= 130 and totallabel < int(person)*130 :
				label += 1
				labelnumberT = 0
		if totallabel >= int(person)*130 :
			label = int(person)
			totallabel += 1
	
	if int(person) >= 6  :
		if labelnumberT < 130 and totallabel < 650 :
			#print("OK的:",totallabel)
			labelnumberT += 1
			totallabel += 1	
			if labelnumberT >= 130 and totallabel < 650 :
				label += 1
				labelnumberT = 0
				#print(label)
		if totallabel == 650 :
			label += 1
			#print("雞雞硬硬的:",label)
			#print(sum(frameofeverysong))			
		if labelnumberD < 80 and totallabel >= 650 and totallabel < 650+(80*(int(person)-5)):
			labelnumberD += 1
			totallabel += 1
			if len(reg_label) == 651:
				labelnumberD = labelnumberD-1
				totallabel = totallabel-1
			#print("哪裡壞掉了:",label)
			if labelnumberD >= 80 and totallabel >= 650 and totallabel < 650+(80*(int(person)-5)) :
				labelnumberD = 0
				label += 1
		if totallabel >= 650+(80*(int(person)-5)) :
			if len(reg_label) == 650+(80*(int(person)-5)) :
				totallabel = totallabel-1
			label = int(person)
			totallabel += 1

for a in range(int(person)+1) : 
	Pdata = reg_label.count(a)
	trainPdata.append(Pdata)

reg_index=list(range(totalframe))
	
for shuffle in range(len(reg_frame)):
	shuffle_copy.extend(reg_index[shuffle:shuffle+1])
	shuffle_copy.extend(reg_frame[shuffle:shuffle+1])
	shuffle_copy.extend(reg_label[shuffle:shuffle+1])
	merge_shuffle.append(shuffle_copy)
	#print(k)
	shuffle_copy=[]

doshuffle = random.sample(merge_shuffle, len(merge_shuffle))			

for a in range(len(doshuffle)):
	for b in range(3):
		if b == 0:
			shuffle_index.append(doshuffle[a][b])
		elif b==1:
			train_frame.append(doshuffle[a][b])
		else:
			train_label.append(doshuffle[a][b])			


print(label)			
#print(len(frameofeverysong))		
print("train_frame:",len(train_frame))
print("totaltraining:",totalframe)
print("labelnumbers:",len(train_label))
print("everyperson's trainframe :",trainPdata)
#print(train_frame)
np.save("traindataindex",shuffle_index)
np.save("trainframeofeverysong",frameofeverysong)
for b in range(label+1) :
	print("label分類:",train_label.count(b))

#####################Testing Data#####################
totalframe = 0
totallabel = 0
labelnumberT = 0
labelnumberD = 0
label = 0 #標籤分類總共15個人所以有15個label
reg_frame=[]
reg_label=[]
reg_index=[]
shuffle_index=[]
merge_shuffle=[]
frameofeverysong=[]

for one in test_wavedata :
	onesongfeature=[]
	for sig in one :
		mfecfeature = speechpy.feature.lmfe(sig, rate, frame_length=0.02, frame_stride=0.01, num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
		onesongfeature.extend(mfecfeature)
	do = math.floor(len(onesongfeature)/mixframe)
	frameofeverysong.append(do)#記錄每個檔案切出幾個TrainingData
	totalframe += do
	for times in range(do) :
		x = (np.array(onesongfeature[times*mixframe:(times+1)*mixframe]).astype(np.float32))
		reg_frame.append(x)
		#test_frame.append(mfecfeature[(times-1)*mixframe:times*mixframe])
	
	for number in range(do) :
		reg_label.append(label)
	if labelnumberT < 20 and totallabel < int(person)*20:
		labelnumberT += 1
		totallabel += 1
		if labelnumberT >= 20 :
			label += 1
			labelnumberT = 0
	if totallabel >= int(person)*20 :
		label = int(person)
		totallabel += 1
			#print(sum(frameofeverysong))

for a in range(int(person)+1) : 
	Pdata = reg_label.count(a)
	testPdata.append(Pdata)			

reg_index=list(range(totalframe))
	
for shuffle in range(len(reg_frame)):
	shuffle_copy.extend(reg_index[shuffle:shuffle+1])
	shuffle_copy.extend(reg_frame[shuffle:shuffle+1])
	shuffle_copy.extend(reg_label[shuffle:shuffle+1])
	merge_shuffle.append(shuffle_copy)
	#print(k)
	shuffle_copy=[]

doshuffle = random.sample(merge_shuffle, len(merge_shuffle))			

for a in range(len(doshuffle)):
	for b in range(3):
		if b == 0:
			shuffle_index.append(doshuffle[a][b]) 
		elif b==1:
			test_frame.append(doshuffle[a][b])
		else:
			test_label.append(doshuffle[a][b])				
			

print(label)
#print(len(frameofeverysong))
print("test_frame:",len(test_frame))
print("totaltesting:",totalframe)
print("labelnumbers:",len(test_label))
print("everyperson's testframe :",testPdata)
#print(test_frame)
for b in range(label+1) :
	print("label分類:",test_label.count(b))

np.save('trainframe', train_frame)
np.save('trainlabel', train_label)
np.save('testframe', test_frame)
np.save('testlabel', test_label)
np.save("testdataindex",shuffle_index)
np.save("testframeofeverysong",frameofeverysong)