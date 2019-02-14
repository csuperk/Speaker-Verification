import os
import numpy as np
import io
import scipy.io.wavfile as wav
import math

train_filename = np.load('trainfilename_list.npy')
test_filename = np.load('testfilename_list.npy')

print('trafilename:',len(train_filename))
print('testfilename',len(test_filename))

train_segments = []
test_segments = []

train_wavedata = []
test_wavedata = []

savesequence_list=[]
energy=[]
epd=[]
noise=[]

#Pdata1=[]
#Pdata2=[]

framesize=320
overlap=160
accumulate=0

for reg in train_filename :
	(rate,sig) = wav.read(os.path.join(str("D:/python/new/data"),reg))
	dotimes = math.floor((len(sig)-framesize)/overlap)+1
	savesequence_list=[]
	for do in range(dotimes) :
		savesequence_list.append(sig[do*overlap:((do+1)*overlap+overlap)])
	energy=[]
	#取絕對值總和
	for a in savesequence_list :
		for b in range(len(a)) :
			absolute = abs(a[b])
			accumulate= absolute+accumulate
		energy.append(accumulate)
		accumulate=0
	#取最大值除10當threshold
	threshold = max(energy)/10
	onset=0
	offset=0
	epd=[]
	noise=[]
	for c in range(len(energy)):
		if energy[c] > threshold and onset == 0:
			onset = c
		if energy[c] < threshold and onset!=0 and offset == 0:
			offset = c
			epd.append(onset)
			epd.append(offset)
			onset=0
			offset=0
	cut=[]
	segments=int(len(epd)/2)
	for b in range(segments):
		getout = epd[2*b+1]-epd[2*b]
		if getout <= 5 :
			noise.append(epd[2*b+1])
			noise.append(epd[2*b])
	
	for d in noise:
		epd.remove(d)
	
	segments=int(len(epd)/2)
	train_segments.append(segments)
	for y in range(segments):
		cut.append(sig[(epd[y*2])*overlap:(epd[(y+1)*2-1]+1)*overlap+overlap])
	train_wavedata.append(cut)
	print("切了幾個traindata:",len(train_segments))
	print("這個traindata切了幾段:",segments)
'''
#每個人有幾個可用連續聲音
if len(train_segments) <= 650 :
	count=0
	person = int(len(train_segments)/130)
	for a in range(person) :
		for b in train_segments[a*130:(a+1)*130]:
			count = count + b
		Pdata1.append(count)
		count = 0
elif len(train_segments) > 650 :
	count=0
	person = int(((len(train_segments)-650)/80))
	for a in range(person) :
		for b in train_segments[a*80+650:(a+1)*80+650]:
			count = count + b
		Pdata1.append(count)
		count=0
	for c in range(5) :
		for d in train_segments[c*130:(c+1)*130]:
			count = count + d
		Pdata1.append(count)
		count = 0
#print(train_segments)
print(Pdata1)	
'''
noise=[]
for reg in test_filename :
	(rate,sig) = wav.read(os.path.join(str("D:/python/new/data"),reg))
	noise.append(sig)
	test_wavedata.append(noise)
	noise=[]
	print("切了幾個testdata:",len(test_wavedata))
'''
#每個人有幾個可用的連續聲音
count=0
person = int(len(test_segments)/20)
for a in range(person) :
		for b in test_segments[a*20:(a+1)*20]:
			count = count + b
		Pdata2.append(count)
		count = 0
print("For training personal data:",Pdata1)
print("For testing personal data:",Pdata2)
'''


'''
for reg in train_filename :
	(rate,sig) = wav.read(os.path.join(str("/Users/vlsi/Desktop/python/new/data"),reg))
	savesequence_list = sig[1*rate:]
	train_wavedata.append(savesequence_list)
print(len(train_wavedata))		

for reg in test_filename:
	(rate,sig) = wav.read(os.path.join(str("/Users/vlsi/Desktop/python/new/data"),reg))
	savesequence_list = sig[1*rate:]
	test_wavedata.append(savesequence_list)
print(len(test_wavedata))

##############################不切一秒###########################

for reg in train_filename :
	(rate,sig) = wav.read(os.path.join(str("/Users/vlsi/Desktop/python/new/data"),reg))
	
	train_wavedata.append(sig)
print(len(train_wavedata))		

for reg in test_filename:
	(rate,sig) = wav.read(os.path.join(str("/Users/vlsi/Desktop/python/new/data"),reg))
	test_wavedata.append(sig)
print(len(test_wavedata))

##############################不切一秒###########################
'''
np.save('trainwavedata', train_wavedata)
np.save('testwavedata', test_wavedata)
#np.save('trainsegments', train_segments)
#np.save('testsegments', test_segments)
#np.save('trainPdata',Pdata1)
#np.save('testPdata',Pdata2)