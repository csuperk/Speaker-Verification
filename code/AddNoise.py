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


for reg in train_filename :
	(rate,sig) = wav.read(os.path.join(str("D:/python/new/data"),reg))
	noise.append(sig)
	train_wavedata.append(noise)
	noise=[]
	print("切了幾個traindata:",len(train_wavedata))

for reg in test_filename :
	(rate,sig) = wav.read(os.path.join(str("D:/python/new/data"),reg))
	noise.append(sig)
	test_wavedata.append(noise)
	noise=[]
	print("切了幾個testdata:",len(test_wavedata))
	
np.save('trainwavedata', train_wavedata)
np.save('testwavedata', test_wavedata)