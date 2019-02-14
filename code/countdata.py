import os
import numpy as np
import io
import scipy.io.wavfile as wav
import math
import wave
import pylab as pl
import speechpy
import tensorflow as tf
import h5py
import random
import operator
import csv
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from matplotlib.colors import LogNorm
from scipy import stats


traindata = np.load("traindata.npy")
countdata=[]
'''
for b in traindata :
	countdata.append(b)

for a in range(15):
	print(countdata.count(a))
print(len(countdata))
print(len(traindata))
'''
print(traindata.shape)













