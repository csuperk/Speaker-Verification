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



#這坨東西就是分類結果，還存取不出來所以乖乖複製貼上複製錯了就慘了
eval_results = [5,5,0,5,5,1,2,3,5,0,1,0,5,5,5,5,2,2,5,4,5,2,3,5,5,5,5,5,5,5,5,5,5,4,0,5,5
,5,5,3,4,5,5,5,0,5,5,5,5,4,5,5,5,0,5,0,2,5,5,5,5,5,5,5,5,5,2,5,4,5,4,5,5,5
,2,0,5,5,5,0,5,0,2,5,5,5,5,5,5,3,3,5,5,2,5,5,5,4,4,5,5,5,5,5,2,5,5,5,0,1,5
,5,5,0,5,5,0,5,2,5,5,5,5,5,0,1,5,5,5,5,3,5,5,5,5,4,5,5,5,4,3,5,5,2,5,2,5,5,1,5,5,5,5,5,5,5,5,5,5,5,5,5,5,0,5
,0,0,5,5,3,5,5,1,5,5,4,5,5,3,5,5,5,3,5,5,0,5,5,5,5,5,3,1,5,5,5,2,5,5,3,5,5
,5,5,3,5,5,5,5,3,5,5,1,5,5,5,5,5,2,5,5,5,4,5,5,5,5,5,5,5,5,5,3,5,5,5,3,3,5
,5,5,4,1,0,5,1,2,3,5,5,0,5,5,5,5,5,0,5,5,4,5,5,4,5,2,0,2,4,4,2,3,5,5,5,2,2,5,5,0,5,5,5,5,5,3,1,5,5,5,5,1,5,5
,5,4,3,5,2,5,5,5,2,2,5,5,5,5,5,0,5,0,2,5,2,5,5,5,5,5,5,3,1,5,2,5,5,5,5,3,5
,5,5,0,5,5,1,5,0,5,0,5,5,5,5,5,5,4,4,5,5,5,5,5,5,5,0,3,5,5,5,5,5,0,3,5,5,1
,5,5,5,5,5,5,5,2,5,5,0,5,5,5,5,5,5,5,5,3,5,3,5,5,5,4,5,5,0,2,5,5,0,5,3,2,5,5,5,5,5,5,5,5,5,5,5,3,2,5,2,3,5,5
,5,5,5,5,5,2,5,5,3,5,5,5,5,3,1,5,3,5,5,5,5,5,5,5,5,4,5,0,5,1,5,5,5,4,5,5,0
,5,5,2,4,5,5,4,5,5,5,5,0,3,5,3,2,5,5,5,5,1,5,5,2,5,2,5,5,5,5,2,2,3,0,0,5,5
,2,5,5,5,5,5,5,5,5,5,5,5,3,5,1,5,5,1,5,1,5,1,5,2,3,5,2,1,5,5,2,5,5,5,5,5,5,3,5,5,5,5,3,5,5,5,5,0,5,1,5,5,5,2
,5,0,5,5,5,5,1,5,5,5,5,4,5,5,5,3,5,5,1,3,5,5,5,5,5,5,5,3,5,5,3,3,0,0,5,5,0
,4,5,5,5,5,5,4,5,2,5,3,5,2,0,5,4,5,5,5,3,0,5,5,5,5,1,5,5,4,5,1,5,1,5,3,2,5
,5,5,0,5,5,5,5,5,0,5,5,5,5,5,2,5,5,5,2,2,5,3,5,5,5,5,5,5,4,0,1,5,5,5,5,5,5,5,5,5,5,5,3,5,5,5,0,5,5,5,4,4,0,5
,5,5,2,5,5,2,5,5,0,5,5,5,5,0,5,0,5,5,5,4,4,0,5,5,4,3,5,0,5,5,5,1,3,5,5,5,5
,5,5,0,5,5,5,5,2,5,5,2,2,5,5,5,5,5,5,5,0,5,5,0,5,5,5,5,5,5,5,5,4,5,5,1,2,3
,5,5,1,5,5,5,5,2,5,5,4,4,2,4,5,4,3,5,5,5,2,5,1,2,1,5,5,5,5,1,5,5,5,2,5,4,5,0,5,2,2,5,5,5,2,5,5,5,3,5,5,5,5,5
,5,5,1,5,5,5,0,5,5,5,5,2,5,5,1,5,5,3,4,5,5,5,0,2,5,5,5,5,5,5,5,5,0,0,5,5,5
,3,5,3,5,5,5,5,5,5,0,4,5,5,5,5,5,4,5,5,5,5,5,0,0,2,5,4,3,5,5,2,5,1,5,4,0,5
,5,4,5,5,5,5,5,5,5,0,5,5,5,5,5,5,5,0,4,5,5,5,0,5,5,3,5,5,5,5,5,3,5,1,5,5,5,5,3,5,3,5,5,0,5,5,1,5,5,5,3,3,5,1
,0,5,5,5,5,5,0,5,2,5,5,5,5,3,5,5,3,5,5,5,5,5,5,0,0,3,5,0,1,3,5,5,0,3,4,5,1
,5,5,5,2,5,5,5,5,1,1,5,5,3,5,5,4,5,1,3,4,5,5,5,5,5,3,5,5,5,5,5,5,4,5,5,5,5
,5,5,5,5,5,1,1,5,5,1,5,5,5,1,1,5,5,5,5,2,0,0,5,0,5,3,5,3,5,0,0,5,4,1,0,5,5,1,5,5,5,5,5,5,1,5,5,5,4,5,5,5,5,5
,5,5,5,5,5,5,4,5,5,5,5,0]


#train_dataindex = np.load('traindataindex.npy')
#traindata = np.load("traindata.npy")
test_dataindex = np.load('testdataindex.npy')
testdata = np.load("testdata.npy")
correctlabel = np.load('testlabel.npy')
correctlabel = correctlabel.tolist()
#eval_results = np.load("evalresults.npy")#等CNN可以把結果存出來再來用

whichone = np.load("testframeofeverysong.npy")
reg=[]
lookerror=[]
Calculation=[]
gopie=[]
title=["who" ,"wavfile" ,"Label" ,"ClassificationResult"]
count=0
which=0
for c in range(len(correctlabel)):
	if operator.eq(correctlabel[c],eval_results[c]) == False	:
		for a in whichone :
			count = count + a
			which = which + 1
			if  count >= test_dataindex[c] :
				who = math.floor(which/20)
				#if who >= 5 :
				#	who = 5
				x = which % 20
				reg.append(who)
				reg.append(x)
				count = 0
				which = 0
				break
		gopie.extend(reg)
		Calculation.append(gopie)
		gopie=[]
		#reg.append(test_dataindex[c])
		reg.append(correctlabel[c])
		reg.append(eval_results[c])
		lookerror.append(reg)
		reg=[]

with open('result.csv', 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
	spamwriter.writerow(title)
	for a in lookerror :
		spamwriter.writerow(a)

#######計算 precisoin recall f1-score zscore accuracy###########
precision = metrics.precision_score(y_true =correctlabel , y_pred=eval_results, average=None)
recall = metrics.recall_score(y_true =correctlabel , y_pred=eval_results, average=None)
f1score = metrics.f1_score(y_true =correctlabel , y_pred=eval_results, average=None)
precisionzscore = stats.zscore(precision)
recallzscore = stats.zscore(recall)
f1scorezscore = stats.zscore(f1score)

Calculation.sort()
error = len(Calculation)
for a in range(len(Calculation)):
	for b in range(len(Calculation)):
		c = Calculation.count([a,b])
		if c > 0 :
			error = error-c+1
acc = (300-error)/300
			

#print(lookerror)
print("幾個分類錯誤:",len(lookerror))
print(len(eval_results))
print(len(correctlabel))
print(metrics.confusion_matrix(correctlabel,eval_results))
print(metrics.classification_report(correctlabel,eval_results))
print("precisio nzscore:",precisionzscore)
print("recall zscore:",recallzscore)
print("f1score zscore:",f1scorezscore)
print(acc)


##########A simple categorical heatmap
# sphinx_gallery_thumbnail_number = 2
###  "T2(Male)", "T3(Female)", "T5(Male)","T13(Female)", "T23(Female)", "D2(Female)", "D7(Male)","D10(Female)","D16(Male)","D18(Male)","D20(Female)","D30(Male)","D32(Female)","D33(Male)","D34(Male)"
vegetables = ["T2(Male)", "T3(Female)", "T5(Male)", "T13(Female)", "T23(Female)",  "Others"]
farmers = ["T2(Male)", "T3(Female)", "T5(Male)", "T13(Female)", "T23(Female)",  "Others"]

harvest = np.array(metrics.confusion_matrix(correctlabel,eval_results))


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Confusion_Matrix_Result(5People6Classification)")
fig.tight_layout()
plt.show()



