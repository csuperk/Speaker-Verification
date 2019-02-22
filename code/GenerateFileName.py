#程式作用:產生檔名
import numpy as np

#編號T可用檔案150，編號D可用檔案100
human_list1 = ['T2_', 'T3_', 'T5_', 'T13_', 'T23_']
TNumber_list1 = [1000001, 1002251, 1000301, 1002551, 1003001] #檔案編號與human_list1對應 1000001 --> T2_
#human_list1 = ['T2_', 'T3_']
#TNumber_list1 = [1000001, 1002251] #檔案編號與human_list1對應 1000001 --> T2_
#testNumber_list1 =  [1000150, 1002400, 1000450, 1002700, 1003150] #反過來取測驗使用的數量

human_list2 = ['D2_', 'D7_', 'D10_', 'D16_', 'D18_', 'D20_', 'D30_', 'D32_', 'D33_', 'D34_']
DNumber_list2 = [1003198, 1000101, 1001599, 1000401, 1000001, 1002199, 1001101, 1003298, 1001299, 1001399] #檔案編號與human_list2對應 1003198 --> D2_
#testNumber_list2 =  [1003297, 1000200, 1001698, 1000500, 1000100, 1002298, 1001200, 1003397, 1001398, 1001498] #反過來取測驗使用的數量


traindata_list=[]
testdata_list=[]
train_filename_list = []
test_filename_list = []

savesequence_list = []

T = 150 #編號T可用檔案150
D = 100 #編號D可用檔案100
testnumber = 20 #testingdata數量

#產生編號T的檔名數量
count = 0
for number in TNumber_list1:
		while count < T:
			savesequence_list.append(number)
			number = number+1
			count = count+1
		reg = savesequence_list[0:testnumber]
		savesequence_list[0:testnumber]=()
		train_filename_list.extend(savesequence_list)
		test_filename_list.extend(reg)
		savesequence_list.clear()
		count=0
		Tlength = len(train_filename_list)
#print(len(train_filename_list))
#print(len(test_filename_list))
#print("trainT:",Tlength)	

#產生編號D的檔名數量
count = 0
for number in DNumber_list2:
		while count < D:
			savesequence_list.append(number)
			number = number+1
			count = count+1
		reg = savesequence_list[0:testnumber]
		savesequence_list[0:testnumber]=()
		train_filename_list.extend(savesequence_list)
		test_filename_list.extend(reg)
		savesequence_list.clear()
		count=0
		Dlength = len(train_filename_list) - Tlength
#print(len(train_filename_list))
#print(len(test_filename_list))
#print("trainD:",Dlength)


#traindata檔名組合
st = 0
countT = T - testnumber
countD = D - testnumber
end =  countT

for name in human_list1 : 
	for number in train_filename_list[st:end] :
		savesequence_list.append(name)
		savesequence_list.append(str(number))
		savesequence_list.append('.wav')
		reg = ''.join(savesequence_list)
		traindata_list.append(reg)
		savesequence_list.clear()
	st = st + countT
	end = end + countT

end = st + countD
for name in human_list2 : 
	for number in train_filename_list[st:end] :
		savesequence_list.append(name)
		savesequence_list.append(str(number))
		savesequence_list.append('.wav')
		reg = ''.join(savesequence_list)
		traindata_list.append(reg)
		savesequence_list.clear()
	st = st + countD
	end = end + countD
#print(len(traindata_list))

#testdata檔名組合

human_list1.extend(human_list2) #不做human_list2可以標記起來

st=0
end = testnumber

for name in human_list1:
	for number in test_filename_list[st:end]:
		savesequence_list.append(name)
		savesequence_list.append(str(number))
		savesequence_list.append('.wav')
		reg = ''.join(savesequence_list)
		testdata_list.append(reg)
		savesequence_list.clear()
	st = st+testnumber
	end = end+testnumber
print('traindata數量:',len(traindata_list))
print('testdata數量:',len(testdata_list))

np.save('trainfilename_list', traindata_list)
np.save('testfilename_list', testdata_list)



