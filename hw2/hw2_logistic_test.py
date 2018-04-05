# coding: iso-8859-1
# Lesson - Machine Learning - HW1
# python this.py test.csv output.csv model.npy

import numpy as np;
import pandas as pd;
import sys;
import csv;
import math

x_all = [];
file = open(sys.argv[1], 'r');
x_all = pd.read_csv(file);
file.close()
x_all = np.array(x_all)
x_all = np.concatenate((np.ones((x_all.shape[0], 1)), x_all), axis = 1);

# initial data matrix
file = open(sys.argv[3], "r");
reader = pd.read_csv(file)
file.close()
x = np.array(reader);
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis = 1);

weight = np.load("./log_4th.npy")
print_weight = np.round(weight, decimals=2)
# np.set_printoptions(1e6)
print(print_weight)
print(len(print_weight))

def feature_scaling(x_all, datamatrix):

	age = x_all[:,1];
	fnlwgt = x_all[:,11];
	capital_gain = x_all[:,79];
	capital_loss = x_all[:,80];
	hr_per_week = x_all[:,81];

	temp = np.array([age, fnlwgt, capital_gain, capital_loss, hr_per_week]);
	mean = np.mean(temp, axis = 1);
	sigma = np.std(temp, axis = 1);

	datamatrix[:,1] = (datamatrix[:,1] - mean[0]) / sigma[0] ;
	datamatrix[:,11] = (datamatrix[:,11] - mean[1]) / sigma[1] ;
	datamatrix[:,79] = (datamatrix[:,79] - mean[2]) / sigma[2] ;
	datamatrix[:,80] = (datamatrix[:,80] - mean[3]) / sigma[3] ;
	datamatrix[:,81] = (datamatrix[:,81] - mean[4]) / sigma[4] ;

	return datamatrix
def sigmoid(z):
	np.clip(z, -100, 100)
	sigmoid =  1/(1+np.exp(-z));
	return np.clip(sigmoid, 0.0000001, 0.9999999) ;
def cal_y_estimated(x, weight):

	if len(weight) != len(x[0]):
		print("Wrong length between length and numbers of feature!")
		return 1;

	x = np.array(x)
	weight = np.array(weight)

	y_estimated = sigmoid(np.dot(x, weight))
	return y_estimated

# Begin
x_all = feature_scaling(x_all,x);

capital_gain = x_all[:,79].reshape((len(x_all),1));
x_all = np.concatenate((x_all, capital_gain**2), axis = 1)
age = x_all[:,1].reshape((len(x_all),1));
x_all = np.concatenate((x_all, age**2), axis = 1)
fnlwgt = x_all[:,11].reshape((len(x_all),1));
x_all = np.concatenate((x_all, fnlwgt**2), axis = 1)
hr_per_week = x_all[:,81].reshape((len(x_all),1));
x_all = np.concatenate((x_all, hr_per_week**2), axis = 1)	

x_all = np.concatenate((x_all, capital_gain**3), axis = 1)
x_all = np.concatenate((x_all, age**3), axis = 1)
x_all = np.concatenate((x_all, fnlwgt**3), axis = 1)
x_all = np.concatenate((x_all, hr_per_week**3), axis = 1)	

x_all = np.concatenate((x_all, np.log(np.absolute(capital_gain))), axis = 1)
x_all = np.concatenate((x_all, np.log(np.absolute(age))), axis = 1)
x_all = np.concatenate((x_all, np.log(np.absolute(fnlwgt))), axis = 1)
x_all = np.concatenate((x_all, np.log(np.absolute(hr_per_week))), axis = 1)

bad_features = np.array([8,55,117,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]) # 98,107,115
x_train = np.delete(x_all, bad_features, axis = 1);
print(x_train.shape)

y_estimated = cal_y_estimated(x_train, weight)

# Decide the label
y_predict = [];
for i in range(len(y_estimated)):
	if y_estimated[i] > 0.5:
		y_predict.append(1)
	else:
		y_predict.append(0)
print(len(y_predict))
# y_predict = np.around(y_estimated)

result = csv.writer(open(sys.argv[4], 'w+'), delimiter = ',', lineterminator = '\n');
result.writerow(['id', 'label'])
for i in range(len(y_predict)):
	result.writerow([(i+1), y_predict[i]] )
