# Hw2 Gradient Descent
#
# 怎麼用csv讀檔?

import numpy as np
import csv
import pandas as pd
import sys
import random
from math import log, floor

datamatrix = [];
file = open(sys.argv[1], 'r');
datamatrix = pd.read_csv(file);
file.close()
datamatrix = np.array(datamatrix)
datamatrix = np.concatenate((np.ones((datamatrix.shape[0], 1)), datamatrix), axis = 1);

file = open(sys.argv[2], "r");
reader = pd.read_csv(file, header = None);
y = np.array(reader);
y = y.transpose()[0];
file.close()

def sigmoid(z):
	sigmoid =  1/(1+np.exp(-z));
	return np.clip(sigmoid, 0.0000001, 0.9999999) ;
def cross_entropy(x, y):
	A = np.dot(y, np.log(x));
	B = np.dot(1-y, np.log(1-x));
	return -(A+B) ;

def feature_scaling(datamatrix):
	age = datamatrix[:,1];
	fnlwgt = datamatrix[:,11];
	capital_gain = datamatrix[:,79];
	capital_loss = datamatrix[:,80];
	hr_per_week = datamatrix[:,81];

	temp = np.array([age, fnlwgt, capital_gain, capital_loss, hr_per_week]);
	mean = np.mean(temp, axis = 1);
	sigma = np.std(temp, axis = 1);

	datamatrix[:,1] = (datamatrix[:,1] - mean[0]) / sigma[0] ;
	datamatrix[:,11] = (datamatrix[:,11] - mean[1]) / sigma[1] ;
	datamatrix[:,79] = (datamatrix[:,79] - mean[2]) / sigma[2] ;
	datamatrix[:,80] = (datamatrix[:,80] - mean[3]) / sigma[3] ;
	datamatrix[:,81] = (datamatrix[:,81] - mean[4]) / sigma[4] ;

	return datamatrix

def linear_regression(x_train, y_train, learning_rate, batch_size, epoch):	# Target

	weight = np.zeros((len(x_train[0]))) ; # [w_bias, w1...]
	# random.seed(20);
	# for i in range(len(weight)):
	# 	weight[i] = random.random();
	# random.seed(100); #10, 20
	sumsquare_gradient = np.ones(len(weight))
	train_data_size = len(x_train)
	step_num = int((train_data_size / batch_size))

	for epoch in range(epoch):
		x_train, y_train = _shuffle(x_train, y_train)
		Loss = 0;
		for i in range(step_num):
			x = x_train[i*batch_size:(i+1)*batch_size]
			y = y_train[i*batch_size:(i+1)*batch_size]

			y_estimated = np.dot(x,weight);
			y_estimated = sigmoid(y_estimated)

			Loss += cross_entropy(y_estimated, y ) / len(y);

			gradient = np.dot(x.transpose(), y - y_estimated) / float(batch_size)

			sumsquare_gradient += gradient**2

			adagrad = np.sqrt(sumsquare_gradient)

			weight += learning_rate * gradient / adagrad
			
		print(Loss)

	np.save("log_4th.npy", weight)
	return weight
def linear_regression_lambda(x_train, y_train, learning_rate, batch_size, epoch, lambda1):
	# Target
	weight = np.zeros((len(x_train[0]))) ; # [w_bias, w1...]
	# random.seed(20);
	# for i in range(len(weight)):
	# 	weight[i] = random.random();
	# random.seed(10); #10, 20
	sumsquare_gradient = np.ones(len(weight))
	train_data_size = len(x_train)
	step_num = int((train_data_size / batch_size))

	for epoch in range(epoch):
		x_train, y_train = _shuffle(x_train, y_train)
		Loss = 0;
		for i in range(step_num):
			x = x_train[i*batch_size:(i+1)*batch_size]
			y = y_train[i*batch_size:(i+1)*batch_size]

			y_estimated = np.dot(x,weight);
			y_estimated = sigmoid(y_estimated)

			Loss += cross_entropy(y_estimated, y ) / len(y) + lambda1 * np.sum(weight**2);

			gradient = np.dot(x.transpose(), y - y_estimated) / batch_size + lambda1 * weight

			sumsquare_gradient += gradient**2

			adagrad = np.sqrt(sumsquare_gradient)

			weight += learning_rate * gradient / adagrad
			
		print(Loss)

	np.save("log.4th.npy", weight)
	return weight

def _shuffle(x, y):
	randomize = np.arange(len(x));
	np.random.shuffle(randomize);
	return x[randomize], y[randomize]
def split_valid_set(X_all, Y_all, percentage):
	all_data_size = len(X_all)
	valid_data_size = int((all_data_size * percentage))
	X_all, Y_all = _shuffle(X_all, Y_all)

	X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
	X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

	return X_train, Y_train, X_valid, Y_valid
def valid(w, X_valid, Y_valid):
	valid_data_size = len(X_valid)

	z = (np.dot(X_valid, np.transpose(w)))
	y = sigmoid(z)
	y_ = np.around(y)
	result = (Y_valid == y_)

	# print('Validation acc = %f' % (float(result.sum()) / valid_data_size))
	acc = float(result.sum()) / valid_data_size
	print(acc)
	return acc
def use_valid(x_all, y_all):
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

	# bad_features = np.array([8,55,117,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]) # 40,32,34,36,30,41,43,42,31,35,28
	# bad_features = np.array([7,8,45,47,55,69, 73,75,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,89,98,107,115]) 
	#11? 47? 57? 69?
	# bad_features = np.array([7,8,9,45,47,55,69, 73,75,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,89,98,107,115]) 
	# 7?
	bad_features = np.array([8,55,117,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]) 
	x_train = np.delete(x_all, bad_features, axis = 1)
	# x_train = x_all

	rate = 0.1;
	batch = 300;
	epoch = 150;
	np.random.seed(10);
	X_train, Y_train, X_valid, Y_valid = split_valid_set(x_train, y, 0.7);
	w = linear_regression_lambda(X_train, Y_train, rate, batch,epoch , 0.000001);
	# w = linear_regression(X_train, Y_train, 0.1, 300, 150)
	valid(w, X_valid, Y_valid);

	np.random.seed(200);
	X_train, Y_train, X_valid, Y_valid = split_valid_set(x_train, y, 0.7);
	w = linear_regression_lambda(X_train, Y_train, rate, batch, epoch, 0.000001);
	# w = linear_regression(X_train, Y_train, 0.1, 300, 150)
	valid(w, X_valid, Y_valid);

	np.random.seed(1000);
	X_train, Y_train, X_valid, Y_valid = split_valid_set(x_train, y, 0.7);
	w = linear_regression_lambda(X_train, Y_train, rate, batch, epoch, 0.000001);
	# w = linear_regression(X_train, Y_train, 0.1, 300, 150)
	valid(w, X_valid, Y_valid);

np.random.seed(20); # 3rd
x_all = feature_scaling(datamatrix);

# Check scaling
# age = x_all[:,1];
# fnlwgt = x_all[:,11];
# capital_gain = x_all[:,79];
# capital_loss = x_all[:,80];
# hr_per_week = x_all[:,81];
# temp = np.array([age, fnlwgt, capital_gain, capital_loss, hr_per_week]);
# mean = np.mean(temp, axis = 1);
# sigma = np.std(temp, axis = 1);
# print(mean)
# print(sigma)

# use_valid(x_all, y)

bad_features = np.array([8,55,117,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])

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

x_train = np.delete(x_all, bad_features, axis = 1);

w = linear_regression(x_train, y, 0.1, 300, 200)
# w = linear_regression_lambda(x_train, y, 0.1, 300, 150, 0.000001)
print(len(w))
