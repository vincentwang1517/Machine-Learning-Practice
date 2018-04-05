# Hw2 generative - gaussian
# python3 this.py 

import numpy as np
import csv
import pandas as pd
import sys
import math

# Train_X
datamatrix = [];
file = open(sys.argv[1], 'r');
datamatrix = pd.read_csv(file);
file.close()
datamatrix = np.array(datamatrix)
datamatrix = datamatrix.astype('float64')

# Train_Y
file = open(sys.argv[2], "r");
reader = pd.read_csv(file, header = None);
y = np.array(reader);
y = y.transpose()[0];
file.close()

# Test_X
file = open(sys.argv[3], 'r');
test_x = pd.read_csv(file);
file.close()
test_x = np.array(test_x)
test_x = test_x.astype('float64')


def feature_scaling1(datamatrix, mean, sigma):
	x = np.copy(datamatrix)
	x[:,0] = (datamatrix[:,0] - mean[0]) / sigma[0] ;
	x[:,10] = (datamatrix[:,10] - mean[1]) / sigma[1] ;
	x[:,78] = (datamatrix[:,78] - mean[2]) / sigma[2] ;
	x[:,79] = (datamatrix[:,79] - mean[3]) / sigma[3] ;
	x[:,80] = (datamatrix[:,80] - mean[4]) / sigma[4] ;
	return x
def sigmoid(z):
	# np.clip(z, -100, 100)
	sigmoid =  1/(1+np.exp(-z));
	return np.clip(sigmoid, 0.0000001, 0.9999999) ;
def generative_gaussian(x, y):
	nums_features = len(x[0]);
	train_data_size = len(x)

	# calculate mean 
	cnt1 = 0;
	cnt2 = 0;

	mu1 = np.zeros((nums_features, ))
	mu2 = np.zeros((nums_features, ))
	for i in range(train_data_size):
	    if y[i] == 1:
	        mu1 += x[i]
	        cnt1 += 1
	    else:
	        mu2 += x[i]
	        cnt2 += 1
	mu1 /= cnt1
	mu2 /= cnt2


	# calculate sigma
	sigma1 = np.zeros((nums_features,nums_features))
	sigma2 = np.zeros((nums_features,nums_features))
	for i in range(train_data_size):
	    if y[i] == 1:
	        sigma1 += np.dot(np.transpose([x[i] - mu1]), [(x[i] - mu1)])
	    else:
	        sigma2 += np.dot(np.transpose([x[i] - mu2]), [(x[i] - mu2)])
	sigma1 /= cnt1
	sigma2 /= cnt2
	shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
	# shared_sigma = (sigma1 + sigma2) / train_data_size
	return mu1, mu2, shared_sigma, cnt1, cnt2
def predict_gaussian(X_test, mu1, mu2, sigma, N1, N2):
	sigma_inverse = np.linalg.pinv(sigma)
	w = np.dot( (mu1-mu2), sigma_inverse)
	x = X_test.T
	b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
	a = np.dot(w, x) + b
	y = sigmoid(a)
	print(y)
	y_ = np.around(y)
	return y;

age = np.copy(datamatrix[:,0])
fnlwgt = np.copy(datamatrix[:,10])
capital_gain = np.copy(datamatrix[:,78])
capital_loss = np.copy(datamatrix[:,79])
hr_per_week = np.copy(datamatrix[:,80])
temp = np.array([age, fnlwgt, capital_gain, capital_loss, hr_per_week])
mean = np.mean(temp, axis = 1)
sigma = np.std(temp, axis = 1)
x_train = feature_scaling1(datamatrix, mean, sigma)

# age = x_train[:,0]
# fnlwgt = x_train[:,10]
# capital_gain = x_train[:,78]
# capital_loss = x_train[:,79]
# hr_per_week = x_train[:,80]
# temp = np.array([age, fnlwgt, capital_gain, capital_loss, hr_per_week])
# mean = np.mean(temp, axis = 1)
# sigma = np.std(temp, axis = 1)
# print(mean)
# print(sigma)

mu1, mu2, shared_sigma, N1, N2 = generative_gaussian(x_train,y);
x_test = feature_scaling1(test_x, mean, sigma)
y_estimated = predict_gaussian(x_test, mu1, mu2, shared_sigma, N1, N2)

y_predict = [];
for i in range(len(y_estimated)):
	if y_estimated[i] > 0.5:
		y_predict.append(1)
	else:
		y_predict.append(0)

result = csv.writer(open(sys.argv[4], 'w+'), delimiter = ',', lineterminator = '\n');
result.writerow(['id', 'label'])
for i in range(len(y_predict)):
	result.writerow([(i+1), y_predict[i]] )