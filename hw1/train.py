# Lesson - Machine Learning - HW1
# python this.py train.csv output(.npy) # hw1_linear_1f_2ord_9hr_learning rate_iteratio_1st

import numpy as np;
import sys;
import csv;
import math
import matplotlib.pyplot as plt

# initial data matrix
rawdata = []
for i in range(18):
	rawdata.append([]);

# read the raw data
reader = csv.reader(open(sys.argv[1], "r"), delimiter = ',');
index_row = 0;
for row in reader:
	if index_row != 0:		# Skip the first line.
		for i in range(3,27): 
			if row[i] == 'NR':
				rawdata[(index_row-1)%18].append(0);
			else:
				rawdata[(index_row-1)%18].append(float(row[i]));	# index_row-1 : modify the position
	index_row += 1;
rawdata = np.array(rawdata);

def check_rawdata(rawdata, feature, criteria):
	for i in range(len(rawdata[0])):
		if rawdata[feature-1][i] > criteria:
			print('%d/%d : too big!!' %(i/480+1, i%480/24+1));
# Except for RainFall, we have to deal with the 0 data point because 0 points are mess data.
def data_preprocessing(rawdata):
	for i in range(len(rawdata[0])):
		for j in range(18):
			# RF can be 0.
			if j != 10 :
				# Deal wieh all the no-data point.
				if rawdata[j][i] <= 0:
					rawdata[j][i] = (rawdata[j][i-2] + rawdata[j][i-1])/2; # x[t] = (x[t-2] + x[t-1]) / 2

			# Deal with the weird value in pm2.5.
			if j == 9:
				if rawdata[j][i] >= 200:
					 rawdata[j][i] = (rawdata[9][i-2] + rawdata[9][i-1])/2;

			# Wind direction: north nig, south small, east and west normal
			if j == 14 or j == 15:
				rawdata[j][i] = abs(rawdata[j][i] - 180) / 18;

	return rawdata
def make_datamatrix(rawmatrix, features = [10], numbers_x = 9, orders=1 ):
	datamatrix = []
	times_permonth = 24*20 - numbers_x;
	for i in range(12):
		# 20*24-9 = 471
		for j in range(times_permonth):
			datamatrix.append([]);
			for f in range(len(features)):
				for h in range(numbers_x):
					datamatrix[i*times_permonth + j].append(rawmatrix[features[f]-1][480*i+j+h])
	x = np.array(datamatrix);

	if orders == 2:
		x = np.concatenate((x,x**2), axis = 1);
	# add bias
	x = np.concatenate((np.ones((x.shape[0],1)),x), axis = 1);

	return x
def linear_regression(rawdata, x, numbers_x, orders , learning_rate, iteration):

	error_history = [];
	# Target
	y = [];
	for i in range(12):
		for j in range(480-numbers_x):
			y.append(rawdata[9][480*i+j+numbers_x])

	weight = np.zeros((len(x[0]))) ; # [w_bias, w1...]
	sumsquare_gradient = np.zeros(len(weight))
	counter = 0;

	for i in range(iteration):

		y_estimated = np.dot(x,weight)

		Loss = y - y_estimated	

		squere_error = np.sum(Loss**2) / len(y_estimated)
		
		error = math.sqrt(squere_error)
		
		gradient = np.dot(x.transpose(), Loss)

		sumsquare_gradient += gradient**2

		adagrad = np.sqrt(sumsquare_gradient)

		weight += learning_rate * gradient / adagrad
		
		print(error)
		#print(weight)
		error_history.append(error);

	np.save(sys.argv[2], weight)
def linear_regression_lambda(rawdata, x, numbers_x, orders , learning_rate, iteration, lambda1):
	error_history = [];
	# Target
	y = [];
	for i in range(12):
		for j in range(480-numbers_x):
			y.append(rawdata[9][480*i+j+numbers_x])

	weight = np.zeros((len(x[0]))) ; # [w_bias, w1...]
	sumsquare_gradient = np.zeros(len(weight))
	counter = 0;

	for i in range(iteration):

		y_estimated = np.dot(x,weight)

		Loss = y - y_estimated	

		squere_error = np.sum(Loss**2) / len(y_estimated) + lambda1 * np.sum(weight**2) ###
		
		error = math.sqrt(squere_error)
		
		gradient = np.dot(x.transpose(), Loss) + lambda1 * weight ###

		sumsquare_gradient += gradient**2

		adagrad = np.sqrt(sumsquare_gradient)

		weight += learning_rate * gradient / adagrad
		
		print(error)
		#print(weight)
		error_history.append(error);

	np.save(sys.argv[2], weight)

# Begin
rawdata = data_preprocessing(rawdata);

#features = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]; # All features
#features = [10];
#features = [5,6,7,8,9,10,12,13,15,18]
#features = [1,3,4,5,6,7,8,9,10,13,14,15,18] # 13 features
#features = [1,4,6,7,9,10,13,14,18] # 9 features (NHMC、THC)
features = [6,7,9,10,13,18] # 6 features

datamatrix = make_datamatrix(rawdata, features, 5, 1);
datamatrix = np.array(datamatrix);

linear_regression(rawdata, datamatrix, 5, 1, 10, 100000);
#linear_regression_lambda(rawdata, datamatrix, 9, 1, 10, 10000, 1)
