# coding: iso-8859-1
# Lesson - Machine Learning - HW1
# python this.py test.csv model.npy output.csv

import numpy as np;
import sys;
import csv;
import math

# initial data matrix
rawdata = []

def data_preprocessing(rawdata):
	for i in range(len(rawdata[0])):
		for j in range(18):
			for z in range(9):
				# RF and Wind_data can be 0.
				if j != 10 :
					# Deal wieh all the no-data point.
					if rawdata[j][i][z] <= 0:
						if z == 0:
							rawdata[j][i][z] = np.sum(rawdata[j][i]) / 9.;
						elif z == 1:
							rawdata[j][i][z] = rawdata[j][i][z-1];
						else:
							rawdata[j][i][z] = ( rawdata[j][i][z-2] + rawdata[j][i][z-1]) / 2;

				# Deal with the weird value in pm2.5.
				if j == 9:
					if rawdata[j][i][z] >= 200:
						if z == 0:
							rawdata[j][i][z] = np.sum(rawdata[j][i]) / 9.;
						elif z == 1:
							rawdata[j][i][z] = rawdata[j][i][z-1];
						else:
							rawdata[j][i][z] = ( rawdata[j][i][z-2] + rawdata[j][i][z-1]) / 2;

				# Wind direction: north nig, south small, east and west normal
				if j == 14 or j == 15:
					rawdata[j][i][z] = abs(rawdata[j][i][z] - 180) / 18;
	return rawdata
def make_datamatrix(rawdata, features, numbers_x, orders):
	x = []
	for i in range(len(rawdata[0])):	#260
		x.append([])
		for f in range(len(features)):
			for j in range(numbers_x):
				x[i].append(rawdata[features[f]-1][i][j+9-numbers_x])

	x = np.array(x);
	if orders == 2:
		x = np.concatenate((x,x**2), axis = 1);
	# add bias
	x = np.concatenate((np.ones((x.shape[0],1)),x), axis = 1);

	return x
def cal_y_estimated(x, weight):

	if len(weight) != len(x[0]):
		print("Wrong length between length and numbers of feature!")
		return 1;

	x = np.array(x)
	weight = np.array(weight)
	y_estimated = np.dot(x, weight)
	return y_estimated


weight = np.load("Hw1_linear_6f_1ord_5hr_10_100000_1st.npy")

# Read data
for i in range(18):
	rawdata.append([]);

inputcsv = open(sys.argv[1], 'r')
reader = csv.reader(inputcsv, delimiter = ',');
counter = 0;
for row in reader:
	rawdata[counter%18].append([]);
	for i in range(2,11):
		if row[i] == 'NR':
			rawdata[counter%18][counter//18].append(0);
		else:
			rawdata[counter%18][counter//18].append(float(row[i]));	# index_row-1 : modify the position
	counter += 1;
inputcsv.close()

# Begin
rawdata = np.array(rawdata)
rawdata = data_preprocessing(rawdata)

#features = [5,6,7,8,9,10,12,13,15,18]
#features = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18];
#features = [10];
#features = [1,3,4,5,6,7,8,9,10,13,14,15,18]
#features = [1,4,6,7,9,10,13,14,18]
features = [6,7,9,10,13,18] # 6 features

x = make_datamatrix(rawdata, features, 5, 1)
y_estimated = cal_y_estimated(x, weight)

result = csv.writer(open(sys.argv[2], 'w+'), delimiter = ',', lineterminator = '\n');
result.writerow(['id', 'value'])
for i in range(len(y_estimated)):
	result.writerow(['id_%d' %i, y_estimated[i]] )