# Hw3 CNN
# python3 this.py predict.csv

import numpy as np
from numpy import random
import csv
import pandas as pd
import sys
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def model_structure_2():	# 1-1-2
	model = Sequential();
	
	model.add(Conv2D( filters=128, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform', input_shape=(48,48,1)));
	model.add(LeakyReLU(alpha=0.1));	#
	model.add(BatchNormalization());
	model.add(MaxPooling2D(pool_size=(2,2)));
	model.add(Dropout(0.3));	

	model.add(Conv2D( filters=128, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(BatchNormalization());
	model.add(MaxPooling2D(pool_size=(2,2))); # 128 x 12 x 12
	model.add(Dropout(0.35));

	model.add(Conv2D( filters=256, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(Dropout(0.25));
	model.add(Conv2D( filters=256, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(BatchNormalization());
	model.add(MaxPooling2D(pool_size=(2,2))); # 64 x 6 x 6
	model.add(Dropout(0.15));

	model.add(Flatten());
	model.add(Dense(512,  kernel_initializer='random_uniform'));
	model.add(LeakyReLU(alpha=0.1));
	model.add(Dropout(0.34));
	model.add(Dense(256,  kernel_initializer='random_uniform'));
	model.add(LeakyReLU(alpha=0.1));
	model.add(BatchNormalization());
	model.add(Dropout(0.14));
	model.add(Dense(7, activation='softmax'));

	print(model.summary());
	return model;
def model_structure_4():	# 3-1-2
	model = Sequential();


	
	model.add(Conv2D( filters=64, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform', input_shape=(48,48,1)));
	model.add(LeakyReLU(alpha=0.15));	#
	model.add(Dropout(0.2));
	model.add(Conv2D( filters=64, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform', input_shape=(48,48,1)));
	model.add(LeakyReLU(alpha=0.15));
	model.add(Dropout(0.2));
	model.add(Conv2D( filters=64, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform', input_shape=(48,48,1)));
	model.add(LeakyReLU(alpha=0.15));
	model.add(BatchNormalization());
	model.add(MaxPooling2D(pool_size=(2,2)));
	model.add(Dropout(0.2));	

	model.add(Conv2D( filters=128, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(BatchNormalization());
	model.add(MaxPooling2D(pool_size=(2,2))); # 128 x 12 x 12
	model.add(Dropout(0.45));

	model.add(Conv2D( filters=256, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(Dropout(0.2));
	model.add(Conv2D( filters=256, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(BatchNormalization());
	model.add(MaxPooling2D(pool_size=(2,2))); # 64 x 6 x 6
	model.add(Dropout(0.23));

	model.add(Flatten());
	model.add(Dense(256,  kernel_initializer='random_uniform'));
	model.add(LeakyReLU(alpha=0.1));
	model.add(Dropout(0.34));
	model.add(Dense(256,  kernel_initializer='random_uniform'));
	model.add(LeakyReLU(alpha=0.1));
	model.add(BatchNormalization());
	model.add(Dropout(0.34));
	model.add(Dense(7, activation='softmax'));

	print(model.summary());
	return model;
def model_structure_5():	# 1-2-4
	model = Sequential();
	
	model.add(Conv2D( filters=64, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform', input_shape=(48,48,1)));
	model.add(LeakyReLU(alpha=0.1));	#
	model.add(BatchNormalization());
	model.add(MaxPooling2D(pool_size=(2,2)));
	model.add(Dropout(0.2));	

	model.add(Conv2D( filters=128, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(Dropout(0.25));
	model.add(Conv2D( filters=128, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(BatchNormalization());
	model.add(MaxPooling2D(pool_size=(2,2))); # 128 x 12 x 12
	model.add(Dropout(0.35));

	model.add(Conv2D( filters=256, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(Dropout(0.25));
	model.add(Conv2D( filters=256, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(Dropout(0.25));
	model.add(Conv2D( filters=256, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));
	model.add(Dropout(0.35));
	model.add(Conv2D( filters=256, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	model.add(LeakyReLU(alpha=0.1));	
	model.add(BatchNormalization());
	model.add(MaxPooling2D(pool_size=(2,2))); # 64 x 6 x 6
	model.add(Dropout(0.25));

	model.add(Flatten());
	model.add(Dense(256,  kernel_initializer='random_uniform'));
	model.add(LeakyReLU(alpha=0.1));
	model.add(BatchNormalization());
	model.add(Dropout(0.44));
	model.add(Dense(256, kernel_initializer='random_uniform'));
	model.add(LeakyReLU(alpha=0.1));
	model.add(BatchNormalization());
	model.add(Dropout(0.37));
	model.add(Dense(7, activation='softmax'));

	print(model.summary());
	return model;

def _shuffle(x, y):
	randomize = np.arange(len(x));
	np.random.shuffle(randomize);
	return x[randomize], y[randomize];
def validation(x, y, percentage):
	all_data_size = len(x);
	train_data_size = int(all_data_size * percentage);
	x_all, y_all = _shuffle(x,y);

	x_train, y_train = x_all[0:train_data_size], y_all[0:train_data_size];
	x_valid, y_valid = x_all[train_data_size:], y_all[train_data_size:];
	return x_train, y_train, x_valid, y_valid

file = open("train.csv", "rb");
reader = pd.read_csv(file);
rawdata = reader.as_matrix();
file.close();
nums_train = rawdata.shape[0];
x_all = np.zeros((rawdata.shape[0], 48*48));
y_all = np.zeros((rawdata.shape[0], ));
for i in range(rawdata.shape[0]):
	x_all[i] = rawdata[i,1].split();
	y_all[i] = rawdata[i,0];
x_all = x_all.astype('float32');
x_all = x_all / 255;

x_all = x_all.reshape((nums_train, 48, 48, 1)); # 48*48 image size; 1 -> gray scale
y_all = np_utils.to_categorical(y_all, 7); # to one hot code

np.random.seed(100); # 8 : 100
x_train, y_train, x_valid, y_valid = validation(x_all, y_all, 0.9);


datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=[0.8, 1.2], shear_range=0.2, horizontal_flip=True)

model2 = model_structure_2();
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']);
batchsize = 256;
hist2 =  model2.fit_generator(datagen.flow(x_train, y_train, batch_size=batchsize), steps_per_epoch=3*len(x_train)//batchsize,  epochs = 30, validation_data=(x_valid, y_valid));
model2.save("model3.h5");


model5 = model_structure_5();
model5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']);
batchsize = 256;
hist5 =  model5.fit_generator(datagen.flow(x_train, y_train, batch_size=batchsize), steps_per_epoch=3*len(x_train)//batchsize,  epochs = 30, validation_data=(x_valid, y_valid));
model5.save("model6.h5");

model4 = model_structure_4();
model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']);
batchsize = 256;
hist4 =  model4.fit_generator(datagen.flow(x_train, y_train, batch_size=batchsize), steps_per_epoch=3*len(x_train)//batchsize,  epochs = 30, validation_data=(x_valid, y_valid));
model4.save("model5.h5");


file1 = open("test.csv", "rb");
reader1 = pd.read_csv(file1);
rawdata1 = reader1.as_matrix();
file1.close();

nums_test = rawdata1.shape[0];
x_test = np.zeros((rawdata1.shape[0], 48*48));

for i in range(rawdata1.shape[0]):
	x_test[i] = rawdata1[i,1].split();

x_test = x_test.astype('float32');
x_test = x_test / 255;
x_test = x_test.reshape((nums_test, 48, 48, 1)); # 48*48 image size; 1 -> gray scale

y_predict1 = model2.predict_classes(x_test);
y_predict2 = model4.predict_classes(x_test);
y_predict3 = model5.predict_classes(x_test);


result = csv.writer(open(sys.argv[1], 'w+'), delimiter = ',', lineterminator = '\n');
result.writerow(['id', 'label'])
for i in range(len(y_predict1)):
	vote = [0,0,0,0,0,0,0];
	vote = np.array(vote);
	vote[y_predict1] += 1;
	vote[y_predict2] += 1;
	vote[y_predict3] += 1;
	index = np.where(vote == np.max(vote));
	result.writerow([(i), index[0][0]] );


del model2;
del model5;
del model4;