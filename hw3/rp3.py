# Hw3 CNN
# python3 this.py 
import matplotlib
matplotlib.use('Agg')
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

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def generate_model_structure():
	model = Sequential();
	
	model.add(Conv2D( filters=64, kernel_size=(5,5), padding='same', kernel_initializer='random_uniform', input_shape=(48,48,1)));
	model.add(LeakyReLU(alpha=0.1));
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
	# model.add(Conv2D( filters=256, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	# model.add(LeakyReLU(alpha=0.1));
	# model.add(Dropout(0.25));
	# model.add(Conv2D( filters=256, kernel_size=(3,3), padding='same', kernel_initializer='random_uniform')); 
	# model.add(LeakyReLU(alpha=0.1));
	model.add(BatchNormalization());
	model.add(MaxPooling2D(pool_size=(2,2))); # 64 x 6 x 6
	model.add(Dropout(0.25));

	model.add(Flatten());
	model.add(Dense(256,  kernel_initializer='random_uniform'));
	model.add(LeakyReLU(alpha=0.1));
	model.add(BatchNormalization());
	model.add(Dropout(0.44));
	# model.add(Dense(256, kernel_initializer='random_uniform'));
	# model.add(LeakyReLU(alpha=0.1));
	# model.add(BatchNormalization());
	# model.add(Dropout(0.37));
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
def plot_confusion_matrix( cnf, classes, title='Confusion Matrix)', cmap=plt.cm.Blues):
	# normalize
	cnf.astype('float32');
	result = np.zeros(cnf.shape);
	summ = np.sum(cnf, axis=1);
	print(summ);
	for i in range(cnf.shape[0]):
		result[i,:] = cnf[i,:] / summ[i];
	print(result);

	plt.imshow(result, cmap=cmap);
	plt.title(title);
	plt.colorbar();
	tick_marks = np.arange(len(classes));
	plt.xticks(tick_marks, classes);
	plt.yticks(tick_marks, classes);
	plt.xlabel('Predict');
	plt.ylabel('True');


classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
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
# x_all = normalization(x_all);
# print(x_all.shape)
x_all = x_all.reshape((nums_train, 48, 48, 1)); # 48*48 image size; 1 -> gray scale

np.random.seed(100);
x_train, y_train, x_valid, y_valid = validation(x_all, y_all, 0.9);
print('Check: ', np.mean(x_train));
y_train = np_utils.to_categorical(y_train, 7); # to one hot code


# datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=[0.8, 1.2], shear_range=0.2, horizontal_flip=True)
# model = generate_model_structure();
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']);
# batchsize = 256;
# hist =  model.fit_generator(datagen.flow(x_train, y_train, batch_size=batchsize), steps_per_epoch=3*len(x_train)//batchsize,  epochs = 10, validation_data=(x_valid, y_valid));
# model.save(sys.argv[1]);

model = load_model('rp_3.h5py');
y_pred = model.predict_classes(x_valid);

cnf_matrix = confusion_matrix(y_valid, y_valid);
plot_confusion_matrix(cnf_matrix, classes);

del model;