# python3 this.py mode epochs output.csv model_name
import io, os, math, sys, csv, random, pickle
import jieba
import pandas as pd
import numpy as np
from gensim.models import word2vec, Word2Vec
from gensim import models

from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Input,LSTM,Dropout,Dense,Activation, GRU, merge, Lambda, Embedding,Convolution1D,Conv1D,Masking,Concatenate,BatchNormalization,Dropout
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint


class QAModel():
	def get_cosine_similarity(self):
		dot = lambda a, b: K.batch_dot(a, b, axes=1)
		return lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())

	def Buildmodel(self, weights):
		margin = 0.05;
		ques_timestep = 30;
		ans_timestep = 30;

		question = Input(shape=(ques_timestep, ), name='question');
		answer = Input(shape=(ans_timestep, ), name='answer');
		answer_good = Input(shape=(ans_timestep, ), dtype='int32', name='answer_good_base')
		answer_bad = Input(shape=(ans_timestep, ), dtype='int32', name='answer_bad_base')

		qa_embedding = Embedding(input_dim=len(weights),output_dim=weights.shape[1],mask_zero=True,weights=[weights], trainable=False)
		gru = GRU(units=136, dropout=0.2, return_sequences=False, name='GRU');

		question_ = qa_embedding(question);
		answer_ = qa_embedding(answer);
		question_encoded = gru(question_);
		answer_encoded = gru(answer_);

		similarity = self.get_cosine_similarity();
		qa_merge = merge(inputs=[question_encoded, answer_encoded], mode=similarity, output_shape=lambda _: (None, 1));

		GRU_model = Model(inputs=[question, answer], outputs=qa_merge);
		good_similarity = GRU_model([question, answer_good]);
		bad_similarity = GRU_model([question, answer_bad]);

		loss = merge([good_similarity, bad_similarity], mode=lambda x: K.relu(margin - x[0] + x[1]), output_shape=lambda x: x[0]);

		training_model = Model(inputs=[question, answer_good, answer_bad], outputs=loss, name='training_model')
		training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")
		prediction_model = Model(inputs=[question, answer_good], outputs=good_similarity, name='prediction_model')
		prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop");

		return training_model, prediction_model;

	def LSTMCNNmodel(self,weights, hidden_dim = 100):
		margin = 0.05
		enc_timesteps = 30
		dec_timesteps = 30
		# hidden_dim = 100

        # initialize the question and answer shapes and datatype
		question = Input(shape=(enc_timesteps,), dtype='int32', name='question_base')
		answer = Input(shape=(dec_timesteps,), dtype='int32', name='answer')
		answer_good = Input(shape=(dec_timesteps,), dtype='int32', name='answer_good_base')
		answer_bad = Input(shape=(dec_timesteps,), dtype='int32', name='answer_bad_base')

		qa_embedding = Embedding(input_dim=len(weights),output_dim=weights.shape[1],mask_zero=True,weights=[weights], trainable=False)
		question_embedding = qa_embedding(question)
		answer_embedding = qa_embedding(answer)

        # pass the question embedding through bi-lstm
		f_rnn = LSTM(hidden_dim, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)
		b_rnn = LSTM(hidden_dim, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)
		qf_rnn = f_rnn(question_embedding)
		qb_rnn = b_rnn(question_embedding)
		question_pool = merge([qf_rnn, qb_rnn], mode='concat', concat_axis=-1)
		af_rnn = f_rnn(answer_embedding)
		ab_rnn = b_rnn(answer_embedding)
		answer_pool = merge([af_rnn, ab_rnn], mode='concat', concat_axis=-1)
		print(answer_pool)

        # pass the embedding from bi-lstm through cnn
		cnns = [Convolution1D(filter_length=filter_length,nb_filter=500,activation='tanh',border_mode='same') for filter_length in [1, 2, 3, 5]] 
		for cnn in cnns:
			cnn.supports_masking = True
		question_cnn = merge([cnn(question_pool) for cnn in cnns], mode='concat') # merge: (None,30,500)*4->(None,30,2000)
		answer_cnn = merge([cnn(answer_pool) for cnn in cnns], mode='concat')

		drop = Dropout(0.2);
		question_cnn = drop(question_cnn);
		answer_cnn = drop(answer_cnn);
		
        # apply max pooling
		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
		# maxpool.__setattr__('supports_masking',True)
		maxpool.supports_masking = True
		question_pool = maxpool(question_cnn)
		answer_pool = maxpool(answer_cnn)


        # get the cosine similarity
		similarity = self.get_cosine_similarity()
		merged_model = merge([question_pool, answer_pool],mode=similarity, output_shape=lambda _: (None, 1))
		lstm_convolution_model = Model(inputs=[question, answer], outputs=merged_model, name='lstm_convolution_model')
		print(lstm_convolution_model.summary())
		good_similarity = lstm_convolution_model([question, answer_good])
		bad_similarity = lstm_convolution_model([question, answer_bad])

        # compute the loss
		loss = merge([good_similarity, bad_similarity],mode=lambda x: K.relu(margin - x[0] + x[1]),output_shape=lambda x: x[0])

        # return the training and prediction model
		adam = Adam(lr=0.0005)
		prediction_model = Model(inputs=[question, answer_good], outputs=good_similarity, name='prediction_model')
		prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=adam)
		training_model = Model(inputs=[question, answer_good, answer_bad], outputs=loss, name='training_model')
		training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=adam)

		print(training_model.summary())
		return training_model, prediction_model

	def BIGRUCNNmodel(self,weights, hidden_dim = 128):
		margin = 0.05
		enc_timesteps = 30
		dec_timesteps = 30
		# hidden_dim = 128

        # initialize the question and answer shapes and datatype
		question = Input(shape=(enc_timesteps,), dtype='int32', name='question_base')
		answer = Input(shape=(dec_timesteps,), dtype='int32', name='answer')
		answer_good = Input(shape=(dec_timesteps,), dtype='int32', name='answer_good_base')
		answer_bad = Input(shape=(dec_timesteps,), dtype='int32', name='answer_bad_base')

		qa_embedding = Embedding(input_dim=len(weights),output_dim=weights.shape[1],mask_zero=True,weights=[weights], trainable=False)
		question_embedding = qa_embedding(question)
		answer_embedding = qa_embedding(answer)

		# pass the question embedding through bi-lstm
		gru1 = Bidirectional(GRU(units=hidden_dim, dropout=0.2, recurrent_dropout=0.2,  return_sequences=True, name='BIGRU'), merge_mode='concat');
		question_pool1 = gru1(question_embedding);
		answer_pool1 = gru1(answer_embedding);
		gru2 = Bidirectional(GRU(units=int(hidden_dim/2), dropout=0.2, recurrent_dropout=0.2,  return_sequences=True, name='BIGRU2'), merge_mode='concat');
		question_pool = gru2(question_pool1);
		answer_pool = gru2(answer_pool1);

		# pass the embedding from bi-lstm through cnn
		cnns = [Convolution1D(filter_length=filter_length,nb_filter=300,activation='relu',border_mode='same', kernel_initializer='random_normal') for filter_length in [1, 2, 3, 5]] 
		for cnn in cnns:
			cnn.supports_masking = True
		question_cnn = merge([cnn(question_pool) for cnn in cnns], mode='concat') # merge: (None,30,500)*4->(None,30,2000)
		answer_cnn = merge([cnn(answer_pool) for cnn in cnns], mode='concat')

		
        # apply max pooling
		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
		# maxpool.__setattr__('supports_masking',True)
		maxpool.supports_masking = True
		question_pool = maxpool(question_cnn)
		answer_pool = maxpool(answer_cnn)


        # get the cosine similarity
		similarity = self.get_cosine_similarity()
		merged_model = merge([question_pool, answer_pool],mode=similarity, output_shape=lambda _: (None, 1))
		lstm_convolution_model = Model(inputs=[question, answer], outputs=merged_model, name='bigru_convolution_model')
		print(lstm_convolution_model.summary())
		good_similarity = lstm_convolution_model([question, answer_good])
		bad_similarity = lstm_convolution_model([question, answer_bad])

        # compute the loss
		loss = merge([good_similarity, bad_similarity],mode=lambda x: K.relu(margin - x[0] + x[1]),output_shape=lambda x: x[0])

        # return the training and prediction model
		prediction_model = Model(inputs=[question, answer_good], outputs=good_similarity, name='prediction_model')
		prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="adam")
		training_model = Model(inputs=[question, answer_good, answer_bad], outputs=loss, name='training_model')
		training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="adam")

		print(training_model.summary())
		return training_model, prediction_model

	def GRUCNNmodel(self,weights, hidden_dim = 128):
		margin = 0.05
		enc_timesteps = 30
		dec_timesteps = 30
		# hidden_dim = 128

        # initialize the question and answer shapes and datatype
		question = Input(shape=(enc_timesteps,), dtype='int32', name='question_base')
		answer = Input(shape=(dec_timesteps,), dtype='int32', name='answer')
		answer_good = Input(shape=(dec_timesteps,), dtype='int32', name='answer_good_base')
		answer_bad = Input(shape=(dec_timesteps,), dtype='int32', name='answer_bad_base')

		qa_embedding = Embedding(input_dim=len(weights),output_dim=weights.shape[1],mask_zero=True,weights=[weights], trainable=False)
		question_embedding = qa_embedding(question)
		answer_embedding = qa_embedding(answer)

		# pass the question embedding through bi-lstm
		gru = GRU(units=hidden_dim, dropout=0.2, return_sequences=True, name='GRU');
		question_pool = gru(question_embedding);
		answer_pool = gru(answer_embedding);

		# pass the embedding from bi-lstm through cnn
		cnns = [Convolution1D(filter_length=filter_length,nb_filter=500,activation='tanh',border_mode='same') for filter_length in [1, 2, 3, 5]] 
		for cnn in cnns:
			cnn.supports_masking = True
		question_cnn = merge([cnn(question_pool) for cnn in cnns], mode='concat') # merge: (None,30,500)*4->(None,30,2000)
		answer_cnn = merge([cnn(answer_pool) for cnn in cnns], mode='concat')

		
        # apply max pooling
		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
		# maxpool.__setattr__('supports_masking',True)
		maxpool.supports_masking = True
		question_pool = maxpool(question_cnn)
		answer_pool = maxpool(answer_cnn)


        # get the cosine similarity
		similarity = self.get_cosine_similarity()
		merged_model = merge([question_pool, answer_pool],mode=similarity, output_shape=lambda _: (None, 1))
		lstm_convolution_model = Model(inputs=[question, answer], outputs=merged_model, name='lstm_convolution_model')
		print(lstm_convolution_model.summary())
		good_similarity = lstm_convolution_model([question, answer_good])
		bad_similarity = lstm_convolution_model([question, answer_bad])

        # compute the loss
		loss = merge([good_similarity, bad_similarity],mode=lambda x: K.relu(margin - x[0] + x[1]),output_shape=lambda x: x[0])

        # return the training and prediction model
		prediction_model = Model(inputs=[question, answer_good], outputs=good_similarity, name='prediction_model')
		prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")
		training_model = Model(inputs=[question, answer_good, answer_bad], outputs=loss, name='training_model')
		training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")

		print(training_model.summary())
		return training_model, prediction_model


class TVShow():

	def __init__(self):
		self.index = [];
		self.questions = [];
		self.options = [];
	def parsedata(data):	# 把答案的 'idx:' 去除
		for idx, sent in enumerate(data):
			data[idx] = sent.replace('%d:' %idx, '' ) 
		return data

	def Predict(self, threshold = 0.3, norm='sqrt', punish=1, mode='o', write=True):

		testdata = "testing_data.csv";

		with io.open(testdata, 'r', encoding='utf-8') as content:
			for line in content:
				text = line.replace('\n','').split(',');	# idx, question, option
				if len(text) == 3:
					self.index.append(text[0]);
					questions_ = text[1].replace('/t', ' ').split();
					self.questions.append(questions_);
					options_ = text[2].split('\t');
					self.options.append(TVShow.parsedata(options_))

			del self.index[0];
			del self.questions[0];
			del self.options[0];
		qq = [];
		opop = [];
		

		for quest in self.questions:
			questSeg = [];
			for q in quest:		# ['a','b',...,'n']
				qSeg = TVShow.jiebaSeg(q);
				questSeg += qSeg;
			qq.append(questSeg);

		for options in self.options:
			optSeg = [];
			for one_opt in options:
				oSeg = TVShow.jiebaSeg(one_opt);
				if oSeg is None:
					optSeg.append([])
				optSeg.append(oSeg);
			opop.append(optSeg);

		return qq, opop;


	def jiebaSeg(lines):	# Seperate the imput sequence.
		words = jieba.cut(lines, cut_all=False);
		# Prevent nonsense output.
		segLine = [];
		for word in words:
			if word != ' ' and word != '':
				segLine.append(word);
		return segLine; 




def main(mode=sys.argv[1]):

	outputname = sys.argv[3];
	model_pk = sys.argv[4];

	with io.open('gru_model/word2idx_'+model_pk + '.pk', 'rb') as f:
		word2idx = pickle.load(f)
	with io.open('gru_model/embeddings_matrix_' + model_pk + '.pk', 'rb') as f2:
		embeddings_matrix = pickle.load(f2);

	print('台灣idx : ', word2idx['台灣'])

	qa_model = QAModel()
	# train_model, predict_model = qa_model.Buildmodel(weights = embeddings_matrix);
	train_model, predict_model = qa_model.LSTMCNNmodel(weights = embeddings_matrix, hidden_dim = 100);

	if mode == '-train':

		data = [];
		with io.open('training_data/alltrainseg1.txt', 'r', encoding='utf-8') as content:
			for line in content:
				cont = line.strip('\n').split()
				for idx, word in enumerate(cont):
					try:
						cont[idx] = word2idx[word]
					except Exception as e:
						cont[idx] = 0
				data.append(cont)
		question = []
		answer = []
		for i in range (len(data)-3):
			question.append(data[i] + data[i+1] + data[i+2])
			answer.append(data[i+3])

		question = pad_sequences(question, maxlen=30, padding='pre', truncating='pre', value=0)
		answer_good = pad_sequences(answer, maxlen=30, padding='pre', truncating='pre', value=0)


		def _shuffle(x):
			randomize = np.arange(len(x));
			np.random.shuffle(randomize);
			return x[randomize];

		answer_bad = _shuffle(np.array(answer_good));

		Y = np.zeros(shape=(question.shape[0],));
		
		callback = ModelCheckpoint('lstmcnn_best.h5', monitor='val_loss', save_best_only=True, period=1,verbose=1)
		callback_list = [callback]
		print(train_model.summary())

		train_model.fit([question, answer_good, answer_bad], Y, epochs=5, batch_size=256, validation_split=0.1, verbose=1,callbacks=callback_list)
		train_model.save('gru_train_' + sys.argv[2] + '.h5');
		predict_model.save('gru_predict_' + sys.argv[2] + '.h5');

	elif mode == '-test':
		jieba.set_dictionary("dict.txt.big");
		e = TVShow();
		question, options = e.Predict();
		print(question[0])
		print(options[0]);

		for quest in question:
			for i in range(len(quest)):				
				try: 
					quest[i] = word2idx[quest[i]];
				except Exception as e:
					quest[i] = 0;
		for option in options:
			for one_opt in option:
				for i in range(len(one_opt)):					
					try: 
						one_opt[i] = word2idx[one_opt[i]];
					except Exception as e:
						one_opt[i] = 0

		question = pad_sequences(question, maxlen=30, padding='pre', truncating='pre', value=0);
		print(question[0])

		opop = [];
		for option in options:
			option = pad_sequences(option, maxlen=30, padding='pre', truncating='pre', value=0);
			opop.append(option);


		########## question、opop #################3

		predict_model.load_weights(sys.argv[2]);

		question = np.array(question);
		opop = np.array(opop);
		op0 = opop[:,0];
		op1 = opop[:,1];
		op2 = opop[:,2];
		op3 = opop[:,3];
		op4 = opop[:,4];
		op5 = opop[:,5];

		sims0 = predict_model.predict([question, op0]);
		sims1 = predict_model.predict([question, op1]);
		sims2 = predict_model.predict([question, op2]);
		sims3 = predict_model.predict([question, op3]);
		sims4 = predict_model.predict([question, op4]);
		sims5 = predict_model.predict([question, op5]);

		max_r = [];
		for i in range(len(sims0)):
			temp_cnt = 0;
			temp_value = sims0[i];
			if (sims1[i] > temp_value):
				temp_cnt = 1;
				temp_value = sims1[i];
			if (sims2[i] > temp_value):
				temp_cnt = 2;
				temp_value = sims2[i];
			if (sims3[i] > temp_value):
				temp_cnt = 3;
				temp_value = sims3[i];
			if (sims4[i] > temp_value):
				temp_cnt = 4;
				temp_value = sims4[i];
			if (sims5[i] > temp_value):
				temp_cnt = 5;
				temp_value = sims5[i];
			max_r.append(temp_cnt);
		# print(question[0])
		# print(opop[1])

		outputfile = sys.argv[3];
		result = csv.writer(open(outputfile, 'w+'), delimiter = ',', lineterminator = '\n');
		result.writerow(['id', 'ans'])
		for i in range(len(max_r)):
			result.writerow([(i), max_r[i]] );

	else:
		pass

if __name__ == "__main__":
	main();