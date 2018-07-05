# python3 this.py mode output.csv
import io, os, math, sys, csv, random, pickle
import jieba
import pandas as pd
import numpy as np
from gensim.models import word2vec, Word2Vec
from gensim import models

from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Input,LSTM,Dropout,Dense,Activation, GRU, merge, Lambda, Embedding,Convolution1D,Conv1D,Masking,Concatenate, Bidirectional
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint


class QAModel():
	def get_cosine_similarity(self):
		dot = lambda a, b: K.batch_dot(a, b, axes=1)
		return lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())

	def LSTMCNN4model(self,weights, hidden_dim = 100):
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
		# print(lstm_convolution_model.summary())
		good_similarity = lstm_convolution_model([question, answer_good])
		bad_similarity = lstm_convolution_model([question, answer_bad])

        # compute the loss
		loss = merge([good_similarity, bad_similarity],mode=lambda x: K.relu(margin - x[0] + x[1]),output_shape=lambda x: x[0])

        # return the training and prediction model
		adam = Adam(lr=0.001)
		prediction_model = Model(inputs=[question, answer_good], outputs=good_similarity, name='prediction_model')
		prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=adam)
		training_model = Model(inputs=[question, answer_good, answer_bad], outputs=loss, name='training_model')
		training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=adam)

		# print(training_model.summary())
		return training_model, prediction_model

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
		f_rnn = LSTM(hidden_dim, return_sequences=True)
		b_rnn = LSTM(hidden_dim, return_sequences=True)
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

		testdata = sys.argv[1];

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

def main(mode='-sum'):

	qa_model = QAModel()
	
	if mode == '-sum':
########################## read the testfile ################################
		jieba.set_dictionary("dict.txt.big");

		with io.open('gru_model/word2idx_w2v_150.pk', 'rb') as f:
			word2idx_150 = pickle.load(f)
		with io.open('gru_model/embeddings_matrix_w2v_150.pk', 'rb') as f2:
			embeddings_matrix_150 = pickle.load(f2);
		with io.open('gru_model/word2idx_w2v_180.pk', 'rb') as f:
			word2idx_180 = pickle.load(f)
		with io.open('gru_model/embeddings_matrix_w2v_180.pk', 'rb') as f2:
			embeddings_matrix_180 = pickle.load(f2);

		e_150 = TVShow();
		question_150, options_150 = e_150.Predict();
		e_180 = TVShow();
		question_180, options_180 = e_180.Predict();

		for quest in question_150:
			for i in range(len(quest)):
				try: 
					quest[i] = word2idx_150[quest[i]];
				except Exception as e:
					quest[i] = 0;
		for option in options_150:
			for one_opt in option:
				for i in range(len(one_opt)):
					try: 
						one_opt[i] = word2idx_150[one_opt[i]];
					except Exception as e:
						one_opt[i] = 0
		for quest in question_180:
			for i in range(len(quest)):
				try: 
					quest[i] = word2idx_180[quest[i]];
				except Exception as e:
					quest[i] = 0;
		for option in options_180:
			for one_opt in option:
				for i in range(len(one_opt)):
					try: 
						one_opt[i] = word2idx_180[one_opt[i]];
					except Exception as e:
						one_opt[i] = 0

		question_150 = pad_sequences(question_150, maxlen=30, padding='pre', truncating='pre', value=0);
		question_180 = pad_sequences(question_180, maxlen=30, padding='pre', truncating='pre', value=0);

		opop_150 = [];
		opop_180 = [];

		for option in options_150:
			option = pad_sequences(option, maxlen=30, padding='pre', truncating='pre', value=0);
			opop_150.append(option);
		for option in options_180:
			option = pad_sequences(option, maxlen=30, padding='pre', truncating='pre', value=0);
			opop_180.append(option);

		question_150 = np.array(question_150)
		opop_150 = np.array(opop_150)
		question_180 = np.array(question_180)
		opop_180 = np.array(opop_180)
#############################################################################

########################### load model #########################			

		train_model, model1 = qa_model.LSTMCNN4model(weights = embeddings_matrix_180, hidden_dim = 150);
		model1.load_weights('gru_model/lstmcnn4_180_150.h5');
		train_model, model2 = qa_model.LSTMCNN4model(weights = embeddings_matrix_150, hidden_dim = 150);
		model2.load_weights('gru_model/lstmcnn4_150_150.h5');
		train_model, model3 = qa_model.GRUCNNmodel(weights = embeddings_matrix_150, hidden_dim = 128);
		model3.load_weights('gru_model/cnngru1_best_150_128.h5');
		train_model, model4 = qa_model.LSTMCNN4model(weights = embeddings_matrix_180, hidden_dim = 100); 
		model4.load_weights('gru_model/lstmcnn4_180_100.h5');

		out1 = np.zeros((len(opop_150),6))
		out2 = np.zeros((len(opop_150),6))
		out3 = np.zeros((len(opop_150),6))
		out4 = np.zeros((len(opop_150),6))

		for i in range(6):
			out = model1.predict([question_180, opop_180[:,i]]).reshape((np.shape(out1)[0]))
			out1[:,i] = out
			out = model2.predict([question_150, opop_150[:,i]]).reshape((np.shape(out1)[0]))
			out2[:,i] = out
			out = model3.predict([question_150, opop_150[:,i]]).reshape((np.shape(out1)[0]))
			out3[:,i] = out
			out = model4.predict([question_180, opop_180[:,i]]).reshape((np.shape(out1)[0]))
			out4[:,i] = out
			print('--- Predict Index', i, ' Finish ---')

		final_out = out1 + out2 + out3 + out4
		final_out = np.argmax(final_out,axis=1)

		outputfile = sys.argv[2]
		result = csv.writer(open(outputfile, 'w+'), delimiter = ',', lineterminator = '\n')
		result.writerow(['id', 'ans'])
		for i in range(len(final_out)):
			result.writerow([(i), final_out[i]] )

	else:
		pass

if __name__ == "__main__":
	main();