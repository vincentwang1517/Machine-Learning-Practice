# python3 this.py Word2Vector.model RNN_model.h5 output.csv
import csv
import sys
import string
import numpy as np
import pandas as pd

from gensim import corpora
from gensim.models import Word2Vec,KeyedVectors
from keras.models import Sequential,load_model 
from keras.layers import LSTM,Dropout,Dense,Activation
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint     

from util import StringTools


class  TextTest():
    
    def read_data(file, stoplist):

        print('--- Reading data ---')
        with open(file,'r',encoding = 'utf8') as content:
            data = []            
            for line in content:

                cont = line.lower().replace('\n','').replace(',','###$###',1).split('###$###')[1].replace('.','').replace(',','').split(' '); 

                cont = StringTools.Manage_Abbreviation(cont);

                # for idx, word in enumerate(cont):
                #         cont[idx] = word.translate(str.maketrans("","", string.punctuation));

                words_temp = []; 
                for word in cont:
                    if word != ' ' and word != '' and word != '\n' and word != '\t' and word not in stoplist:

                        new_word = StringTools.Delete_Duplicate_chars(word);

                        words_temp.append(new_word);

                data.append(words_temp);

        del data[0];
        print(data[1])
        print(data[2])
        print(data[29031])
        return data
    
    
    def Word2vec(data, time_step):

        print('--- Word2Vec ---');
        model = Word2Vec.load('Word2Vector_rp4.model')
        vector_size = model.vector_size

        
        input_ = np.zeros((len(data),time_step,vector_size))

        for i, words in enumerate(data):
            for j in range(len(words)):
                try:
                    input_[i][time_step-len(words)+j] = model[words[j]];
                except Exception as e:
                    input_[i][time_step-len(words)+j] = np.zeros(vector_size);
        
        return input_
    

if __name__ == '__main__':

    stoplist = set('of to a b c d e f g h i j l m n o p q r s t u v w x y'.split());
    time_step = 40;

    testfile = sys.argv[1];
    data = TextTest.read_data(testfile, stoplist);

    input_ = TextTest.Word2vec(data, time_step);

    RNN_model = load_model(sys.argv[3]);
    print(RNN_model.summary());

    print('--- Predict ---');
    output = RNN_model.predict(input_);

    result = csv.writer(open(sys.argv[2] , 'w+'), delimiter = ',', lineterminator = '\n');
    result.writerow(['id', 'label']);
    for i in range(len(output)):
    	result.writerow(('%d' %(i), int(np.round(output[i])))) ;




