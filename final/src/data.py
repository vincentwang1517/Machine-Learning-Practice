# this.py mode(1 or 2) model_name
# 68

import io
import os
import jieba
import math
import sys
from gensim.models import word2vec

class Data_Preprocessing():

	def Split1(self):
		jieba.set_dictionary('dict.txt.big');	# 使用繁中字庫，需下載

		main_datafile = sys.argv[2];
		output_file = "training_data/alltrainseg" + sys.argv[1] + ".txt";
		output = io.open(output_file,'w', encoding="utf-8");

		# Split data.
		with io.open(os.path.join(main_datafile), 'r', encoding="utf-8") as content:
			for line in content:
				# Cut the sentence into seperate words.
				words = jieba.cut(line, cut_all=False);	
				# Write output
				wordcount = 0;
				for id, word in enumerate(words):
					if word != '\n' and word != ' ' and word != '，' and word != '、' and word != '"' and word != '(' and word != ')' and word != '-' and word != '.':
						output.write(word + ' ');
						wordcount += 1;
				if wordcount != 0:
					output.write(u'\n');	# unicode
		output.close();

	def Split2(self):
		jieba.set_dictionary('dict.txt.big');	# 使用繁中字庫，需下載
		# load stopwords set
		stopword_set = set();
		with open('stop_words_vi.txt', 'r', encoding='utf-8') as stopwords:
			for stopword in stopwords:
				stopword_set.add(stopword.strip('\n'))

		main_datafile = sys.argv[2];
		output_file = "training_data/alltrainseg" + sys.argv[1] + ".txt";
		output = io.open(output_file,'w', encoding="utf-8");

		# Split data.
		with io.open(os.path.join(main_datafile), 'r', encoding="utf-8") as content:
			for line in content:
				# Cut the sentence into seperate words.
				words = jieba.cut(line, cut_all=False);	
				# Write output
				wordcount = 0;
				for id, word in enumerate(words):
					if word != '\n' and word != ' ' and word not in stopword_set:
						output.write(word + ' ');
						wordcount += 1;
				if wordcount != 0:
					output.write(u'\n');	# unicode
		output.close();

	def Embedding(self, model_address):
		
		datafile = "training_data/alltrainseg" + sys.argv[1] + ".txt";

		sentences = word2vec.Text8Corpus(datafile);		#?????

		model = word2vec.Word2Vec(sentences, size=150, iter = 13, min_count = 3, window = 8);

		model.save(model_address);

if __name__ == '__main__':

	model_output = "w2v_150.model";

	e = Data_Preprocessing();
	if sys.argv[1] == '1':
		e.Split1();
	elif sys.argv[1] == '2':
		e.Split2();

	e.Embedding(model_output);