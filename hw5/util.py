# python3 this.py output.model
# Seperate the word and Word2Vec
# no.50, 51

import io
import os
import sys
import csv
import string
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

class StringTools():
	def Manage_Abbreviation(sentence_split):
		sentence_split.reverse();
		new_split = []
		while sentence_split:
			word = sentence_split.pop();
			if word != "'":
				new_split.append(word);
			else:
				try:
					next = sentence_split.pop();
					new_split[-1] += next;
				except Exception as e:
					pass;
		return new_split;
	def Delete_Duplicate_chars(word):
		if len(word) > 1:
			newstring = word[0];
			for char in word[1:]:
				if char != newstring[-1]:
					newstring += char;
			if len(newstring) == 1:
				return 'xxx';
			return newstring;
		else:
			return word;


class DataManager():

	def __init__(self):
		self.label = [];
		self.data = [];

	def readfile(self, stoplist, mode='plus'):

		inputfile = 'training_label.txt';	

		cnt = 0;
		with io.open(inputfile, 'r', encoding='utf-8') as content:
			
			for line in content:
				cont = line.lower().split('+++$+++');

				if len(cont) == 2:
					self.label.append(cont[0]);
					self.data.append([]);
					# cont_split = cont[1].replace('\n','').split(' ');
					
					cont_split = cont[1].replace('\n','').replace('.','').replace(',','').split(' ');

					### Deal with the token '.
					cont_split = StringTools.Manage_Abbreviation(cont_split);

					for idx, word in enumerate(cont_split):
						cont_split[idx] = word.translate(str.maketrans("","", string.punctuation));
						

					for word in cont_split:
						if word != ' ' and word != '' and word != '\n' and word != '\t' and word not in stoplist:

							## Delete the duplicate word in a string.
							
							new_word = StringTools.Delete_Duplicate_chars(word);
							self.data[cnt].append(new_word);
					cnt += 1;
		print(self.data[41492]);

		if mode == 'plus':
			with io.open('training_nolabel.txt', 'r', encoding='utf-8') as content:
				for line in content:
					cont_split = line.lower().replace('\n','').replace('.','').replace(',','').split(' ');
					cont_split = StringTools.Manage_Abbreviation(cont_split);
					for idx, word in enumerate(cont_split):
						cont_split[idx] = word.translate(str.maketrans("","", string.punctuation));
					temp_word = [];
					for word in cont_split:
						if word != ' ' and word != '' and word != '\n' and word != '\t' and word not in stoplist:
							new_word = StringTools.Delete_Duplicate_chars(word);
							temp_word.append(new_word);
					self.data.append(temp_word)

		
		print(self.data[200001])

		print('--- Readfile Success ---' );

	def WordVec_label(self, stoplist):

		DataManager.readfile(self, stoplist);	# [label], [[data1], [data2]...]

		model = Word2Vec(self.data, size = 200, window = 8, min_count = 10, iter = 15);
		model.save(sys.argv[1])
		
		print('--- Word2Vector Success ---' );
		return model
# class model():



if __name__ == '__main__':

	# stoplist = set('for a of the and to in on is are he she i they you me him her their there that this with'.split()); # dictionary
	stoplist = set('of to a b c d e f g h i j l m n o p q r s t u v w x y'.split());

	e = DataManager();
	model = e.WordVec_label(stoplist);