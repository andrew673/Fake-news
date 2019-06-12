#!/usr/bin/env python3
import os
import re
import pandas as pd
import pymongo
import string

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

class DataProcessor:
	def __init__(self, filepath, labels):
		self.filepath = filepath
		self.db_collection = None
		self.data_frame = None
		self.labels = labels
		self.docs = None

	def df2docs(self):
		self.docs = self.data_frame.apply(lambda row: [w for w in row[self.labels[0]]], axis = 1)
		for i in range(1, len(self.labels)):
			self.docs += self.data_frame.apply(lambda row: [w for w in row[self.labels[i]]], axis = 1)

	def trim_label(self, label):
		stops = set(stopwords.words("english"))
		self.data_frame[label] = self.data_frame[label].str.replace(r'[^\w\s]', '')
		self.data_frame[label] = self.data_frame[label].str.replace(r'[0-9]', '')
		self.data_frame[label] = self.data_frame[label].str.lower()
		self.data_frame[label] = self.data_frame.apply(lambda row: nltk.word_tokenize(row[label]), axis=1)
		self.data_frame[label] = self.data_frame.apply(lambda row: [w for w in row[label] if not w in stops and len(w) > 2 and len(w) < 20], axis=1)
		# join optional in cazul in care doresc sa resalvez datele procesate in mongo
		#doc_set[section] = doc_set.apply(lambda row: ( " ".join(row[section])), axis=1)
		lemmatizer = WordNetLemmatizer()
		self.data_frame[label] = self.data_frame.apply(lambda row: [lemmatizer.lemmatize(w) for w in row[label]], axis=1)

	def trim_data(self):
		for label in self.labels:
			self.data_frame[label] = self.data_frame[label].fillna(0)
			self.trim_label(label)
		self.df2docs()


	def import_content(self):
		mng_client = pymongo.MongoClient('localhost', 27017)
		mng_db = mng_client['test_extractor']
		collection_name = 'news_sample_trainer'
		db_cm = mng_db[collection_name]

		cdir = os.path.dirname(__file__)
		file_res = os.path.join(cdir, self.filepath)
		data = pd.read_csv(file_res)

		#db_cm.remove()
		#db_cm.insert(data)
		self.db_collection = db_cm
		self.data_frame = data
		self.trim_data()
