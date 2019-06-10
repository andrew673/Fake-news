#!/usr/bin/env python3
import os
import re
import pandas as pd
import pymongo
import json
import csv
import string

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV


#nltk.download('punkt')
#nltk.download('stopwords')
stemming = PorterStemmer()
stops = set(stopwords.words("english")) 

no_features = 1000
no_topics = 3  
no_top_words = 10

filepath = './news_sample.csv'

doc_set = []


def remove_html_tags(text):
        """Remove html tags from a string"""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

def import_content(filepath):
    mng_client = pymongo.MongoClient('localhost', 27017)
    mng_db = mng_client['test_extractor'] 
    collection_name = 'news_sample_trainer'
    db_cm = mng_db[collection_name]
    cdir = os.path.dirname(__file__)
    file_res = os.path.join(cdir, filepath)
    data = pd.read_csv(file_res)
    return data

def section_trimming(doc_set, section):
	doc_set[section] = doc_set[section].str.replace(r'[^\w\s]', '')
	doc_set[section] = doc_set[section].str.replace(r'[0-9]', '')
	doc_set[section] = doc_set[section].str.lower()
	doc_set[section] = doc_set.apply(lambda row: nltk.word_tokenize(row[section]), axis=1)
	# poti scrie in licenta si despre stemming ca e optional, insa rez nu sunt relevante
	#doc_set[section] = doc_set.apply(lambda row: [stemming.stem(word) for word in row[section]], axis=1)
	doc_set[section] = doc_set.apply(lambda row: [w for w in row[section] if not w in stops and len(w) > 2 and not w[0:3] == "www" and not w[0:4] == "http"], axis=1)
	# join optional in cazul in care doresc sa resalvez datele procesate in mongo
	#doc_set[section] = doc_set.apply(lambda row: ( " ".join(row[section])), axis=1)

	#print(doc_set[section])
	return doc_set

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def evaluate_topics(doc_set, comparing_section, model, feature_names, idx):
	print("Acrticle ", end="")
	print(idx)
	if len(doc_set[comparing_section][idx]) == 0:
		print("Topics from this article cannot be evaluated - non existing meta keywords")
		return

	for topic_idx, topic in enumerate(model.components_):
		n_top_words = [feature_names[i] for i in topic.argsort()]
		print("Topic", end =" ")
		print(topic_idx, end =" ")
		print("manage to capture :", end =" ")
		print((len(set(n_top_words).intersection(doc_set[comparing_section][idx])) / len(doc_set[comparing_section][idx])) * 100, end ="")
		print("%", end = " ")
		print("of the meta keywords")
		n_top_words = [feature_names[i] for i in topic.argsort()[:-5 - 1:-1]]
		print("First 5 top words from the topic are covered in a proportion of", end =" ")
		print((len(set(n_top_words).intersection(doc_set[comparing_section][idx])) / 5) * 100, end ="")
		print("%", end = " ")
		print("in the meta keywords")


def apply_nmf_on_articles(flat_words, doc_set, comparing_section):
	for (i, list) in enumerate(flat_words):
		# NMF is able to use tf-idf
		tfidf_vectorizer = TfidfVectorizer(max_features=no_features, stop_words='english')
		tfidf = tfidf_vectorizer.fit_transform(list)
		tfidf_feature_names = tfidf_vectorizer.get_feature_names()

		# Run NMF
		nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='random').fit(tfidf)
		#display_topics(nmf, tfidf_feature_names, no_top_words)
		evaluate_topics(doc_set, comparing_section, nmf, tfidf_feature_names, i)
		print("\n\n\n")

def apply_lda_on_articles(flat_words, doc_set, comparing_section):
	for (i, list) in enumerate(flat_words):
		# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
		tf_vectorizer = CountVectorizer(max_features=no_features, stop_words='english')
		tf = tf_vectorizer.fit_transform(list)
		tf_feature_names = tf_vectorizer.get_feature_names()

		lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
		#display_topics(lda, tf_feature_names, no_top_words)
		evaluate_topics(doc_set, comparing_section, lda, tf_feature_names, i)
		print("\n\n\n")

def apply_enhanced_lda_on_articles(flat_words, doc_set, comparing_section):
	for (i, list) in enumerate(flat_words):
		# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
		tf_vectorizer = CountVectorizer(max_features=no_features, stop_words='english')
		tf = tf_vectorizer.fit_transform(list)
		tf_feature_names = tf_vectorizer.get_feature_names()

		lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
		#display_topics(lda, tf_feature_names, no_top_words)

		search_params = {'n_components': [3, 5, 10, 15], 'learning_decay': [.5, .7, .9]}
		model = GridSearchCV(lda, param_grid=search_params).fit(tf)
		best_lda = model.best_estimator_

		evaluate_topics(doc_set, comparing_section, best_lda, tf_feature_names, i)
		print("\n\n\n")

def apply_lsa_on_articles(flat_words, doc_set, comparing_section):
	for (i, list) in enumerate(flat_words):
		# LSA
		tf_vectorizer = CountVectorizer(max_features=no_features, stop_words='english')
		tf = tf_vectorizer.fit_transform(list)
		tf_feature_names = tf_vectorizer.get_feature_names()

		# SVD to reduce dimensionality:
		svd = TruncatedSVD(n_components=no_topics, algorithm='randomized', n_iter=10).fit(tf)
		display_topics(svd, tf_feature_names, no_top_words)
		evaluate_topics(doc_set, comparing_section, svd, tf_feature_names, i)
		print("\n\n\n")


doc_set = import_content(filepath)
doc_set['title'] = doc_set['title'].fillna(0)
doc_set['meta_keywords'] = doc_set['meta_keywords'].fillna(0)
doc_set = section_trimming(doc_set, 'title')
doc_set = section_trimming(doc_set, 'content')
doc_set = section_trimming(doc_set, 'meta_keywords')

flat_words = doc_set.apply(lambda row: [w for w in row['title'] + row['content']], axis = 1)

#apply_nmf_on_articles(flat_words, doc_set, 'meta_keywords')
apply_lda_on_articles(flat_words, doc_set, 'meta_keywords')
#apply_enhanced_lda_on_articles(flat_words, doc_set, 'meta_keywords')
#apply_lsa_on_articles(flat_words, doc_set, 'meta_keywords')




