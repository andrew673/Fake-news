#!/usr/bin/env python3
import sklearn_topic_modeling as skltm
import gensim_topic_modeling as gsmtm

from gensim.corpora.dictionary import Dictionary


class TopicModelingClass:
	def __init__(self, parser, data_processor):
		self.docs = data_processor.docs
		self.type_of_dataset = parser.type_of_dataset
		self.algorithms_used = parser.algorithms_used
		self.topic_visualization = parser.topic_visualization
		self.uniquetm = None
		self.collectiontm = None

	def run(self):
		if self.type_of_dataset == "--unique":
			self.uniquetm = skltm.TopicModelingClass(self.docs, self.algorithms_used)
			self.uniquetm.run_topic_modeling()
		elif self.type_of_dataset == "--collection":
			dictionary = Dictionary(self.docs)
			dictionary.filter_extremes(no_below=10, no_above=0.2)

			#Create dictionary and corpus required for Topic Modeling
			corpus = [dictionary.doc2bow(doc) for doc in self.docs]
			self.collectiontm = gsmtm.TopicModelingClass(self.docs, self.algorithms_used, \
				self.topic_visualization, dictionary, corpus)
			self.collectiontm.run_topic_modeling()
