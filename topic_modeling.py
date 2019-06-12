#!/usr/bin/env python3
import sklearn_topic_modeling as skltm
import gensim_topic_modeling as gsmtm


class TopicModelingClass:
	def __init__(self, filepath, parser, data_processor):
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
			self.collectiontm = gsmtm.TopicModelingClass(self.docs, self.algorithms_used, self.topic_visualization)
			self.collectiontm.run_topic_modeling()
