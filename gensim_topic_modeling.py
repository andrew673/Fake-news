#!/usr/bin/env python3
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim
from sklearn.decomposition import NMF


class TopicModelingClass:
	def __init__(self, docs, algorithms_used, topic_visualization, dictionary, corpus):
		self.docs = docs
		self.algorithms_used = algorithms_used
		self.topic_visualization = topic_visualization
		self.dictionary = dictionary
		self.corpus = corpus

		self.num_topics = 10
		self.chunksize = 500 
		self.passes = 20 
		self.iterations = 400
		self.eval_every = 1
		self.kappa = 0.1

	def apply_nmf_on_articles(self):
		nmf_model = Nmf(corpus=self.corpus, num_topics=50, \
			id2word=self.id2word, chunksize=self.chunksize, \
			passes=self.passes, kappa=self.kappa, \
			h_max_iter=self.iterations, eval_every=self.eval_every)
		print(nmf_model.print_topics())

		if self.topic_visualization is True:
			nmf_display = pyLDAvis.gensim.prepare(nmf_model, self.corpus, \
				self.dictionary, sort_topics=True)
			pyLDAvis.show(nmf_display)

	def apply_lda_on_articles(self):
		lda_model = LdaModel(corpus=self.corpus, id2word=self.dictionary, \
			chunksize=self.chunksize, alpha='auto', eta='auto', \
			iterations=self.iterations, num_topics=self.num_topics, \
			passes=self.passes, eval_every=self.eval_every)
		print(lda_model.print_topics())

		if self.topic_visualization is True:
			lda_display = pyLDAvis.gensim.prepare(lda_model, self.corpus, \
				self.dictionary, sort_topics=True)
			pyLDAvis.show(lda_display)

	def apply_lsa_on_articles(self):
		lsa_model = LsiModel(corpus=self.corpus, id2word=self.dictionary, \
			chunksize=self.chunksize, num_topics=self.num_topics)
		print(lsa_model.print_topics())

	def run_topic_modeling(self):
		for alg in self.algorithms_used:
			if alg == "--nmf":
				self.apply_nmf_on_articles()
			elif alg == "--lda":
				self.apply_lda_on_articles()
			elif alg == "--lsa":
				self.apply_lsa_on_articles()
