#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

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

	def get_coherence_cv(self, model):
		# Compute Coherence Score using c_v
		coherence_model = CoherenceModel(model=model, texts=self.docs, dictionary=self.dictionary, coherence='c_v')
		coherence = coherence_model.get_coherence()
		print('\nCoherence Score C_V: ', coherence)

	def get_coherence_umass(self, model):
		# Compute Coherence Score using UMass
		coherence_model = CoherenceModel(model=model, texts=self.docs, dictionary=self.dictionary, coherence="u_mass")
		coherence = coherence_model.get_coherence()
		print('\nCoherence Score U_MASS: ', coherence)

	def plot_n_components_graph(self, model_name, start=2, limit=50, step=5):
		coherence_values = []
		if model_name == "--lsa":
			for num_topics in range(start, limit, step):
				model=LsiModel(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topics)
				coherencemodel = CoherenceModel(model=model, texts=self.docs, dictionary=self.dictionary, coherence='c_v')
				coherence_values.append(coherencemodel.get_coherence())
		elif model_name == "--lda":
			for num_topics in range(start, limit, step):
				model=LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topics)
				coherencemodel = CoherenceModel(model=model, texts=self.docs, dictionary=self.dictionary, coherence='c_v')
				coherence_values.append(coherencemodel.get_coherence())
		#elif model == "nmf":

		x = range(start, limit, step)
		plt.plot(x, coherence_values)
		plt.xlabel("Num Topics " + model_name)
		plt.ylabel("Coherence score")
		plt.legend(("coherence_values"), loc='best')
		plt.show()

	"""def apply_nmf_on_articles(self):
		nmf_model = Nmf(corpus=self.corpus, num_topics=50, \
			id2word=self.id2word, chunksize=self.chunksize, \
			passes=self.passes, kappa=self.kappa, \
			h_max_iter=self.iterations, eval_every=self.eval_every)
		print(nmf_model.print_topics())

		if self.topic_visualization is True:
			nmf_display = pyLDAvis.gensim.prepare(nmf_model, self.corpus, \
				self.dictionary, sort_topics=True)
			pyLDAvis.show(nmf_display)"""

	def apply_lda_on_articles(self):
		self.lda_model = LdaModel(corpus=self.corpus, id2word=self.dictionary, \
			chunksize=self.chunksize, alpha='auto', eta='auto', \
			iterations=self.iterations, num_topics=self.num_topics, \
			passes=self.passes, eval_every=self.eval_every)
		output = self.lda_model.print_topics()
		output = output.str.replace(r'[^\w\s]', '')
		output = output.str.replace(r'[0-9]', '')
		print(output)
		print("<br><br><br>")

		self.get_coherence_cv(self.lda_model)
		print("<br>")
		self.get_coherence_umass(self.lda_model)

		if self.topic_visualization is True:
			lda_display = pyLDAvis.gensim.prepare(self.lda_model, self.corpus, \
				self.dictionary, sort_topics=True)
			pyLDAvis.show(lda_display)

	def apply_lsa_on_articles(self):
		self.lsa_model = LsiModel(corpus=self.corpus, id2word=self.dictionary, \
			chunksize=self.chunksize, num_topics=self.num_topics)
		output = self.lsa_model.print_topics()
		output = output.str.replace(r'[^\w\s]', '')
		output = output.str.replace(r'[0-9]', '')
		print(output)
		print("<br><br><br>")

		self.get_coherence_cv(self.lsa_model)
		print("<br>")
		self.get_coherence_umass(self.lsa_model)

	def run_topic_modeling(self):
		for alg in self.algorithms_used:
			if alg == "--nmf":
				self.apply_nmf_on_articles()
			elif alg == "--lda":
				self.apply_lda_on_articles()
			elif alg == "--lsa":
				self.apply_lsa_on_articles()
			self.plot_n_components_graph(alg)
