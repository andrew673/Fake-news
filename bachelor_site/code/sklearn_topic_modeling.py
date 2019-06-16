#!/usr/bin/env python3
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV


class TopicModelingClass:
	def __init__(self, docs, art_no, algorithms_used):
		self.docs = docs
		self.art_no = int(art_no)
		self.algorithms_used = algorithms_used
		self.no_features = 1000
		self.no_topics = 3
		self.no_top_words = 10

	def display_topics(self, model, model_name, feature_names, best_no_topics):
		if best_no_topics is not None:
			for i in range(best_no_topics.value()):
				topic = model.components_[1]
				print("Topic %d:" % (i))
				print(" ".join([feature_names[j]
					for j in topic.argsort()[:-self.no_top_words - 1:-1]]))
			return
		print("Article %d on %s" % (self.art_no, model_name), end = "<br>")
		for topic_idx, topic in enumerate(model.components_):
			print("Topic %d:" % (topic_idx))
			print(" ".join([feature_names[i]
				for i in topic.argsort()[:-self.no_top_words - 1:-1]]), end = "<br>")

	def apply_nmf_on_articles(self):
		doc = self.docs[self.art_no]
		# NMF is able to use tf-idf
		tfidf_vectorizer = TfidfVectorizer(max_features=self.no_features, stop_words='english')
		tfidf = tfidf_vectorizer.fit_transform(doc)
		tfidf_feature_names = tfidf_vectorizer.get_feature_names()

		# Run NMF
		nmf = NMF(n_components=self.no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='random').fit(tfidf)

		self.display_topics(nmf, "NMF", tfidf_feature_names, None)
		print("<br><br><br>")

	def apply_lda_on_articles(self):
		doc = self.docs[self.art_no]
		# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
		tf_vectorizer = CountVectorizer(max_features=self.no_features, stop_words='english')
		tf = tf_vectorizer.fit_transform(doc)
		tf_feature_names = tf_vectorizer.get_feature_names()

		# Run LDA
		lda = LatentDirichletAllocation(n_components=self.no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

		self.display_topics(lda, "LDA", tf_feature_names, None)
		print("<br><br><br>")

	def apply_enhanced_lda_on_articles(self):
		doc = self.docs[self.art_no]
		# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
		tf_vectorizer = CountVectorizer(max_features=self.no_features, stop_words='english')
		tf = tf_vectorizer.fit_transform(doc)
		tf_feature_names = tf_vectorizer.get_feature_names()

		lda = LatentDirichletAllocation(n_components=self.no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)

		search_params = {'n_components': [5, 10, 15, 20, 25], 'learning_decay': [.5, .7, .9]}
		model = GridSearchCV(lda, param_grid=search_params).fit(tf)
		nt = model.best_params_.get('no_components')
		best_lda = model.best_estimator_

		self.display_topics(best_lda, "enhanced LDA", tf_feature_names, nt)
		print("<br><br><br>")

	def apply_lsa_on_articles(self):
		doc = self.docs[self.art_no]
		# LSA
		tf_vectorizer = CountVectorizer(max_features=self.no_features, stop_words='english')
		tf = tf_vectorizer.fit_transform(doc)
		tf_feature_names = tf_vectorizer.get_feature_names()

		# SVD to reduce dimensionality:
		svd = TruncatedSVD(n_components=self.no_topics, algorithm='randomized', n_iter=10).fit(tf)

		self.display_topics(svd, "LSA", tf_feature_names, None)
		print("<br><br><br>")

	def run_topic_modeling(self):
		for alg in self.algorithms_used:
			if alg == "--nmf":
				self.apply_nmf_on_articles()
			elif alg == "--lda":
				self.apply_lda_on_articles()
			elif alg == "--lsa":
				self.apply_lsa_on_articles()
			elif alg == "--enhanced_lda":
				self.apply_enhanced_lda_on_articles()
