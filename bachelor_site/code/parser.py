#!/usr/bin/env python3

class Parser:
	def __init__(self, args):
		self.args = args
		self.filename = ""
		self.type_of_dataset = ""
		self.art_no = None
		self.algorithms_used = []
		self.topic_visualization = False
		self.data_labels = []

	def print_help(self):
		print("Usage: python3 bachelor_project.py [-h]", end = " ")
		print("<dataset_filename> <dataset_structure> [article number ]<algorithms>", end = " ")
		print("[topic_visualization] <labels>")
		print("Options and arguments:")
		print("-h : Prints help page. If no arguments given from the", end = " ")
		print("command line, the program will print the help page", end = "\n\n")
		print("dataset_filename: eg -> dataset.csv")
		print("The dataset file used for topic modeling", end = "\n\n")
		print("dataset_structure: --unique | --collection")
		print("Indicates if topic modeling should be done on one", end = " ")
		print("article\nfrom the dataset, or on the entire", end = " ")
		print("collection of articles on the dataset", end = "\n\n")
		print("article number: In case the tool has been called with --unique", end = " ")
		print("this argument specifies the index of the article in the dataset")
		print("algorthms: --nmf | --lda | --lsa | --enhanced_lda")
		print("The algorithms used for topic modelling.", end = " ")
		print("Enhanced LDA works only in --unique mode,", end=" ")
		print("and finds out the optimal number of topics for modeling.", end = "\n\n")
		print("topic_visualization: --visualize")
		print("This option is used in the scope of visualizing", end = " ")
		print("the topics using pLDAvis.gensim. The default value is False", end = "\n\n")
		print("labels: --labels <n> <label_names>")
		print("Labels from the dataset which contains the content used for topic modeling")


	def process_args(self):
		if len(self.args) == 1 or self.args[1] == "-h":
			self.print_help()
			return -1
		i = 1
		while i < len(self.args):
			if i == 1:
				self.filename = self.args[1]
			elif self.args[i] == "--collection":
				self.type_of_dataset = self.args[i]
			elif self.args[i] == "--unique":
				self.type_of_dataset = self.args[i]
				self.art_no = self.args[i + 1]
			elif self.args[i] == "--nmf" or self.args[i] == "--lda" or self.args[i] == "--lsa" or self.args[i] == "--enhanced_lda":
				self.algorithms_used.append(self.args[i])
			elif self.args[i] == "--visualize":
				self.topic_visualization = True
			elif self.args[i] == "--labels":
				for j in range(i + 2, i + 2 + int(self.args[i + 1])):
					self.data_labels.append(self.args[j])
			i += 1
