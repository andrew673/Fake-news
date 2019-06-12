#!/usr/bin/env python3
import sys

import parser as ps
import data_processor as dp
import topic_modeling as tm


def main():
	parser = ps.Parser(sys.argv)
	if(parser.process_args() == -1):
		return

	data_processor = dp.DataProcessor(parser.filename, parser.data_labels)
	data_processor.import_content()

	topic_modeling_class = tm.TopicModelingClass(parser, data_processor)
	topic_modeling_class.run()

if __name__== "__main__":
  main()