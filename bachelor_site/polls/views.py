# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

import os
import time


def index(request):

	if request.method == 'GET':
		print("hello")
	else:
		filename = request.POST.get("filename", '')
		single = request.POST.get("single", '')
		collection = request.POST.get("collection", '')
		art_no = request.POST.get("art_no", '')
		nmf = request.POST.get("nmf", '')
		lda = request.POST.get("lda", '')
		lsa = request.POST.get("lsa", '')
		enhanced_lda = request.POST.get("enhanced_lda", '')
		labels_no = request.POST.get("labels_no", '')
		labels = request.POST.get("labels", '')
		visualize = request.POST.get("visualize", '')

		if single.startswith( 'on' ):
			single = '--unique'
		else:
			single = ''
		if collection.startswith( 'on' ):
			collection = '--collection'
		else:
			collection = ''
		if nmf.startswith( 'on' ):
			nmf = '--nmf'
		else:
			nmf = ''
		if enhanced_lda.startswith( 'on' ):
			enhanced_lda = '--enhanced_lda'
		else:
			nmf = ''
		if lda.startswith( 'on' ):
			lda = '--lda'
		else:
			lda = ''
		if lsa.startswith( 'on' ):
			lsa = '--lsa'
		else:
			lsa = ''
		if visualize.startswith( 'on' ):
			visualize = '--visualize'
		else:
			visualize = ''
		command = "python3 /home/andrew/Desktop/Licenta/Fake-news/bachelor_site/code/bachelor_project.py"
		command += " "
		command += filename
		command += " "
		command += single
		command += " "
		command += collection
		command += " "
		command += art_no
		command += " "
		command += nmf
		command += " "
		command += lda
		command += " "
		command += enhanced_lda
		command += " "
		command += lsa
		command += " "
		command += "--labels "
		command += labels_no
		command += " "
		command += labels
		command += " > /home/andrew/Desktop/Licenta/Fake-news/bachelor_site/code/output.txt"
		path = "/home/andrew/Desktop/Licenta/Fake-news/bachelor_site/code/output.txt"
		os.system(command)
		file = open(path,'r')
		response_text = file.read()
		return HttpResponse(response_text)

	return render(request, 'home.html')
	#return HttpResponse("Hello, world. You're at the polls index.")
