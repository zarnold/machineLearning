#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import unicodedata
import pandas as pd

class MarkovBot():
	def __init__(self,corpus,deepness=4):
		self.deep=deepness
		with open(corpus,'r') as fh:
		  txt=fh.read()
		txt=unicode(txt,'utf-8')
		txt = txt.lower()
		txt = unicodedata.normalize('NFD', txt).encode('ascii', 'ignore')
		txt = re.sub('[^a-z_]', ' ', txt)
		txt=txt.split()
		df=pd.DataFrame()
		for nc in range(deepness):
		  s=pd.Series(txt[nc:])
		  df=pd.concat([df,s],axis=1)
		df['lines']=range(len(df))
		df.columns=range(deepness+1)
		self.vocabulaire=df.groupby(range(deepness)).count().reset_index()
    def talk(self,firstWord):
		print vocabulaire[(vocabulaire[0]==firstWord)]


nicolas=MarkovBot('sarkozy.txt') 
nicolas.talk('je')

