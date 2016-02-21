#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import unicodedata

class MarkovBot():
  def __init__(self,corpus):
      fh=open(corpus)
      txt=fh.read()
      txt=unicode(txt,'utf-8')
      txt = txt.lower()
      txt = unicodedata.normalize('NFD', txt).encode('ascii', 'ignore')
      txt = re.sub('[^a-z_]', ' ', txt)
      txt=txt.split()
      fh.close()
      self.words=txt



nicolas=MarkovBot('sarkozy.txt') 

print nicolas.words[:300]
