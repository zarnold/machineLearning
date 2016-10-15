#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gensim
import re


PATH  = "/home/arnold/Data/datasets-text/word2vec/"
source  = "news"

with open(PATH+source+".txt","r") as fh:
  text = fh.read().split("\n")


sentences = map(lambda s: re.split("[, \-!?:.']", s),text)
model = gensim.models.Word2Vec(
  sentences,
  min_count = 5,
  sg = 0,
  size = 100
)

for w in model.vocab:
  print w

with open("./models/"+source+".wv","w") as fname:
  model.save(fname)



