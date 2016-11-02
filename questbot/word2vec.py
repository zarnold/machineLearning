#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gensim, logging
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

PATH  = "/home/arnold/Data/datasets-text/"
#source  = "news"
source  = "classiques"

with open(PATH+source+".txt","r") as fh:
  text = fh.read().lower().split("\n")


sentences = map(lambda s: re.split("[(), \-!?:.']", s),text)
model = gensim.models.Word2Vec(
  sentences,
  min_count = 50,
  sg = 0,
  size = 100
)
model.save('./models/'+source+'.wv')

#########################################################################

#model = gensim.models.Word2Vec.load('./models/'+source+'.wv')

## build array of vector
X=model.syn0

## Get coutn of each word
word = []
count= []

for k in model.vocab:
  word.append(k)
  count.append(model.vocab[k].count)

idx = np.argsort(count)
ranked_w = np.array(word)[idx]


## Get top words
limit =10000
top_w = ranked_w[-1*limit:]

x=[]
for w in top_w:
  x.append(model[w])

## Visualisation
x=np.array(x)
projection = TSNE(n_components=2, random_state=42)
p = projection.fit_transform(x) 
df  =pd.DataFrame(p,index= top_w)
df.to_csv(source+'_vec.csv')


def tellme(word):
  for w in model.most_similar(word):
    print w[0]








