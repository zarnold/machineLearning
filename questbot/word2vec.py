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

SKIP_WINDOWS=5
DIM=40
MAXLEN=100
def embed(word):
  if word in model.vocab: 
    return np.array(model[word])
  else :
    return np.zeros(DIM)


def embedSentence(sentence):
   es = map(embed,sentence)
   esp=np.zeros((MAXLEN,DIM))
   esp[:len(es)] = es[:MAXLEN]
   return esp

## This does not work -__-
def targets(sentence):
  outWord= sentence[1::SKIP_WINDOWS]
  return outWord

def samples(sentence):
  inWord = sentence.reshape(-1,SKIP_WINDOWS,DIM)
  return inWord

with open(PATH+source+".txt","r") as fh:
  text = fh.read().lower().split("\n")

## Embedding of 40 is good for grammatical structure
sentences = map(lambda s: re.split("[(), \-!?:.']", s),text)
model = gensim.models.Word2Vec(
  sentences,
  min_count = 20,
  sg = 0,
  #size = 100
  size = DIM
)

### See if it's something
for m in model.most_similar('partir'):
  print m[0]

for m in model.most_similar('qui'):
  print m[0]

for m in model.most_similar('manger'):
  print m[0]

for m in model.most_similar('joli'):
  print m[0]

## Now, transform sequence of words to sequence of embeddings
corpusSize = len(sentences)
embSent = np.array(map(embedSentence,sentences))

X=embSent
y=X[:,
## Stats


## TODO built the input and ouput
## it should predict
## the next word given the current word
# Maybe we just should use a simple linear regression ... ?
# because it s continue to continue


## model is probaby something like that
BATCH=16
yurie = Sequential()
yurie.add(LSTM(10,input_shape=(SKIP_WINDOWS,DIM)))
yurie.add(Dense(DIM))
yurie.add(Activation('relu'))
yurie.compile(loss='mean_squared_error', optimizer='sgd')


## So this should work
# Here an exemple for shape, with a linear combination
# So we should get a good score
X = np.random.random((30000,SKIP_WINDOWS,DIM))
y=X[:,0,:]+2*X[:,1,:]-1*X[:,2,:]+1.4*X[:,3,:]+2.3*X[:,4,:]
yurie.fit(X,y, batch_size=8,nb_epoch=300)

X_test=np.random.random((3,SKIP_WINDOWS,DIM))
pred = yurie.predict(X_test)
y=X_test[:,0,:]+2*X_test[:,1,:]-1*X_test[:,2,:]+1.4*X_test[:,3,:]+2.3*X_test[:,4,:]

