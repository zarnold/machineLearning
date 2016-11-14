#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import re
import unicodedata 
import tflearn
from tflearn.data_utils import *



## Get Datas

## Functions
def cleanText(txt):
  txt=unicode(txt,'utf-8')
  txt = txt.lower()
  txt = unicodedata.normalize('NFD', txt).encode('ascii', 'ignore')
  return txt

WINDOW_SIZE=5
SKIP_WINDOW=2

def makeWindows(s):
  sentence=s.split(' ')
  w=[]
  start=0
  while (start+WINDOW_SIZE)<len(sentence):
    w.append(  (sentence[start:start+WINDOW_SIZE][:-1],sentence[start:start+WINDOW_SIZE][-1])  )
    start+=SKIP_WINDOW
  return w



DATA_PATH="/home/arnold/Data/datasets-text/"
source = "classiques"

with open(DATA_PATH+source+".txt","r") as fh:
  text = fh.read()

doc=cleanText(text)
vocab=set(re.findall(r"[\w']+",doc))

rev_dict = dict(enumerate(vocab))
v_dict = { v:k for k,v in  v_dict.items() }


doc=doc.split('\n')
doc=map(lambda s: s.split('.'),doc)
doc=[item for sublist in doc for item in sublist]
splitDoc = map(makeWindows,doc)
docX=[item for sublist in splitDoc for item in sublist]

X=map(lambda t: t[0],docX)
Y=map(lambda t: t[1],docX)


# TEST. DELETE AFTER THIS LINE

net = tflearn.input_data([None, WINDOW_SIZE -1])
# Masking is not required for embedding, sequence length is computed prior to
# the embedding op and assigned as 'seq_length' attribute to the returned Tensor.
net = tflearn.embedding(net, input_dim=3, output_dim=3)
net = tflearn.lstm(net, 12, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,loss='categorical_crossentropy')


X=np.array([
	    [0,0,1],
	    [0,0,1],
	    [0,1,1],
	    [0,1,0],
	    [1,0,0]])

Y=np.array([1,1,0,0,0])


trainX = pad_sequences(X, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(Y, nb_classes=2)


# Training
model = tflearn.DNN(net, tensorboard_verbose=4)
model.fit(trainX, 
          trainY, 
          n_epoch=10,
	  run_id="lol",
          show_metric=True,
          batch_size=32)
