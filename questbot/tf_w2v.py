#!/usr/bin/env python
# -*- coding: utf-8 -*-

from  collections import Counter
from  collections import deque
import math
import re
import os
import random
import zipfile

import numpy as np
import tensorflow as tf



PATH="/home/arnold/Data/datasets-text/"

class WvBot:
  def __init__(self,path):
    ## Parameters of the model 
    self.vocabulary_size= 10000 
    # Keep only a small bunch of word, the most common
    # replace other with UNK (unknown)
    with open(path, 'r') as f:
      rawtxt = f.read()
    ## build a list of all words in the text
    self.text=re.findall(r"[\w']+",rawtxt.decode('utf8'),re.UNICODE)
    ## First word is UNK 
    count=[['UNK',1]]
    ## count occurence of each most frequent word
    count.extend(Counter(self.text).most_common(self.vocabulary_size - 1))
    ## build an index
    d = dict()
    for w,c in count:
      d[w] = len(d)
    ## encode the text with number
    ## the text is rewritten with index of the word
    self.data=list()
    unk_count = 0
    for w in self.text :
      if w in d :
	index = d[w]
      else :
	index = 0
	unk_count += 1
      self.data.append(index)
    count[0][1] = unk_count
    self.reverse_d = dict(zip(d.values(), d.keys())) 
    self.count = count
    self.dictionnary = d
    self.data_index = 0
  def make_batch(self,batch_size, num_skip, skip_window) :
    assert batch_size % num_skip == 0
    assert num_skip <= 2 * skip_window
    batch = np.ndarray(shape = (batch_size), dtype = np.int32)
    label = np.ndarray(shape = (batch_size,1), dtype = np.int32)
    ## Largeur de la fenetre = skip avant, skip apres et target
    span = 2*skip_window + 1
    buffer = deque(maxlen=span)
    ## build a buffer in the current place
    for _ in range(span):
      ## keep trace of where we are and go circular
      buffer.append(self.data[self.data_index])
      self.data_index = (self.data_index + 1) % len(self.data)
    for i in range(batch_size // num_skip ) :
      target = skip_window
      targets_to_avoid = [skip_window]
      for j in range(num_skip) : 
	while target in targets_to_avoid :
	  target = random.randint(0,span -1)
	targets_to_avoid.append(target)
	batch[i * num_skip + j] = buffer[skip_window]
	label[i * num_skip + j,0] = buffer[target]
      buffer.append(self.data[self.data_index])
      self.data_index  =(self.data_index +1) % len(self.data)
    return batch,label



wb = WvBot(PATH+'antioch.txt')
batch,labels = wb.make_batch(12,3,6)

for i in range(64):
  print(batch[i], wb.reverse_d[batch[i]],
      '->', labels[i, 0], wb.reverse_d[labels[i, 0]])

