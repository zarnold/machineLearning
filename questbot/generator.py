from __future__ import absolute_import, division, print_function

import os
import pickle

import tflearn
from tflearn.data_utils import *

DATA_PATH="/home/arnold/Data/datasets-text/"
path = "classiques.txt"

maxlen = 25

char_idx = None

X, Y, char_idx =   textfile_to_semi_redundant_sequences(DATA_PATH + path, seq_maxlen=maxlen, redun_step=3)



g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='classiques')
#u = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='classiques')


def talk():
  seed = random_sequence_from_textfile(DATA_PATH + path, maxlen)
  ph=m.generate(2000,temperature=0.7, seq_seed=seed)
  print(ph)
  with open("Flauxbert.txt","a") as fh:
    fh.write('========================\n')
    fh.write(ph+'\n')
  seed="Je pense que la vie est une "[:maxlen]
  ph=m.generate(200,temperature=0.7, seq_seed=seed)
  with open("Flauxbert.txt","a") as fh:
    fh.write('===\n')
    fh.write(ph+'\n')



for i in range(50):
    m.fit(X, Y, validation_set=0.1, batch_size=128, n_epoch=1, run_id='classiques')
    talk()




