#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import math
import os
import random
import zipfile

import numpy as np
import tensorflow as tf



PATH="/home/arnold/Data/datasets-text/"

class WvBot:
  def __init__(self,path):
    with open(path, 'r') as f:
      rawtxt = f.read()
    self.text=re.findall(r"[\w']+",rawtxt.decode('utf8'),re.UNICODE)

         
   


wb = WvBot(PATH+'antioch.txt')

