#!/usr/bin/env python
# -*- coding: utf-8 -*-

##============ apply neural net on graph
import json

class nGraph:
    def __init__(self):
       nodes=[]
    def dump(self):
      g={}
      nodes=[]
      for name in ['paul','eric','remi']:
        n={}
        n['name']=name
        n['group']=1
        nodes.append(n)
      g['nodes'] = nodes
      links=[]
      for link in[(2,0),(2,1)]:
        l={}
        l['source'] = link[0]
	l['target'] = link[1]
        links.append(l)
      g['links']=links
      print g
      with open('data.json', 'w') as outfile:
	json.dump(g, outfile)
      print('Launch some server with python -m SimpleHTTPServer and go to index.html')
       


gTest = nGraph()
gTest.dump()





