#!/usr/bin/env python
# -*- coding: utf-8 -*-

##============ apply neural net on graph
import json
class node:
  def __init__(self, name="node", value=1,posX=1,posY=1):
    self.name=name
    self.value=value
    self.posX  =posX
    self.posY = posY


class nGraph:
    def __init__(self):
       self.nodes=[]
       self.links=[]
    def clear(self):
       self.nodes=[]
       self.links=[]
    def addNode(self,node):
       self.nodes.append(node)
    def getNodeIdx(self,nodeName):
       for idx,val in enumerate(self.nodes):
         if nodeName == val.name:
           return idx
       return -1
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
gTest.addNode(node("Luc"))
gTest.addNode(node("Jean"))
gTest.addNode(node("Lea"))
gTest.addNode(node("Marc"))




