#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

##============ apply neural net on graph
import json
class node:
  def __init__(self, name="node", value=1,posX=1,posY=1):
    self.name=name
    self.value=value
    self.posX  =posX
    self.posY = posY

class edge:
  def __init__(self,origin,destination, weight=1):
     self.origin = origin
     self.destination = destination 


class nGraph:
    def __init__(self):
       self.nodes=[]
       self.edges=[]
    def clear(self):
       self.nodes=[]
       self.edges=[]
    def addNode(self,node):
       self.nodes.append(node)
    def addEdge(self, origin, destination):
      oIdx = self.getNodeIdx(origin)
      dIdx = self.getNodeIdx(destination)
      e=edge(oIdx,dIdx) 
      self.edges.append(e)
    def asMatrix(self):
      graphMatrix = np.zeros((len(self.nodes),len(self.nodes)))
      for e in self.edges:
        graphMatrix[e.origin,e.destination] = 1
        graphMatrix[e.destination,e.origin] = 1
      return graphMatrix
    def getNodeIdx(self,nodeName):
       for idx,val in enumerate(self.nodes):
         if nodeName == val.name:
           return idx
       return -1
    def dump(self):
      g={}
      nodes=[]
      for node in self.nodes:
        n={}
        n['name']=node.name
        n['group']=node.value
        nodes.append(n)
      g['nodes'] = nodes
      links=[]
      for e in self.edges:
        l={}
        l['source'] = e.origin
	l['target'] = e.destination
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

gTest.addEdge("Luc","Lea")
gTest.addEdge("Luc","Marc")
gTest.addEdge("Jean","Marc")
gTest.addEdge("Lea","Marc")
gTest.dump()


