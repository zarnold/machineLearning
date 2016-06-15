#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import tensorflow as tf

# more digit = more time but better learning
N_DIGITS	    = 16
SAMPLE_SIZE	    = 2**N_DIGITS
N_ROUND		    = 30000
BATCH_SIZE	    = 128




# d'une maniere generale, ce code permet de voir si 
# un réseau de neurone est capable d'apprendre une certaine propriété des nombres
# la propriété est fabriquée a la main  dans la fonction eratosthene
# et peut etre n'importe quoi
# Pour les propriétés telles que "divisible par 2 " ou "divisible par 5"
# les réseaux s'en sortent tres bien
# ( essayez en decommentant la ligne) concernéé
# pour la primalité, c'est moins trivial.

# Ce code permet aussi de voir la démarche générale
# quand on veut créer un reseau de neurone (tres basique )
# avec Tensorflow, l'outil de google.

# le code est volontairement simple pour etre didactique.


# Note de triche : on test la qualité de notre modele sur l'echantillon 
# surlequel on a entrainé le reseau
# C'est donc completement overfitee et abusé.



# pas la methode la plus rapide pour savoir si un nombre est premier mais la plus courte en code
# d'autant plus qu'elle permet de sauvegarder les résultats

# Note : pour des raisons de simplicité, cette fonction retourne
# la primalité en indice 0 et la non primalité en indice 1
def eratosthene(sample_size = 50000) :
  # le crible d'eratostrucmescouilles commence avec des zeros partout
  erth = np.zeros(sample_size)
  # pour tous les entiers entre 0 et n/2
  for i in range(sample_size/2):
    if(i%100 == 0) :  print i
    # pour tous les multiples de i, on met un "1" dans le tableau
    # qui n'est donc pas un nombre premier
    if i>1 :
      erth[range(sample_size)[i*2::i]] = 1
      #decommente pour apprendre un cas simple : les divisions par 3
      #erth[range(sample_size)[::3]] = 1
  return erth

def is_prime(i,crible):
  return (1-crible[i],crible[i])

## return un tableau de bit de l'encodage de i en binaire
def binary_encode(i, num_digits = N_DIGITS):
  return np.array([ i >> d & 1 for d in range(num_digits)])

def binary_decode(i):
  out = 0
  for b in i[::-1]:
    out = (out << 1) | b
  return out

## Comment initialiser les reseaux de neurones ?
## l'eternel debat. Moi perso, c'est à la gueule
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


####################################################
# init
print("Making primes")
crible = eratosthene(SAMPLE_SIZE)

# autant sauver les primes deja calculés tan tqu'a faire
idx = range(0,SAMPLE_SIZE)
sf = np.vstack([idx,crible]).astype('int')
np.savetxt('primes.csv', sf.T,fmt="%d",delimiter=',')

## maintenant, le réseau de neurone
### on construit l'entrainement

# On construit les données d'entrainement
# en binarisant tous les entiers
X = np.array(map(binary_encode,range(SAMPLE_SIZE)))

# Verifier 
# X.shape


# et on indique dans un autre tableau
# ce qu'est sensé répondre le réseau de neurone
# aka si un nombre est premier ou pas 

print("set up")
Y = np.array([ is_prime(i,crible) for i in range(SAMPLE_SIZE)])


# OK. Maintenant, construisons notre réseau de neurone avec tensorflow
# le truc de google

# Combien de neurones ?
# on pourra toujours faire du Random Grid search 
NUM_HIDDEN = 18

# Ca c'est du formalisme de tensorflow
# pour construie le reseau.  ON s'en fout
# c'est l'input et l'output
# en entrée on a 32 float et en sortie 1 seul
# Pourquoi des float ?
# parce que le reseau va apprendre des probabilité
#  et non pas des certitudes
I = tf.placeholder("float", [None, N_DIGITS])
O = tf.placeholder("float", [None, 2])

# w_h represente les poids des connexions neuronales. C'est ca qui va etre appris
w_h = init_weights([N_DIGITS, NUM_HIDDEN])
# w_o est la proba d'etre un nombre premier
w_o = init_weights([NUM_HIDDEN, 2])

# et là, la partie interessante donc, ou l'on 
# fabrique la structure du reseau
# de neurone que l'on pense la plus adaptée pour notre probleme
# ensuite, on le laissera grandir tout seul
# ici, le reseau n'a qu'une couche cachée, ce qui est sans doute un peu 
# léger

def model(X, w_h, w_o):
  # la base : la sortie du réseau est 
  # l'entree une fois passé dans le reseau de neurone  
  # c'est une simple multiplication de matrices
  z=tf.matmul(X, w_h)
  # ensuite, comme c'est un reseau de neurone
  # on colle l'équivalent informatique d'une neurone qui s'ctive
  # ici une fonction qui s'appelle relu. Il y en a pleins d'autres possible mais 
  # celle la est cool
  h = tf.nn.relu(z)
  # et enfin, la probabilité d'etre en nb premier en sortie
  # est fonction de comment les neurones des couches cachées se sont activées.
  # encore une fois une simple matrice de poids
  return tf.matmul(h,w_o)
  
# voici notre bebe cerveau. C'est lui
# appelons le Ness :

ness = model(I, w_h,w_o)


# maintenant ness va apprendre mais 
# pour ca il lui faut un moyen de lui indiquer 
# s il s'est trompé et de combien
# on appelel ca une fonction de cout
# ici on prend une softmax cross entropy
# qui est standard poru ce genre de chose

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(ness, O))

# et enfin, on lui dit que pour corriger ses erreurs
# il doit faire une descente de gradient
# c'est a dire modifier son cerveau de maniere a 
# atteindre le minimum de la fonction de cout 
# c'est aussi relativement standard
# je ne connais pas d'autres méthodes en fait

train = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# pour simplifier, voyons quand meme si c'est plutot un prime ou plutot pas
pred = tf.argmax(ness, 1)

# allez c'est partie.
# on a tout préparé, on lance l'entrainement

## note: on fait gaffe a bien apprendre sur une partie
## et en cacher une autre
# sinon il ne s'agit asp d'apprentissage mais de mémorisation
# c'est a dire d'overfitting

split = int(SAMPLE_SIZE/2)
Xtr = X[:split]
Ytr = Y[:split]
Xte = X[split:]
Yte = Y[split:]

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for c_round in range(N_ROUND):
      p = np.random.permutation(range(len(Xtr)))
      Xtr, Ytr = Xtr[p], Ytr[p]
      print("***********  round %d ****** "%c_round)
      for start in range(0, len(Xtr), BATCH_SIZE):
	end = start + BATCH_SIZE
	sess.run(train, feed_dict={I: Xtr[start:end],O: Ytr[start:end]})
      print(c_round, np.mean(np.argmax(Ytr, axis=1) == sess.run(pred, feed_dict={I: Xtr, O: Ytr})))
      print("** Pediction for never seen number")
      print(c_round, np.mean(np.argmax(Yte, axis=1) == sess.run(pred, feed_dict={I: Xte})))
    # Une fois entrainé, voyons ce que ness predit avec des nouveaux chiffres
    ness_said = sess.run(pred, feed_dict = {I: Xtr})
    ness_try  = sess.run(pred, feed_dict = {I: Xte})

print np.mean(ness_said)

df = pd.DataFrame({'nombre':map(binary_decode,Xtr), 'proposition':ness_said.astype('int'),'verite':Ytr[:,1].astype('int')})
df.sort('nombre').to_csv('ness_proposition_for_prime.csv', index=False)

df = pd.DataFrame({'nombre':map(binary_decode,Xte), 'proposition':ness_said.astype('int'),'verite':Yte[:,1].astype('int')})
df.sort('nombre').to_csv('ness_proposition_for_unknown_prime.csv', index=False)

##  Maintenant, testons avec des nombres jamais vu
